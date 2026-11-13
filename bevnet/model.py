"""Image-based collaborative network: images + camera poses -> a map of the boxes on the floor.

  1. image encoder (every robot, the same weights): a small CNN, or an ImageNet-pretrained ResNet-18
     (--encoder resnet18), features at 1/4 of the image size
  2. lift to the floor (every robot): the features are read at the pixels where each floor cell, lifted to the
     heights of bevnet/geometry.py, falls in the image (grid_sample with the grids of the data loader), joined
     with the direction and distance of the cell from the camera, and turned by a 1 x 1 network into the
     robot's MESSAGE: --message channels per floor cell it sees (zero elsewhere)
  3. fusion (the team): per cell, the mean and the maximum of the messages of the robots that see it, and how
     many see it
  4. floor decoder: a small U-Net on the 80 x 80 grid -> the probability that a box centre lies in each cell
     and the sub-cell offset of that centre; the boxes are the local maxima above a threshold chosen on the
     validation set (bevnet/geometry.decode)
Trained with the focal loss of CenterNet on a Gaussian around each true box and an L1 loss on the offsets.
The message is what a robot would send to the others: its size is message x the cells it sees.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ImageEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(conv(3, 32, 2), conv(32, 32), conv(32, 64, 2), conv(64, 64),
                                 conv(64, channels, 1, 2), conv(channels, channels, 1, 4))

    def forward(self, x):
        return self.net(x)


class ResNetEncoder(nn.Module):
    """ImageNet-pretrained ResNet-18 (torchvision): layer1 (1/4), layer2 (1/8) and layer3 (1/16) brought to
    1/4 of the image and summed, so the features sit where the small encoder's do. Features learned on
    real photographs carry over to real frames better than ones learned on rendered images only."""

    def __init__(self, channels=64, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.p1, self.p2, self.p3 = nn.Conv2d(64, channels, 1), nn.Conv2d(128, channels, 1), nn.Conv2d(256, channels, 1)
        self.out = conv(channels, channels)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        size = c1.shape[-2:]
        f = self.p1(c1) + F.interpolate(self.p2(c2), size=size, mode="bilinear", align_corners=False) + \
            F.interpolate(self.p3(c3), size=size, mode="bilinear", align_corners=False)
        return self.out(f)


class FloorDecoder(nn.Module):
    """U-Net on the floor grid: c channels at 80 x 80, 2c at 40 x 40, 4c at 20 x 20 (dilated)."""

    def __init__(self, cin, c=32):
        super().__init__()
        self.e1 = nn.Sequential(conv(cin, c), conv(c, c))
        self.e2 = nn.Sequential(conv(c, 2 * c, 2), conv(2 * c, 2 * c))
        self.e3 = nn.Sequential(conv(2 * c, 4 * c, 2), conv(4 * c, 4 * c, 1, 2))
        self.u2 = conv(6 * c, 2 * c)
        self.u1 = conv(3 * c, c)
        self.heat = nn.Conv2d(c, 1, 1)
        self.off = nn.Conv2d(c, 2, 1)
        nn.init.constant_(self.heat.bias, -math.log(9.0))       # start at p = 0.1 (CenterNet)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.u2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.u1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.heat(d1), self.off(d1)


class BevNet(nn.Module):
    """The image network: the same encoder on every robot's image, the features lifted to the floor grid at
    `heights` heights and turned into a message of `message` channels per cell, the messages of the robots of
    a scene fused cell by cell (mean, maximum, number of robots that see the cell), and a decoder on the floor
    that returns the map of box centres and the offsets inside the cells."""

    def __init__(self, channels=64, message=16, heights=3, encoder="small", pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(channels, pretrained) if encoder == "resnet18" else ImageEncoder(channels)
        self.lift = nn.Sequential(nn.Conv2d(channels * heights + 4, channels, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, message, 1))
        self.decoder = FloorDecoder(2 * message + 1, channels // 2)

    def messages(self, images, grids, valid, view):
        """Per robot image: (message (N, M, G, G), cells it sees (N, 1, G, G))."""
        f = self.encoder(images)
        samples = [F.grid_sample(f, grids[:, k], mode="bilinear", padding_mode="zeros", align_corners=False)
                   * valid[:, k, None].float() for k in range(grids.shape[1])]
        seen = valid.any(1, keepdim=True).float()
        x = torch.cat(samples + [view * seen, seen], 1)
        return self.lift(x) * seen, seen

    def forward(self, images, grids, valid, view, scene, n_scenes):
        m, seen = self.messages(images, grids, valid, view)
        S, (_, M, G, _) = n_scenes, m.shape
        total = m.new_zeros(S, M, G, G).index_add_(0, scene, m)
        count = seen.new_zeros(S, 1, G, G).index_add_(0, scene, seen)
        mean = total / count.clamp(min=1.0)
        big = torch.where(seen.bool(), m, torch.full_like(m, -1e4))
        mx = torch.stack([big[scene == s].max(0).values for s in range(S)])
        mx = torch.where(count > 0, mx, torch.zeros_like(mx))
        return self.decoder(torch.cat([mean.float(), mx.float(), count.float() / 4.0], 1))


def losses(heat_logits, off, target_heat, target_off, mask):
    """CenterNet focal loss on the box-centre map and L1 on the offsets of the box cells."""
    p = torch.sigmoid(heat_logits[:, 0]).clamp(1e-4, 1 - 1e-4)
    pos = target_heat.eq(1).float()
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p ** 2 * (1 - target_heat) ** 4 * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    focal = (pos_loss.sum() + neg_loss.sum()) / n
    l1 = ((off - target_off).abs() * mask[:, None]).sum() / (2 * mask.sum().clamp(min=1.0))
    return focal, l1
