"""End-to-end network: the camera images and poses of the robots -> a map of the boxes on the floor.

  1. image encoder (every robot, the same weights): a small CNN, features at 1/4 of the image size
  2. lift to the floor (every robot): the features are read at the pixels where each floor cell, lifted to the
     heights of e2e/geometry.py, falls in the image (grid_sample), joined with the direction and distance of the
     cell from the camera, and turned by a 1 x 1 network into the robot's node state for that cell
  3. fusion among the robots (graph neural network): for every floor cell, a complete graph on the robots that
     see it. LAYERS rounds of attention message passing: each robot weighs the states of the others by their
     content and by the geometry of the pair (the angle between the two lines of sight at the cell and the two
     distances: two robots that see a cell from different directions can triangulate it, two on the same line
     cannot). The same weights on every cell and for any number of robots; a robot that does not see a cell takes
     no part in its graph. Readout per cell: mean and maximum of the robot states, and how many robots see it.
  4. floor decoder: a small U-Net on the 80 x 80 grid -> the probability that a box centre lies in each cell and
     the sub-cell offset of that centre; the boxes are the local maxima above a threshold chosen on the
     validation set (e2e/geometry.decode)
Trained with the focal loss of CenterNet on a Gaussian around each true box and an L1 loss on the offsets.

The graph is over the robots, not over the cells: message passing between neighbouring cells of a regular grid
would be a convolution, which is what the floor decoder is.
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


PAIR_FEATURES = 5          # cos, |sin| and cos^2 of the angle between the lines of sight, the two distances


class RobotGraphLayer(nn.Module):
    """One round of attention message passing among the robots that see a cell, on every cell at once.
    x (S, R, P, C): state of robot slot r of scene s at cell p; seen (S, R, P) bool; pair (S, P, R, R, 5)."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.h, self.d = heads, c // heads
        self.qkv = nn.Linear(c, 3 * c)
        self.geo = nn.Linear(PAIR_FEATURES, heads)                # geometry of the pair -> attention bias per head
        self.out = nn.Linear(c, c)
        self.norm1, self.norm2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, 2 * c), nn.ReLU(inplace=True), nn.Linear(2 * c, c))

    def forward(self, x, seen, pair):
        S, R, P, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(S, R, P, 3, self.h, self.d).unbind(3)
        logits = torch.einsum("sipht,sjpht->sphij", q, k) / math.sqrt(self.d)
        logits = logits + self.geo(pair).permute(0, 1, 4, 2, 3)               # (S, P, H, R, R)
        sender = seen.permute(0, 2, 1)[:, :, None, None, :]                    # (S, P, 1, 1, R)
        logits = logits.float().masked_fill(~sender, -1e4)
        w = torch.softmax(logits, -1).to(v.dtype)
        msg = torch.einsum("sphij,sjpht->sipht", w, v).reshape(S, R, P, C)
        x = x + self.out(msg)
        x = x + self.ffn(self.norm2(x))
        return x * seen[..., None].to(x.dtype)


class E2ENet(nn.Module):
    """Images and camera poses of a team -> map of box centres. See the module docstring."""

    def __init__(self, channels=64, state=32, heights=3, layers=2, heads=4):
        super().__init__()
        self.encoder = ImageEncoder(channels)
        self.lift = nn.Sequential(nn.Conv2d(channels * heights + 4, channels, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, state, 1))
        self.layers = nn.ModuleList([RobotGraphLayer(state, heads) for _ in range(layers)])
        self.decoder = FloorDecoder(2 * state + 1, channels // 2)

    def node_states(self, images, grids, valid, view):
        """Per robot image: (state (N, C, G, G), cells it sees (N, 1, G, G))."""
        f = self.encoder(images)
        samples = [F.grid_sample(f, grids[:, k], mode="bilinear", padding_mode="zeros", align_corners=False)
                   * valid[:, k, None].to(f.dtype) for k in range(grids.shape[1])]
        seen = valid.any(1, keepdim=True).to(f.dtype)
        x = torch.cat(samples + [view.to(f.dtype) * seen, seen], 1)
        return self.lift(x) * seen, seen

    def forward(self, images, grids, valid, view, scene, n_scenes):
        h, seen = self.node_states(images, grids, valid, view)
        N, C, G, _ = h.shape
        S, P = n_scenes, G * G
        counts = torch.bincount(scene, minlength=S)
        R = int(counts.max())
        slot = torch.arange(N, device=h.device) - (torch.cumsum(counts, 0) - counts)[scene]
        x = h.new_zeros(S, R, P, C)
        x[scene, slot] = h.flatten(2).transpose(1, 2)
        sm = torch.zeros(S, R, P, dtype=torch.bool, device=h.device)
        sm[scene, slot] = seen.flatten(1) > 0
        vw = view.new_zeros(S, R, P, 3)
        vw[scene, slot] = view.flatten(2).transpose(1, 2)
        pair = pair_features(vw)
        for layer in self.layers:
            x = layer(x, sm, pair)
        smf = sm[..., None].to(x.dtype)
        count = smf.sum(1)                                                     # (S, P, 1)
        mean = x.sum(1) / count.clamp(min=1.0)
        mx = torch.where(sm[..., None], x, torch.full_like(x, -1e4)).max(1).values
        mx = torch.where(count > 0, mx, torch.zeros_like(mx))
        fused = torch.cat([mean, mx, count / 4.0], -1).transpose(1, 2).reshape(S, 2 * C + 1, G, G)
        return self.decoder(fused.float())


def pair_features(vw):
    """vw (S, R, P, 3): unit direction camera -> cell (2) and distance / MAX_RANGE_M. Returns (S, P, R, R, 5)."""
    d = vw[..., :2].permute(0, 2, 1, 3)                                        # (S, P, R, 2)
    r = vw[..., 2].permute(0, 2, 1)                                            # (S, P, R)
    cos = torch.einsum("spic,spjc->spij", d, d)
    sin = (d[..., :, None, 0] * d[..., None, :, 1] - d[..., :, None, 1] * d[..., None, :, 0]).abs()
    ri = r[..., :, None].expand_as(cos)
    rj = r[..., None, :].expand_as(cos)
    return torch.stack([cos, sin, cos * cos, ri, rj], -1)


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
