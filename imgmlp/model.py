"""ImageMLP + detection-based GNN.

Inputs of one scene, and nothing else:
  image         per robot, the whole grey frame (H x W), only resized
  measurement   per measured object: distance rho and bearing theta (any sensor)
  pose          per robot: x, y, psi

  ImageMLP            flattened image (standardized per image) -> H*W -> 256 -> 64 -> k features z (no CNN, so the
                      cost per robot does not depend on the scene and the team can grow)
  report node         one per measured object: [rho, theta, x, y, cos psi, sin psi] of its robot (positions in a frame
                      centred on the robots, in units of RING_SCALE_M)
  message passing     `hops` rounds (residual)
  read-out            same-box probability per pair of reports, false-measurement probability per report, box
                      position per report; decode() of gnn/model.py groups the reports into boxes, a box sits at the
                      mean position of its reports

Two ways to bring the image in (graph=):
  A   z of the robot is appended to every report node of that robot; the graph has report nodes only. A robot
      without reports has no node, so its image is not used.
  B   heterogeneous graph: the report nodes carry the measurement only; one robot node per robot carries its pose
      and z. In every hop: robot -> its reports, report <-> report (pairs), reports -> their robot, robot <-> robot
      (robots closer than robot_gate). Every robot has a node, also one that measured nothing, so its image still
      reaches the others. The output is still per report (detection based).

Two geometry modes:
  fixed   a fixed, not trained, layer turns (pose, rho, theta) into the floor position of every measurement; the
          GNN uses it to choose the pairs (closer than the gate) and their offsets (along / across the line of
          sight), and predicts the box position as that position plus a learned correction; robot-robot edges join
          robots closer than robot_gate and carry their relative pose. Nothing hand-made enters the nodes.
  strict  no geometry at all: every pair of reports and every pair of robots of a scene is linked, the only pair
          feature is "same robot", robot edges carry nothing, and the box position is predicted from the node alone.

k = 0 drops the image: the measurements alone (the ablation that tells what the image adds).
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gnn.model import PAIR_FEATURES, mlp, pair_features, segment_mean
from imgmlp.measure import detection_positions

RING_SCALE_M = 1.3
NODE_FEATURES = 6                    # rho, theta, x, y, cos psi, sin psi
ROBOT_FEATURES = 4                   # x, y, cos psi, sin psi
PAIR_WIDTH = dict(fixed=PAIR_FEATURES, strict=1)
ROBOT_EDGE_WIDTH = dict(fixed=5, strict=0)
GRAPHS = ("A", "B")


# ====================================================================
# Encoding (numpy)
# ====================================================================

def encode_scene(s, geometry="fixed", gate=1.0, rotation=0.0, robot_gate=4.0):
    """numpy arrays of one scene, in a frame centred on the robots and rotated by `rotation`."""
    poses = np.asarray(s["poses"], float).reshape(-1, 3)
    center = poses[:, :2].mean(axis=0)
    c, si = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -si], [si, c]])
    rxy = (poses[:, :2] - center) @ R.T
    ryaw = poses[:, 2] + rotation
    r_of = np.asarray(s["det_robot"], int)
    n, nr = len(r_of), len(poses)
    node = np.column_stack([s["distance"], s["bearing"], rxy[r_of] / RING_SCALE_M,
                            np.cos(ryaw[r_of]), np.sin(ryaw[r_of])]).reshape(n, NODE_FEATURES)
    robot = np.column_stack([rxy / RING_SCALE_M, np.cos(ryaw), np.sin(ryaw)]).reshape(nr, ROBOT_FEATURES)
    local = dict(s, poses=np.column_stack([rxy, ryaw]))
    geo = detection_positions(local).reshape(n, 2)                  # fixed layer (also used for decoding)

    ii, jj = np.triu_indices(n, k=1)
    ri, rj = [x.ravel() for x in np.meshgrid(np.arange(nr), np.arange(nr), indexing="ij")]
    ri, rj = ri[ri != rj], rj[ri != rj]                             # directed robot edges ri -> rj
    if geometry == "fixed":
        near = np.hypot(*(geo[jj] - geo[ii]).T) <= gate if len(ii) else np.zeros(0, bool)
        ii, jj = ii[near], jj[near]
        ray = ryaw[r_of] - np.asarray(s["bearing"], float)
        u = np.stack([np.cos(ray), np.sin(ray)], axis=1).reshape(-1, 2)
        p_ij = pair_features(geo, u, r_of, ii, jj).reshape(-1, PAIR_FEATURES)
        p_ji = pair_features(geo, u, r_of, jj, ii).reshape(-1, PAIR_FEATURES)
        d = rxy[ri] - rxy[rj]
        keep = np.hypot(d[:, 0], d[:, 1]) <= robot_gate
        ri, rj, d = ri[keep], rj[keep], d[keep]
        cj, sj = np.cos(-ryaw[rj]), np.sin(-ryaw[rj])               # offset of ri seen from rj, in rj's frame
        dyaw = ryaw[ri] - ryaw[rj]
        r_feat = np.column_stack([(cj * d[:, 0] - sj * d[:, 1]) / RING_SCALE_M, (sj * d[:, 0] + cj * d[:, 1]) / RING_SCALE_M,
                                  np.cos(dyaw), np.sin(dyaw), np.hypot(d[:, 0], d[:, 1]) / RING_SCALE_M])
    elif geometry == "strict":
        p_ij = (r_of[ii] == r_of[jj]).astype(float).reshape(-1, 1)
        p_ji = p_ij.copy()
        r_feat = np.zeros((len(ri), 0))
    else:
        raise ValueError(f"unknown geometry {geometry}")

    label = np.asarray(s.get("label", np.full(n, -1)), int)
    truth = (np.asarray(s["truth"], float).reshape(-1, 2) - center) @ R.T
    box = np.zeros((n, 2))
    has_box = label >= 0
    box[has_box] = truth[label[has_box]] / RING_SCALE_M
    frame = np.asarray(s.get("frame", np.full(nr, -1)), int)
    return dict(node=node, robot=robot, det_robot=r_of, frame=frame, geo=geo / RING_SCALE_M,
                pairs=np.column_stack([ii, jj]).astype(int).reshape(-1, 2), pair_ij=p_ij, pair_ji=p_ji,
                redges=np.column_stack([ri, rj]).astype(int).reshape(-1, 2),
                redge_feat=r_feat.reshape(len(ri), ROBOT_EDGE_WIDTH[geometry]),
                same=((label[ii] >= 0) & (label[ii] == label[jj])).astype(float), false=(label < 0).astype(float),
                box=box, has_box=has_box.astype(float), center=center, rotation=rotation)


def collate(encoded, images=None):
    """Several encoded scenes -> one batch (block-diagonal graph). images: (F, H, W) uint8 array of the frame cache;
    robots without a frame get a black image."""
    r_off = np.cumsum([0] + [len(e["frame"]) for e in encoded])
    d_off = np.cumsum([0] + [len(e["node"]) for e in encoded])
    pw = encoded[0]["pair_ij"].shape[1]
    ew = encoded[0]["redge_feat"].shape[1]

    def cat(key, width=None, offset=None):
        parts = [np.asarray(e[key]) + (offset[k] if offset is not None else 0) for k, e in enumerate(encoded)]
        if width is not None:
            parts = [p.reshape(len(p), width) for p in parts]
        return np.concatenate(parts)

    f32 = lambda x: torch.as_tensor(np.asarray(x, np.float32))
    i64 = lambda x: torch.as_tensor(np.asarray(x, np.int64))
    pairs = cat("pairs", 2, d_off)
    redges = cat("redges", 2, r_off)
    b = dict(node=f32(cat("node", NODE_FEATURES)), robot=f32(cat("robot", ROBOT_FEATURES)),
             det_robot=i64(cat("det_robot", offset=r_off)), n_robots=int(r_off[-1]), n_det=int(d_off[-1]),
             geo=f32(cat("geo", 2)), pair_a=i64(pairs[:, 0]), pair_b=i64(pairs[:, 1]),
             pair_ij=f32(cat("pair_ij", pw)), pair_ji=f32(cat("pair_ji", pw)),
             redge_src=i64(redges[:, 0]), redge_dst=i64(redges[:, 1]), redge_feat=f32(cat("redge_feat", ew)),
             same=f32(cat("same")), false=f32(cat("false")), box=f32(cat("box", 2)), has_box=f32(cat("has_box")))
    if images is not None:
        frame = cat("frame")
        img = np.zeros((len(frame),) + images.shape[1:], np.uint8)
        ok = frame >= 0
        img[ok] = images[frame[ok]]
        b["image"] = torch.as_tensor(img)
    return b


# ====================================================================
# Model
# ====================================================================

class ImageMLP(nn.Module):
    """Whole grey image -> k features. Each image is standardized (zero mean, unit spread) before the MLP."""

    def __init__(self, n_pixels, k, hidden=(256, 64)):
        super().__init__()
        self.net = mlp(n_pixels, *hidden, k)

    def forward(self, img):
        x = img.reshape(img.shape[0], -1).float()
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1.0)
        return self.net(x)


class ImageDetectionGNN(nn.Module):
    def __init__(self, k=8, img_size=(64, 36), geometry="fixed", hidden=64, hops=2, img_hidden=(256, 64), graph="A"):
        super().__init__()
        if graph not in GRAPHS:
            raise ValueError(f"unknown graph {graph}")
        h = hidden
        self.k, self.geometry, self.graph = k, geometry, graph
        pw, ew = PAIR_WIDTH[geometry], ROBOT_EDGE_WIDTH[geometry]
        relu = lambda *s: nn.Sequential(mlp(*s), nn.ReLU())
        self.image = ImageMLP(img_size[0] * img_size[1], k, img_hidden) if k > 0 else None
        self.message = nn.ModuleList([relu(h + pw, h, h) for _ in range(hops)])
        if graph == "A":
            self.enc = relu(NODE_FEATURES + k, h, h)
            self.update = nn.ModuleList([mlp(2 * h, h, h) for _ in range(hops)])
        else:
            self.enc = relu(NODE_FEATURES, h, h)
            self.robot_enc = relu(ROBOT_FEATURES + k, h, h)
            self.robot_to_det = nn.ModuleList([relu(h, h, h) for _ in range(hops)])
            self.det_to_robot = nn.ModuleList([relu(h, h, h) for _ in range(hops)])
            self.robot_msg = nn.ModuleList([relu(h + ew, h, h) for _ in range(hops)])
            self.update = nn.ModuleList([mlp(3 * h, h, h) for _ in range(hops)])
            self.robot_update = nn.ModuleList([mlp(3 * h, h, h) for _ in range(hops)])
        self.pair_head = mlp(2 * h + pw, h, h // 2, 1)
        self.false_head = mlp(h, h // 2, 1)
        self.pos_head = mlp(h, h, h // 2, 2)

    def features(self, b):
        """z of every robot of the batch (n_robots, k)."""
        return self.image(b["image"]) if self.image is not None else None

    def forward(self, b, z=None):
        if self.image is not None and z is None:
            z = self.features(b)
        x = b["node"]
        if self.graph == "A" and z is not None:
            x = torch.cat([x, z[b["det_robot"]]], dim=1)
        g = self.enc(x)
        if self.graph == "B":
            rx = b["robot"] if z is None else torch.cat([b["robot"], z], dim=1)
            r = self.robot_enc(rx)
        a, c = b["pair_a"], b["pair_b"]
        src, dst = torch.cat([a, c]), torch.cat([c, a])
        feat = torch.cat([b["pair_ji"], b["pair_ij"]])               # offset of src seen from dst
        for hop, (message, update) in enumerate(zip(self.message, self.update)):
            m = message(torch.cat([g[src], feat], dim=1))
            agg = segment_mean(m, dst, b["n_det"])
            if self.graph == "A":
                g = g + update(torch.cat([g, agg], dim=1))
                continue
            from_robot = self.robot_to_det[hop](r)[b["det_robot"]]
            to_robot = segment_mean(self.det_to_robot[hop](g), b["det_robot"], b["n_robots"])
            rm = self.robot_msg[hop](torch.cat([r[b["redge_src"]], b["redge_feat"]], dim=1))
            from_robots = segment_mean(rm, b["redge_dst"], b["n_robots"])
            g = g + update(torch.cat([g, agg, from_robot], dim=1))
            r = r + self.robot_update[hop](torch.cat([r, to_robot, from_robots], dim=1))
        if self.graph == "B":                                         # last word of the robot to its reports
            g = g + self.robot_to_det[-1](r)[b["det_robot"]]
        ga, gc = g[a], g[c]
        same = 0.5 * (self.pair_head(torch.cat([ga, gc, b["pair_ij"]], dim=1)) +
                      self.pair_head(torch.cat([gc, ga, b["pair_ji"]], dim=1)))
        pos = self.pos_head(g)
        if self.geometry == "fixed":
            pos = b["geo"] + 0.1 * pos                                  # learned correction of the fixed layer
        return same.view(-1), self.false_head(g).view(-1), pos


def loss_fn(out, b, w_same, w_false, w_pos=1.0):
    same, false, pos = out
    loss = F.binary_cross_entropy_with_logits(false, b["false"], pos_weight=torch.tensor(float(w_false)))
    if len(same):
        loss = loss + F.binary_cross_entropy_with_logits(same, b["same"], pos_weight=torch.tensor(float(w_same)))
    m = b["has_box"] > 0
    if m.any():
        loss = loss + w_pos * F.smooth_l1_loss(pos[m], b["box"][m], beta=0.05)
    return loss


def to_world(pos, e):
    """Positions in the model frame (units of RING_SCALE_M) -> world [m]."""
    c, s = math.cos(-e["rotation"]), math.sin(-e["rotation"])
    return (np.asarray(pos, float) * RING_SCALE_M) @ np.array([[c, -s], [s, c]]).T + e["center"]
