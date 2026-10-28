"""GNN for multi-robot box association, stage 1 (feature vectors only).

Graph: one NODE per robot. A node carries the robot pose and the set of its detections (feature vectors:
distance, bearing, viewing angle, world position). EDGES join robots that are neighbours on the ring
(all pairs when there are 3 robots or fewer) and carry the relative pose of the two robots. Two rounds
of message passing (two hops) let every robot receive the information of every other robot of a
four-robot ring. Everything runs on the host PC.

Read-out:
  same-box head   for every pair of detections of the scene (also two detections of the same robot):
                  probability that they are the same box
  false head      for every detection: probability that it is a false detection
Decoding (decode()): false detections are dropped, the rest are grouped by average-linkage clustering
on the same-box probabilities; the number of groups is the box count and the mean position of each
group is the box position.

Only the PyTorch operations nn.Linear, ReLU, indexing, cat, index_add and BCE-with-logits are used.
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

RING_SCALE_M = 1.3          # ring radius, used to scale positions
PAIR_SCALE_M = 0.1          # pair offsets are given in units of 10 cm
PAIR_CLIP_M = 1.0           # pair offsets beyond 1 m carry no extra information

DET_FEATURES = 10
ROBOT_FEATURES = 5
EDGE_FEATURES = 5
PAIR_FEATURES = 6


# ====================================================================
# Features
# ====================================================================

def ring_edges(robot_xy, center):
    """Directed edges between ring neighbours (every pair when there are 3 robots or fewer)."""
    n = len(robot_xy)
    if n <= 1:
        return []
    if n <= 3:
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    ang = np.arctan2(robot_xy[:, 1] - center[1], robot_xy[:, 0] - center[0])
    order = list(np.argsort(ang))
    edges = set()
    for k, i in enumerate(order):
        for j in (order[(k - 1) % n], order[(k + 1) % n]):
            edges.add((int(i), int(j)))
            edges.add((int(j), int(i)))
    return sorted(edges)


def encode_scene(scene, rotation=0.0):
    """numpy features of one scene, in a frame centred on the robots and rotated by `rotation`."""
    poses = scene["poses"]
    center = poses[:, :2].mean(axis=0)
    c, s = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -s], [s, c]])

    robot_xy = (poses[:, :2] - center) @ R.T
    robot_yaw = poses[:, 2] + rotation
    pos = (scene["pos"] - center) @ R.T
    r_of = scene["det_robot"]
    ray = robot_yaw[r_of] - scene["bearing"]                  # world direction of each detection
    u = np.stack([np.cos(ray), np.sin(ray)], axis=1)          # unit vector along the line of sight
    n_det = np.bincount(r_of, minlength=len(poses))

    robot = np.column_stack([robot_xy / RING_SCALE_M, np.cos(robot_yaw), np.sin(robot_yaw),
                             n_det / 4.0])
    det = np.column_stack([pos / RING_SCALE_M, u, scene["distance"], scene["bearing"],
                           np.cos(scene["view"]), np.sin(scene["view"]), robot_xy[r_of] / RING_SCALE_M])

    edges = ring_edges(robot_xy, np.zeros(2))
    edge_feat = []
    for i, j in edges:                                        # message i -> j, seen from robot j
        d = robot_xy[i] - robot_xy[j]
        cj, sj = math.cos(-robot_yaw[j]), math.sin(-robot_yaw[j])
        local = np.array([cj * d[0] - sj * d[1], sj * d[0] + cj * d[1]])
        dyaw = robot_yaw[i] - robot_yaw[j]
        edge_feat.append([*(local / RING_SCALE_M), math.cos(dyaw), math.sin(dyaw),
                          np.hypot(*d) / RING_SCALE_M])

    pairs = [(i, j) for i in range(len(pos)) for j in range(i + 1, len(pos))]
    pair_ij, pair_ji = [], []
    for i, j in pairs:
        pair_ij.append(pair_features(pos, u, r_of, i, j))
        pair_ji.append(pair_features(pos, u, r_of, j, i))

    label = scene.get("label")
    same = [float(label[i] >= 0 and label[i] == label[j]) for i, j in pairs] if label is not None else []
    false = (label < 0).astype(float) if label is not None else np.zeros(len(pos))
    return dict(robot=robot, det=det, det_robot=r_of, edges=edges, edge_feat=edge_feat,
                pairs=pairs, pair_ij=pair_ij, pair_ji=pair_ji, same=same, false=false)


def pair_features(pos, u, r_of, i, j):
    """Offset of detection j seen from detection i: along and across i's line of sight, distance,
    same robot, angle between the two lines of sight."""
    d = pos[j] - pos[i]
    dist = min(float(np.hypot(*d)), PAIR_CLIP_M)
    along = float(np.clip(d @ u[i], -PAIR_CLIP_M, PAIR_CLIP_M))
    across = float(np.clip(u[i][0] * d[1] - u[i][1] * d[0], -PAIR_CLIP_M, PAIR_CLIP_M))
    cos_rays = float(u[i] @ u[j])
    return [along / PAIR_SCALE_M, across / PAIR_SCALE_M, dist / PAIR_SCALE_M,
            float(r_of[i] == r_of[j]), cos_rays, abs(across) / PAIR_SCALE_M]


def collate(encoded):
    """Several encoded scenes -> one batch of tensors (block-diagonal graph)."""
    robot, det, det_robot, edge_src, edge_dst, edge_feat = [], [], [], [], [], []
    pair_a, pair_b, pair_ij, pair_ji, same, false, pair_scene = [], [], [], [], [], [], []
    r0 = d0 = 0
    for k, e in enumerate(encoded):
        robot.append(e["robot"])
        det.append(e["det"])
        det_robot.extend(e["det_robot"] + r0)
        for (i, j), f in zip(e["edges"], e["edge_feat"]):
            edge_src.append(i + r0)
            edge_dst.append(j + r0)
            edge_feat.append(f)
        for (i, j), fij, fji in zip(e["pairs"], e["pair_ij"], e["pair_ji"]):
            pair_a.append(i + d0)
            pair_b.append(j + d0)
            pair_ij.append(fij)
            pair_ji.append(fji)
            pair_scene.append(k)
        same.extend(e["same"])
        false.extend(e["false"])
        r0 += len(e["robot"])
        d0 += len(e["det"])

    def f32(x, width):
        return torch.tensor(np.asarray(x, float).reshape(-1, width), dtype=torch.float32)

    def i64(x):
        return torch.tensor(np.asarray(x, np.int64), dtype=torch.long)

    return dict(robot=f32(np.concatenate(robot), ROBOT_FEATURES), det=f32(np.concatenate(det), DET_FEATURES),
                det_robot=i64(det_robot), n_robots=r0, n_det=d0,
                edge_src=i64(edge_src), edge_dst=i64(edge_dst), edge_feat=f32(edge_feat, EDGE_FEATURES),
                pair_a=i64(pair_a), pair_b=i64(pair_b),
                pair_ij=f32(pair_ij, PAIR_FEATURES), pair_ji=f32(pair_ji, PAIR_FEATURES),
                same=torch.tensor(np.asarray(same, float), dtype=torch.float32),
                false=torch.tensor(np.asarray(false, float), dtype=torch.float32))


# ====================================================================
# Model
# ====================================================================

def mlp(*sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])                       # no activation after the last layer


def segment_mean(values, index, n):
    """Mean of the rows of `values` grouped by `index` (empty groups give 0)."""
    total = torch.zeros(n, values.shape[1]).index_add(0, index, values)
    count = torch.zeros(n, 1).index_add(0, index, torch.ones(values.shape[0], 1))
    return total / count.clamp(min=1.0)


class RobotGNN(nn.Module):
    def __init__(self, hidden=64, hops=2):
        super().__init__()
        h = hidden
        self.det_enc = nn.Sequential(mlp(DET_FEATURES, h, h), nn.ReLU())
        self.robot_enc = nn.Sequential(mlp(ROBOT_FEATURES + h, h, h), nn.ReLU())
        self.message = nn.ModuleList([nn.Sequential(mlp(h + EDGE_FEATURES, h, h), nn.ReLU())
                                      for _ in range(hops)])
        self.update = nn.ModuleList([mlp(2 * h, h, h) for _ in range(hops)])
        self.det_ctx = nn.Sequential(mlp(2 * h, h, h), nn.ReLU())
        self.pair_head = mlp(2 * h + PAIR_FEATURES, h, h // 2, 1)
        self.false_head = mlp(h, h // 2, 1)

    def forward(self, b):
        h_det = self.det_enc(b["det"])                                       # detections
        pooled = segment_mean(h_det, b["det_robot"], b["n_robots"])          # set of detections per robot
        node = self.robot_enc(torch.cat([b["robot"], pooled], dim=1))        # robot nodes
        for message, update in zip(self.message, self.update):               # two hops on the ring
            m = message(torch.cat([node[b["edge_src"]], b["edge_feat"]], dim=1))
            agg = segment_mean(m, b["edge_dst"], b["n_robots"])
            node = node + update(torch.cat([node, agg], dim=1))
        g = self.det_ctx(torch.cat([h_det, node[b["det_robot"]]], dim=1))   # detection in its context
        ga, gb = g[b["pair_a"]], g[b["pair_b"]]
        same = 0.5 * (self.pair_head(torch.cat([ga, gb, b["pair_ij"]], dim=1)) +
                      self.pair_head(torch.cat([gb, ga, b["pair_ji"]], dim=1)))
        false = self.false_head(g)
        return same.view(-1), false.view(-1)


def loss_fn(same_logit, false_logit, b, pos_weight_same, pos_weight_false):
    loss = F.binary_cross_entropy_with_logits(
        same_logit, b["same"], pos_weight=torch.tensor(float(pos_weight_same), dtype=torch.float32))
    return loss + F.binary_cross_entropy_with_logits(
        false_logit, b["false"], pos_weight=torch.tensor(float(pos_weight_false), dtype=torch.float32))


# ====================================================================
# Decoding
# ====================================================================

def decode(pos, pairs, p_same, p_false, threshold=0.5):
    """Boxes of one scene: drop false detections, then average-linkage clustering on p_same.
    Returns (list of box positions, cluster index per detection, -1 for dropped ones)."""
    n = len(pos)
    P = np.eye(n)
    for (i, j), p in zip(pairs, p_same):
        P[i, j] = P[j, i] = p
    keep = [i for i in range(n) if p_false[i] < threshold]
    clusters = [[i] for i in keep]
    while len(clusters) > 1:
        best, bi, bj = threshold, -1, -1
        for a in range(len(clusters)):
            for c in range(a + 1, len(clusters)):
                link = float(np.mean(P[np.ix_(clusters[a], clusters[c])]))
                if link > best:
                    best, bi, bj = link, a, c
        if bi < 0:
            break
        clusters[bi] += clusters.pop(bj)
    assign = np.full(n, -1)
    for k, cl in enumerate(clusters):
        assign[cl] = k
    return [pos[cl].mean(axis=0) for cl in clusters], assign