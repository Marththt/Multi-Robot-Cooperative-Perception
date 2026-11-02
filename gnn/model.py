"""GNN for multi-robot box association (feature vectors only, no images).

Two graph designs, same inputs and same read-out:

  RobotGNN       one NODE per robot, carrying the robot pose and the set of its detections; EDGES join
                 robots that can exchange messages and carry their relative pose.
  DetectionGNN   one NODE per detection, carrying the feature vector and the pose of its robot; EDGES join
                 detections that may be the same box (closer than the pair gate) and whose robots can
                 exchange messages, or that come from the same robot (two faces of one box); they carry
                 the offset between the two detections.

Which robots exchange messages (edges=):
  ring     neighbours on the ring around the robots' centre (every pair with 3 robots or fewer)
  opposite the robots facing each robot across the team: the one in front and its two neighbours
           (the diagonals); with an odd number of robots, the two that face it
  range    robots closer than comm_range metres (limited communication)
  full     every pair of robots (upper bound: one central model sees everything)
`hops` rounds of message passing; with a local graph a robot hears robots up to `hops` links away.

Association (association=):
  central  every pair of detections closer than the pair gate is scored, whichever robots made them: one
           central read-out sees all detections (the communication graph only carries context)
  local    distributed: two detections are scored only if their robots are at most `hops` links apart in
           the communication graph (or are the same robot), so a robot can only associate what reaches it
           through the network; detections of robots that cannot reach each other are never merged

Inputs (features=):
  full       detection position, line of sight, distance, bearing, viewing angle of the marker, robot pose
  positions  the same without the viewing angle: only what follows from the positions and the robot poses

Skip connections (skip=):
  residual  each hop adds its update to the node, so a node keeps what its robot saw (default)
  none      each hop replaces the node
  jk        jumping knowledge: residual hops, and the read-out sees the node after every hop

Read-out:
  same-box head   for every pair of detections closer than the pair gate (also two detections of the same
                  robot): probability that they are the same box
  false head      for every detection: probability that it is a false detection
Decoding (decode()): false detections are dropped, the rest are grouped by average-linkage clustering on
the same-box probabilities (pairs beyond the gate count as 0); the number of groups is the box count and
the mean position of each group is the box position.

Everything runs on the host PC. Only nn.Linear, ReLU, indexing, cat, index_add and BCE-with-logits are used.
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

RING_SCALE_M = 1.3          # ring radius, used to scale positions
PAIR_SCALE_M = 0.1          # pair offsets are given in units of 10 cm
PAIR_CLIP_M = 1.0           # pair offsets beyond 1 m carry no extra information

DET_FEATURES = {"full": 10, "positions": 8}
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


def opposite_edges(robot_xy, center):
    """Directed edges from every robot to the robots facing it across the team: the robots are ordered by
    their angle around the centre, and robot i is linked to the robots whose position in that order is
    within one step of the opposite point (i + n/2): with an even n the robot in front and the two next
    to it (the diagonals), with an odd n the two robots that face it. Every pair with 3 robots or fewer."""
    n = len(robot_xy)
    if n <= 1:
        return []
    if n <= 3:
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    ang = np.arctan2(robot_xy[:, 1] - center[1], robot_xy[:, 0] - center[0])
    order = list(np.argsort(ang))
    offsets = [o for o in range(1, n) if abs(o - n / 2) <= 1]
    edges = set()
    for k, i in enumerate(order):
        for o in offsets:
            j = order[(k + o) % n]
            edges.add((int(i), int(j)))
            edges.add((int(j), int(i)))
    return sorted(edges)


def robot_edges(robot_xy, mode="ring", comm_range=None):
    """Directed robot-robot edges for the chosen communication graph."""
    n = len(robot_xy)
    if mode == "ring":
        return ring_edges(robot_xy, np.zeros(2))
    if mode == "opposite":
        return opposite_edges(robot_xy, np.zeros(2))
    if mode == "full":
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    if mode == "range":
        if comm_range is None:
            raise ValueError("edges='range' needs comm_range")
        d = np.linalg.norm(robot_xy[:, None, :] - robot_xy[None, :, :], axis=2)
        return [(i, j) for i in range(n) for j in range(n) if i != j and d[i, j] <= comm_range]
    raise ValueError(f"unknown edge mode {mode}")


def pair_features(pos, u, r_of, i, j):
    """Offset of detection j seen from detection i (arrays of indices): along and across i's line of sight,
    distance, same robot, angle between the two lines of sight, |across|."""
    d = pos[j] - pos[i]
    dist = np.minimum(np.hypot(d[:, 0], d[:, 1]), PAIR_CLIP_M)
    ui = u[i]
    along = np.clip((d * ui).sum(axis=1), -PAIR_CLIP_M, PAIR_CLIP_M)
    across = np.clip(ui[:, 0] * d[:, 1] - ui[:, 1] * d[:, 0], -PAIR_CLIP_M, PAIR_CLIP_M)
    cos_rays = (ui * u[j]).sum(axis=1)
    return np.column_stack([along / PAIR_SCALE_M, across / PAIR_SCALE_M, dist / PAIR_SCALE_M,
                            (r_of[i] == r_of[j]).astype(float), cos_rays, np.abs(across) / PAIR_SCALE_M])


def reachable(n, edge_arr, hops):
    """n x n bool: robot j can be reached from robot i in at most `hops` links (every robot reaches itself)."""
    adj = np.eye(n, dtype=bool)
    if len(edge_arr):
        adj[edge_arr[:, 0], edge_arr[:, 1]] = True
        adj[edge_arr[:, 1], edge_arr[:, 0]] = True
    reach = np.eye(n, dtype=bool)
    for _ in range(hops):
        reach = (reach.astype(int) @ adj.astype(int)) > 0
    return reach


def encode_scene(scene, rotation=0.0, features="full", edges="ring", comm_range=None, gate=None,
                 association="central", hops=2):
    """numpy features of one scene, in a frame centred on the robots and rotated by `rotation`.
    gate: only pairs of detections closer than this [m] are scored (None = every pair).
    association='local': only pairs whose robots are at most `hops` links apart are scored."""
    poses = scene["poses"]
    center = poses[:, :2].mean(axis=0)
    c, s = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -s], [s, c]])

    robot_xy = (poses[:, :2] - center) @ R.T
    robot_yaw = poses[:, 2] + rotation
    pos = (np.asarray(scene["pos"], float).reshape(-1, 2) - center) @ R.T
    r_of = np.asarray(scene["det_robot"], int)
    ray = robot_yaw[r_of] - scene["bearing"]                  # world direction of each detection
    u = np.stack([np.cos(ray), np.sin(ray)], axis=1).reshape(-1, 2)   # unit vector along the line of sight
    n_det = np.bincount(r_of, minlength=len(poses))

    robot = np.column_stack([robot_xy / RING_SCALE_M, np.cos(robot_yaw), np.sin(robot_yaw), n_det / 4.0])
    cols = [pos / RING_SCALE_M, u, scene["distance"], scene["bearing"]]
    if features == "full":
        cols += [np.cos(scene["view"]), np.sin(scene["view"])]
    elif features != "positions":
        raise ValueError(f"unknown feature set {features}")
    cols.append(robot_xy[r_of] / RING_SCALE_M)
    det = np.column_stack(cols).reshape(len(pos), DET_FEATURES[features])

    edge_list = robot_edges(robot_xy, edges, comm_range)
    edge_arr = np.array(edge_list, int).reshape(-1, 2)
    if len(edge_arr):
        i, j = edge_arr[:, 0], edge_arr[:, 1]                 # message i -> j, seen from robot j
        d = robot_xy[i] - robot_xy[j]
        cj, sj = np.cos(-robot_yaw[j]), np.sin(-robot_yaw[j])
        local = np.column_stack([cj * d[:, 0] - sj * d[:, 1], sj * d[:, 0] + cj * d[:, 1]])
        dyaw = robot_yaw[i] - robot_yaw[j]
        edge_feat = np.column_stack([local / RING_SCALE_M, np.cos(dyaw), np.sin(dyaw),
                                     np.hypot(d[:, 0], d[:, 1]) / RING_SCALE_M])
    else:
        edge_feat = np.zeros((0, EDGE_FEATURES))

    ii, jj = np.triu_indices(len(pos), k=1)
    if gate is not None and len(ii):
        near = np.hypot(*(pos[jj] - pos[ii]).T) <= gate
        ii, jj = ii[near], jj[near]
    if association == "local":
        reach = reachable(len(poses), edge_arr, hops)
        ok = reach[r_of[ii], r_of[jj]]
        ii, jj = ii[ok], jj[ok]
    elif association != "central":
        raise ValueError(f"unknown association {association}")
    pairs = np.column_stack([ii, jj]).astype(int).reshape(-1, 2)
    pair_ij = pair_features(pos, u, r_of, ii, jj).reshape(-1, PAIR_FEATURES)
    pair_ji = pair_features(pos, u, r_of, jj, ii).reshape(-1, PAIR_FEATURES)
    linked = np.zeros((len(poses), len(poses)), bool)
    np.fill_diagonal(linked, True)
    if len(edge_arr):
        linked[edge_arr[:, 0], edge_arr[:, 1]] = True
    pair_link = linked[r_of[ii], r_of[jj]] | linked[r_of[jj], r_of[ii]]

    label = scene.get("label")
    if label is not None:
        label = np.asarray(label, int)
        same = ((label[ii] >= 0) & (label[ii] == label[jj])).astype(float)
        false = (label < 0).astype(float)
    else:
        same, false = np.zeros(len(ii)), np.zeros(len(pos))
    return dict(robot=robot, det=det, det_robot=r_of, edges=edge_arr, edge_feat=edge_feat,
                pairs=pairs, pair_ij=pair_ij, pair_ji=pair_ji, pair_link=pair_link, same=same, false=false)


def collate(encoded):
    """Several encoded scenes -> one batch of tensors (block-diagonal graph)."""
    r_off = np.cumsum([0] + [len(e["robot"]) for e in encoded])
    d_off = np.cumsum([0] + [len(e["det"]) for e in encoded])

    def cat(key, width=None, offset=None):
        parts = [e[key] + (offset[k] if offset is not None else 0) for k, e in enumerate(encoded)]
        if width is not None:
            parts = [np.asarray(p, float).reshape(-1, width) for p in parts]
        return np.concatenate(parts) if parts else np.zeros((0,) if width is None else (0, width))

    det_width = encoded[0]["det"].shape[1]
    edges = cat("edges", offset=r_off).reshape(-1, 2) if encoded else np.zeros((0, 2), int)
    pairs = cat("pairs", offset=d_off).reshape(-1, 2) if encoded else np.zeros((0, 2), int)

    def f32(x):
        return torch.as_tensor(np.asarray(x, np.float32))

    def i64(x):
        return torch.as_tensor(np.asarray(x, np.int64))

    return dict(robot=f32(cat("robot", ROBOT_FEATURES)), det=f32(cat("det", det_width)),
                det_robot=i64(cat("det_robot", offset=r_off)), n_robots=int(r_off[-1]), n_det=int(d_off[-1]),
                edge_src=i64(edges[:, 0]), edge_dst=i64(edges[:, 1]), edge_feat=f32(cat("edge_feat", EDGE_FEATURES)),
                pair_a=i64(pairs[:, 0]), pair_b=i64(pairs[:, 1]),
                pair_ij=f32(cat("pair_ij", PAIR_FEATURES)), pair_ji=f32(cat("pair_ji", PAIR_FEATURES)),
                pair_link=torch.as_tensor(np.concatenate([e["pair_link"] for e in encoded]).astype(bool)),
                same=f32(np.concatenate([e["same"] for e in encoded])),
                false=f32(np.concatenate([e["false"] for e in encoded])))


# ====================================================================
# Models
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


class PairReadout(nn.Module):
    """Same-box logit per scored pair (symmetric) and false-detection logit per detection."""

    def __init__(self, h):
        super().__init__()
        self.pair_head = mlp(2 * h + PAIR_FEATURES, h, h // 2, 1)
        self.false_head = mlp(h, h // 2, 1)

    def forward(self, g, b):
        ga, gb = g[b["pair_a"]], g[b["pair_b"]]
        same = 0.5 * (self.pair_head(torch.cat([ga, gb, b["pair_ij"]], dim=1)) +
                      self.pair_head(torch.cat([gb, ga, b["pair_ji"]], dim=1)))
        return same.view(-1), self.false_head(g).view(-1)


class RobotGNN(nn.Module):
    """skip: how the hops are chained (see SKIP_MODES)."""

    def __init__(self, hidden=64, hops=2, det_features=DET_FEATURES["full"], skip="residual"):
        super().__init__()
        h = hidden
        self.skip = skip
        self.det_enc = nn.Sequential(mlp(det_features, h, h), nn.ReLU())
        self.robot_enc = nn.Sequential(mlp(ROBOT_FEATURES + h, h, h), nn.ReLU())
        self.message = nn.ModuleList([nn.Sequential(mlp(h + EDGE_FEATURES, h, h), nn.ReLU())
                                      for _ in range(hops)])
        self.update = nn.ModuleList([mlp(2 * h, h, h) for _ in range(hops)])
        n_node = (hops + 1) if skip == "jk" else 1
        self.det_ctx = nn.Sequential(mlp(h + n_node * h, h, h), nn.ReLU())
        self.readout = PairReadout(h)

    def forward(self, b):
        h_det = self.det_enc(b["det"])                                       # detections
        pooled = segment_mean(h_det, b["det_robot"], b["n_robots"])          # set of detections per robot
        node = self.robot_enc(torch.cat([b["robot"], pooled], dim=1))        # robot nodes
        states = [node]
        for message, update in zip(self.message, self.update):               # message passing between robots
            m = message(torch.cat([node[b["edge_src"]], b["edge_feat"]], dim=1))
            agg = segment_mean(m, b["edge_dst"], b["n_robots"])
            new = update(torch.cat([node, agg], dim=1))
            node = new if self.skip == "none" else node + new
            states.append(node)
        robot_view = torch.cat(states, dim=1) if self.skip == "jk" else node
        g = self.det_ctx(torch.cat([h_det, robot_view[b["det_robot"]]], dim=1))   # detection in its context
        return self.readout(g, b)


class DetectionGNN(nn.Module):
    def __init__(self, hidden=64, hops=2, det_features=DET_FEATURES["full"], skip="residual"):
        super().__init__()
        h = hidden
        self.skip = skip
        self.enc = nn.Sequential(mlp(det_features + ROBOT_FEATURES, h, h), nn.ReLU())
        self.message = nn.ModuleList([nn.Sequential(mlp(h + PAIR_FEATURES, h, h), nn.ReLU())
                                      for _ in range(hops)])
        self.update = nn.ModuleList([mlp(2 * h, h, h) for _ in range(hops)])
        self.jk = nn.Sequential(mlp((hops + 1) * h, h, h), nn.ReLU()) if skip == "jk" else None
        self.readout = PairReadout(h)

    def forward(self, b):
        g = self.enc(torch.cat([b["det"], b["robot"][b["det_robot"]]], dim=1))
        link = b["pair_link"]
        a, c = b["pair_a"][link], b["pair_b"][link]
        src = torch.cat([a, c])
        dst = torch.cat([c, a])
        feat = torch.cat([b["pair_ji"][link], b["pair_ij"][link]])      # offset of src seen from dst
        states = [g]
        for message, update in zip(self.message, self.update):
            m = message(torch.cat([g[src], feat], dim=1))
            agg = segment_mean(m, dst, b["n_det"])
            new = update(torch.cat([g, agg], dim=1))
            g = new if self.skip == "none" else g + new
            states.append(g)
        if self.jk is not None:
            g = self.jk(torch.cat(states, dim=1))
        return self.readout(g, b)


SKIP_MODES = {
    "residual": "each hop adds its update to the node (node = node + update); the read-out uses the "
                "detection's own encoding and the node after the last hop",
    "none": "each hop replaces the node (node = update); no shortcut across the hops",
    "jk": "residual hops, and the read-out uses the node after EVERY hop (0, 1, ... hops): "
          "jumping knowledge, the first layer connected to the last",
}


def build_model(graph="robot", hidden=64, hops=2, features="full", skip="residual"):
    cls = {"robot": RobotGNN, "detection": DetectionGNN}[graph]
    return cls(hidden=hidden, hops=hops, det_features=DET_FEATURES[features], skip=skip)


def loss_fn(same_logit, false_logit, b, pos_weight_same, pos_weight_false):
    loss = F.binary_cross_entropy_with_logits(
        false_logit, b["false"], pos_weight=torch.tensor(float(pos_weight_false), dtype=torch.float32))
    if len(same_logit):
        loss = loss + F.binary_cross_entropy_with_logits(
            same_logit, b["same"], pos_weight=torch.tensor(float(pos_weight_same), dtype=torch.float32))
    return loss


# ====================================================================
# Decoding
# ====================================================================

def decode(pos, pairs, p_same, p_false, threshold=0.5):
    """Boxes of one scene: drop false detections, then average-linkage clustering on p_same (pairs not
    scored count as 0). Returns (list of box positions, cluster index per detection, -1 for dropped ones)."""
    pos = np.asarray(pos, float).reshape(-1, 2)
    n = len(pos)
    P = np.eye(n)
    for (i, j), p in zip(pairs, p_same):
        P[i, j] = P[j, i] = p
    keep = [i for i in range(n) if p_false[i] < threshold]
    clusters = [[i] for i in keep]
    if clusters:
        S = P[np.ix_(keep, keep)].copy()          # current sum of probabilities between clusters
        sizes = np.ones(len(keep))
        alive = np.ones(len(keep), bool)
        members = [[i] for i in keep]
        np.fill_diagonal(S, -np.inf)
        while alive.sum() > 1:
            link = S / np.outer(sizes, sizes)
            link[~alive, :] = -np.inf
            link[:, ~alive] = -np.inf
            a, c = np.unravel_index(np.argmax(link), link.shape)
            if link[a, c] <= threshold:
                break
            a, c = min(a, c), max(a, c)
            S[a, :] += S[c, :]
            S[:, a] += S[:, c]
            S[a, a] = -np.inf
            sizes[a] += sizes[c]
            alive[c] = False
            members[a] += members[c]
        clusters = [members[k] for k in range(len(keep)) if alive[k]]
    assign = np.full(n, -1)
    for k, cl in enumerate(clusters):
        assign[cl] = k
    return [pos[cl].mean(axis=0) for cl in clusters], assign
