"""Variants of the detection-level GNN asked for in the review of the architecture slides (Nov 2026).

Only the encoding and the read-out change; everything else (message passing, decoding, metrics) is the one of
gnn/model.py, so the variants are directly comparable with results/plan_v2/M4/gnn/detection_full_full_h2.

  frame   centroid  positions relative to the centroid of the robots, divided by L = 1.3 m (as gnn/model.py)
          world     positions in the world frame of the plan (origin at the arena centre), divided by L: the box
                    disc of the plan is shifted at random, so the position of a detection is not a cue by itself
  degree  False     node input as gnn/model.py: [x_d, s_r(d)], 15 values
          True      + log(1 + |N(d)|), N(d) = detections within the candidate-pair distance d_c: 16 values
  false_head  True  two heads, as gnn/model.py
              False pair head only; no detection is dropped by the decoding (p_f = 0) and the loss has no false term
  hops    K, the number of message-passing layers (gnn/model.py --hops)
"""

import numpy as np
import torch
import torch.nn.functional as F

from gnn.model import DET_FEATURES, RING_SCALE_M, DetectionGNN, encode_scene, segment_mean


def encode(scene, frame="centroid", degree=False, gate=1.0):
    """gnn.model.encode_scene (detection graph, every robot linked, central association) plus the review options."""
    e = encode_scene(scene, features="full", edges="full", gate=gate, association="central", hops=1)
    det, robot = e["det"].copy(), e["robot"].copy()
    if frame == "world":
        # encode_scene subtracts the centroid of the robots (no rotation): add it back
        shift = scene["poses"][:, :2].mean(axis=0) / RING_SCALE_M
        det[:, 0:2] += shift              # q_d / L
        det[:, 8:10] += shift             # position of the detection's robot / L
        robot[:, 0:2] += shift            # robot position / L
    elif frame != "centroid":
        raise ValueError(f"unknown frame {frame}")
    if degree:
        n = len(det)
        count = np.bincount(e["pairs"].reshape(-1), minlength=n) if len(e["pairs"]) else np.zeros(n)
        det = np.column_stack([det, np.log1p(count)])
    e["det"], e["robot"] = det, robot
    return e


class ReviewGNN(DetectionGNN):
    """DetectionGNN with an optional false head. Without it the false logit is a large negative constant, so
    p_f = 0 and the decoding keeps every detection."""

    def __init__(self, hidden=64, hops=2, degree=False, false_head=True):
        super().__init__(hidden=hidden, hops=hops, det_features=DET_FEATURES["full"] + int(degree), skip="residual")
        self.use_false = false_head
        if not false_head:
            del self.readout.false_head           # no parameters for a head that is not trained

    def forward(self, b):
        if self.use_false:
            return super().forward(b)
        g = self.enc(torch.cat([b["det"], b["robot"][b["det_robot"]]], dim=1))
        link = b["pair_link"]
        a, c = b["pair_a"][link], b["pair_b"][link]
        src, dst = torch.cat([a, c]), torch.cat([c, a])
        feat = torch.cat([b["pair_ji"][link], b["pair_ij"][link]])
        for message, update in zip(self.message, self.update):
            m = message(torch.cat([g[src], feat], dim=1))
            g = g + update(torch.cat([g, segment_mean(m, dst, b["n_det"])], dim=1))
        ga, gb = g[b["pair_a"]], g[b["pair_b"]]
        head = self.readout.pair_head
        same = 0.5 * (head(torch.cat([ga, gb, b["pair_ij"]], dim=1)) + head(torch.cat([gb, ga, b["pair_ji"]], dim=1)))
        return same.view(-1), torch.full((g.shape[0],), -30.0)


def loss_fn(same_logit, false_logit, b, w_same, w_false, false_head=True):
    loss = torch.zeros(())
    if false_head:
        loss = loss + F.binary_cross_entropy_with_logits(
            false_logit, b["false"], pos_weight=torch.tensor(float(w_false), dtype=torch.float32))
    if len(same_logit):
        loss = loss + F.binary_cross_entropy_with_logits(
            same_logit, b["same"], pos_weight=torch.tensor(float(w_same), dtype=torch.float32))
    return loss
