"""Classical data association -- MHT (point 3 of Section 3.2 of the
professor's document), for ONE static shot with several robots.

The problem: every robot sends a list of positions (x, y) for a class,
without identity. Which reports from different robots are the same
object? With static objects, tracking over several frames reduces
(Section 3.5, point 3) to an association over a single shot: here the
"steps" of the tracking are not time instants but the robots, one after
the other. Robot 1 creates the first hypotheses, robot 2 confirms them,
contradicts them or adds new ones, and so on.

Model (sensor_model.py):
  - a report z comes from an object at x with density p_D * N(z; x, R)
  - an object the robot should have seen and did not report costs
    (1 - p_D): this is how the field of view and the faults enter the count
  - a report can also be a false alarm (clutter density)
    or a new object that the previous robots did not see
    (density prior * p_D * product of the (1 - p_D) of the previous robots)
  - a robot never reports the same object twice

MHT (Multi-Hypothesis Tracking) keeps the N best COMPLETE hypotheses:
each one is a partition of all the reports into objects, with its
score. At the end the best one wins. Exact when N is large enough
(with a few robots and few objects it is; the cost grows with the
number of robots and of objects, see n_hypotheses).
"""

import heapq
import math
from itertools import count as _counter

import numpy as np

from vf.fusion.sensor_model import P_MIN, gauss2, mahalanobis2

GATE_THRESHOLD = 13.8  # chi^2 with 2 degrees of freedom at 99.9%


def top_k_combinations(options, k):
    """options: one list per element, each [(log_score,
    exclusive_key_or_None, label), ...]. Returns the k combinations
    (one option per element, keys never repeated) with the highest sum
    of scores: [(sum, [label, ...])], best first.
    Branch and bound: a branch is cut when even the best it can still
    achieve does not make it into the k best."""
    n = len(options)
    opts = [sorted(o, key=lambda t: -t[0]) for o in options]
    max_after = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        max_after[i] = max_after[i + 1] + opts[i][0][0]
    best = []  # min-heap of (sum, id, choices)
    ids = _counter()

    def search(i, total, used, choices):
        if len(best) >= k and total + max_after[i] <= best[0][0]:
            return
        if i == n:
            entry = (total, next(ids), list(choices))
            if len(best) < k:
                heapq.heappush(best, entry)
            else:
                heapq.heapreplace(best, entry)
            return
        for score, key, label in opts[i]:
            if key is not None and key in used:
                continue
            if key is not None:
                used.add(key)
            choices.append(label)
            search(i + 1, total + score, used, choices)
            choices.pop()
            if key is not None:
                used.discard(key)

    search(0, 0.0, set(), [])
    return [(s, ch) for s, _, ch in sorted(best, key=lambda v: -v[0])]


def _kalman(m, P, z, R):
    S = P + R
    K = P @ np.linalg.inv(S)
    return m + K @ (z - m), (np.eye(2) - K) @ P


def _log_birth(sensor, pose, z, previous_poses):
    """log of the "new object" density for report z, or None if the
    robot could not see it (p_D=0 at z)."""
    pn = sensor.p_d_measured_one(pose, z)
    if pn <= 0:
        return None
    missed_before = 1.0
    for pp in previous_poses:
        missed_before *= 1.0 - sensor.p_d_measured_one(pp, z)
    return (math.log(sensor.prior_density()) + math.log(pn)
            + math.log(max(missed_before, P_MIN)))


# ====================================================================
# MHT
# ====================================================================

def mht(order, reports, poses, sensor, n_hypotheses=100, gate_threshold=GATE_THRESHOLD):
    """order: robot_id in processing order (ALL the robots of the shot,
    even those without reports: their "I saw nothing" counts).
    reports: {robot_id: [array([x, y]), ...]} for ONE class.
    poses: {robot_id: (x, y, yaw) or None}."""
    log_clutter = math.log(sensor.clutter_density())
    hypotheses = [(0.0, [])]  # (log weight, [object]) ; object = (m, P, [(rid, j)])
    seen_poses = []
    for rid in order:
        Z = [np.asarray(z, float) for z in reports.get(rid, [])]
        pose = poses.get(rid)
        R = sensor.cov(pose)
        log_new = [_log_birth(sensor, pose, z, seen_poses) for z in Z]
        children = []
        for log_w, objects in hypotheses:
            if objects:
                means = np.array([o[0] for o in objects])
                pd_o = np.clip(sensor.p_d_measured(pose, means), P_MIN, 1 - P_MIN)
            else:
                pd_o = np.zeros(0)
            base = log_w + float(np.sum(np.log(1.0 - pd_o)))
            options = []
            for j, z in enumerate(Z):
                o = [(log_clutter, None, ("c", None))]
                if log_new[j] is not None:
                    o.append((log_new[j], None, ("n", None)))
                for t, (m, P, _) in enumerate(objects):
                    S = P + R
                    if mahalanobis2(z, m, S) > gate_threshold:
                        continue
                    ll = (math.log(pd_o[t]) + math.log(float(gauss2(z, m, S)[0]) + 1e-300)
                          - math.log(1.0 - pd_o[t]))
                    o.append((ll, t, ("t", t)))
                options.append(o)
            for total, choices in top_k_combinations(options, n_hypotheses):
                new_objects = list(objects)
                for j, (kind, t) in enumerate(choices):
                    if kind == "t":
                        m, P, history = new_objects[t]
                        m2, P2 = _kalman(m, P, Z[j], R)
                        new_objects[t] = (m2, P2, history + [(rid, j)])
                    elif kind == "n":
                        new_objects.append((Z[j].copy(), R.copy(), [(rid, j)]))
                children.append((base + total, new_objects))
        children.sort(key=lambda v: -v[0])
        # also drop the hypotheses a million times less likely than the
        # best one: they do not change the result, they only slow it down
        hypotheses = [h for h in children[:n_hypotheses] if h[0] >= children[0][0] - 14.0]
        seen_poses.append(pose)

    maximum = hypotheses[0][0]
    weights = np.array([math.exp(w - maximum) for w, _ in hypotheses])
    weights /= weights.sum()
    n_distr = {}
    for p, (_, objs) in zip(weights, hypotheses):
        n_distr[len(objs)] = n_distr.get(len(objs), 0.0) + float(p)
    best = hypotheses[0][1]
    return dict(
        objects=[tuple(float(v) for v in m) for m, _, _ in best],
        n=len(best),
        extra=dict(p_best_hypothesis=float(weights[0]), n_distribution=n_distr,
                   reports_per_object=[s for _, _, s in best]))