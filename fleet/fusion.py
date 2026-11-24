"""From the detections of the whole fleet to the objects: how many targets, and where.

  objects(dets, looks, grid, ...)   detections grouped by sequential association (the most precise first, a
                                    Mahalanobis gate with the range / bearing covariance of each detection),
                                    information-weighted positions (rays from many places: a triangulation, so
                                    the bearing makes up for a poor range); a group is a target when it has at least
                                    min_det detections AND it was detected in at least min_ratio of the frames
                                    whose rays reached its place (looks of the brain, median over the cells
                                    within 10 cm: a ray stopping on the box counts): a false detection is seen once, a
                                    real target again every time a camera looks there. Over a whole run this is
                                    the negative evidence ("I looked and it was not there") of the free space.
  score(est, truth, gate_m)         optimal matching (Hungarian) within gate_m: TP, FP, FN, position error, the
                                    count error and OSPA (c = 0.5 m, as vf/fusion/localization_methods.ospa).

Other fusions of the thesis (MHT, the graph network) take one shot of a few robots; here a run has hundreds of
frames, every frame a "virtual robot", which is what objects() handles directly. They can be run on the same
detections later (pipeline/build_scans style) to compare.
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


GATE_CHI2 = 9.21        # 99% of a 2-D Gaussian
MERGE_M = 0.20          # objects closer than this are one box (boxes 13.6 cm across, placed farther apart)


def det_cov(d, sd_range, sd_bearing_rad):
    """Covariance of a detection's position: range error along the line of sight, bearing error across it."""
    th = math.atan2(d["xy"][1] - d["lens"][1], d["xy"][0] - d["lens"][0])
    sr = sd_range[0] + sd_range[1] * d["range"]
    sb = max(d["range"], 0.05) * sd_bearing_rad
    c, s = math.cos(th), math.sin(th)
    Rot = np.array([[c, -s], [s, c]])
    return Rot @ np.diag([sr * sr, sb * sb]) @ Rot.T


def objects(dets, looks, grid, sd_range=(0.02, 0.03), sd_bearing_deg=1.0, min_det=3, min_ratio=0.25,
            gate=GATE_CHI2, merge_m=MERGE_M):
    """Sequential association, most precise detections first: a detection joins the most likely object among
    those within the Mahalanobis gate (its covariance + the object's), otherwise it starts an object.
    Object position: information-weighted mean. Objects closer than merge_m, or within the gate of each other
    (their two covariances), are merged (the precise detections make an object's covariance tiny, and the 1% of
    detections outside the gate would start a twin a few cm away). Then the support test (min_det, min_ratio of the looks)."""
    if not dets:
        return []
    sb = math.radians(sd_bearing_deg)
    covs = [det_cov(d, sd_range, sb) for d in dets]
    order = np.argsort([np.trace(C) for C in covs])
    info, vec, members = [], [], []          # per object: sum of R^-1, sum of R^-1 z, detection indices
    for i in order:
        z = np.asarray(dets[i]["xy"], float)
        Ri = np.linalg.inv(covs[i])
        best, bd = None, np.inf
        for k in range(len(info)):
            P = np.linalg.inv(info[k])
            e = z - P @ vec[k]
            S = covs[i] + P
            m2 = float(e @ np.linalg.solve(S, e))
            cost = m2 + math.log(np.linalg.det(S))     # likelihood, not Mahalanobis alone: a vague object
            if m2 < gate and cost < bd:                 # (one far or false detection) must not attract the rest
                best, bd = k, cost
        if best is None:
            info.append(Ri.copy())
            vec.append(Ri @ z)
            members.append([i])
        else:
            info[best] += Ri
            vec[best] += Ri @ z
            members[best].append(i)
    merged = True                              # two boxes cannot stand closer than merge_m: one object
    while merged and len(info) > 1:
        merged = False
        Ps = [np.linalg.inv(L) for L in info]
        X = np.array([P @ v for P, v in zip(Ps, vec)])
        D = np.hypot(X[:, None, 0] - X[None, :, 0], X[:, None, 1] - X[None, :, 1])
        np.fill_diagonal(D, np.inf)
        for a in range(len(X)):                 # or the same place within their uncertainty
            for b in range(a + 1, len(X)):
                if D[a, b] < 1.0:
                    e = X[a] - X[b]
                    if float(e @ np.linalg.solve(Ps[a] + Ps[b], e)) < gate:
                        D[a, b] = D[b, a] = 0.0
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if D[i, j] < merge_m:
            i, j = min(i, j), max(i, j)
            info[i], vec[i], members[i] = info[i] + info[j], vec[i] + vec[j], members[i] + members[j]
            del info[j], vec[j], members[j]
            merged = True
    out = []
    for L, v, mem in zip(info, vec, members):
        P = np.linalg.inv(L)
        xy = P @ v
        ix, iy = grid.index(*xy)
        r = max(1, int(round(0.10 / grid.res)))
        nv = int(np.median(looks[max(0, ix - r):ix + r + 1, max(0, iy - r):iy + r + 1]))
        n = len(mem)
        ratio = n / max(nv, n)
        out.append(dict(xy=(float(xy[0]), float(xy[1])), n_det=n, n_views=nv, ratio=ratio,
                        robots=sorted({dets[i]["robot"] for i in mem}), sd_m=float(math.sqrt(np.trace(P) / 2)),
                        kept=n >= min_det and ratio >= min_ratio))
    return out


def score(est, truth, gate_m=0.3, c=0.5):
    E = np.asarray(est, float).reshape(-1, 2)
    T = np.asarray(truth, float).reshape(-1, 2)
    out = dict(n_est=len(E), n_true=len(T), count_err=len(E) - len(T))
    if len(E) and len(T):
        D = np.hypot(E[:, None, 0] - T[None, :, 0], E[:, None, 1] - T[None, :, 1])
        i, j = linear_sum_assignment(np.minimum(D, c))
        ok = D[i, j] <= gate_m
        tp = int(ok.sum())
        err = D[i, j][ok]
        ospa = (np.minimum(D[i, j], c).sum() + c * abs(len(E) - len(T))) / max(len(E), len(T))
        out.update(tp=tp, fp=len(E) - tp, fn=len(T) - tp, mean_err_m=float(err.mean()) if tp else None,
                   max_err_m=float(err.max()) if tp else None, ospa_m=float(ospa),
                   matches=[(int(a), int(b), float(D[a, b])) for a, b in zip(i, j)])
    else:
        out.update(tp=0, fp=len(E), fn=len(T), mean_err_m=None, max_err_m=None,
                   ospa_m=0.0 if not len(E) and not len(T) else c, matches=[])
    p = out["tp"] / out["n_est"] if out["n_est"] else 1.0
    r = out["tp"] / out["n_true"] if out["n_true"] else 1.0
    out.update(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)
    return out
