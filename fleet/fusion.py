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
    """Covariance of a detection's position: range error along the line of sight, bearing error across it (or the
    detection's own sd_along / sd_across, e.g. an IR range)."""
    th = math.atan2(d["xy"][1] - d["lens"][1], d["xy"][0] - d["lens"][0])
    sr = d.get("sd_along") or sd_range[0] + sd_range[1] * d["range"]
    sb = d.get("sd_across") or max(d["range"], 0.05) * sd_bearing_rad
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


# ---------------------------------------------------------------------------------------------- online
class Track:
    _next = 0

    def __init__(self, z, R, t, robot):
        Track._next += 1
        self.id = Track._next
        self.L = np.linalg.inv(R)
        self.v = self.L @ z
        self.n_cam, self.looks, self.n_ir, self.ir_miss = 1, 1, 0, 0
        self.robots = {robot}
        self.state = "tentative"
        self.t0 = self.t_last = t

    @property
    def xy(self):
        return np.linalg.solve(self.L, self.v)

    @property
    def P(self):
        return np.linalg.inv(self.L)

    def add(self, z, R):
        Ri = np.linalg.inv(R)
        self.L = self.L + Ri
        self.v = self.v + Ri @ z

    @property
    def ratio(self):
        return self.n_cam / max(self.looks, self.n_cam)


class Tracker:
    """Online fusion at the captain: the object list is updated with every report as it arrives.

    Camera report (one frame of one robot: its lens pose, its YOLO detections and where it looked):
      - every live object the frame looked at gets one more look (the negative evidence);
      - detections <-> objects by the Hungarian algorithm (no two detections of one frame on one object), cost =
        Mahalanobis distance + log det of the innovation covariance, gated at GATE_CHI2; an object is updated in
        information form, a detection left over starts a tentative object.
    IR report (the gimbal aimed the IR at one object): a precise range along the beam; within the gate it updates
      the object (n_ir), otherwise the object counts an IR miss (the beam found something else, or nothing).
    State of an object, after every report:
      confirmed   n_cam >= min_det, n_cam / looks >= confirm_ratio, and the IR agrees: an IR range on it, or the
                  IR map is occupied where it is (the shape is there) ; stays confirmed unless deleted
      deleted     looked at >= delete_looks times and detected in < delete_ratio of them, or IR missed twice and
                  never hit
      tentative   otherwise
    Objects closer than MERGE_M, or within the gate of each other, are merged (one box seen twice).
    """

    def __init__(self, sd_range=(0.02, 0.15), sd_bearing_deg=1.0, gate=GATE_CHI2, merge_m=MERGE_M, min_det=3,
                 confirm_ratio=0.4, delete_looks=6, delete_ratio=0.15):
        self.sd_range, self.sb = sd_range, math.radians(sd_bearing_deg)
        self.gate, self.merge_m = gate, merge_m
        self.min_det, self.confirm_ratio = min_det, confirm_ratio
        self.delete_looks, self.delete_ratio = delete_looks, delete_ratio
        self.tracks = []

    def live(self):
        return [k for k in self.tracks if k.state in ("tentative", "confirmed")]

    def get(self, tid):
        for k in self.tracks:
            if k.id == tid:
                return k
        return None

    def confirmed(self):
        return [k for k in self.tracks if k.state == "confirmed"]

    def tentative(self):
        return [k for k in self.tracks if k.state == "tentative"]

    def camera(self, robot, t, dets, looked, occ_near):
        """dets: dicts with xy, range, lens (and sd_along / sd_across); looked(xy) -> bool: this frame looked
        there; occ_near(xy) -> bool: the IR map is occupied there."""
        live = self.live()
        for k in live:
            if looked(k.xy):
                k.looks += 1
        if dets:
            Z = [np.asarray(d["xy"], float) for d in dets]
            Rs = [det_cov(d, self.sd_range, self.sb) for d in dets]
            big = 1e6
            C = np.full((len(dets), len(live)), big)
            for i, (z, R) in enumerate(zip(Z, Rs)):
                for j, k in enumerate(live):
                    S = R + k.P
                    e = z - k.xy
                    m2 = float(e @ np.linalg.solve(S, e))
                    if m2 < self.gate:
                        C[i, j] = m2 + math.log(np.linalg.det(S))
            used = set()
            if live:
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] < big:
                        live[j].add(Z[i], Rs[i])
                        live[j].n_cam += 1
                        live[j].robots.add(robot)
                        live[j].t_last = t
                        used.add(i)
            for i in range(len(dets)):
                if i not in used:
                    self.tracks.append(Track(Z[i], Rs[i], t, robot))
        self.merge()
        self.states(occ_near)

    def ir(self, tid, z, R, occ_near):
        k = self.get(tid)
        if k is None or k.state not in ("tentative", "confirmed"):
            return False
        e = z - k.xy
        if float(e @ np.linalg.solve(R + k.P, e)) < self.gate:
            k.add(z, R)
            k.n_ir += 1
            ok = True
        else:
            k.ir_miss += 1
            ok = False
        self.merge()
        self.states(occ_near)
        return ok

    def states(self, occ_near):
        for k in self.live():
            if (k.looks >= self.delete_looks and k.ratio < self.delete_ratio) or (k.ir_miss >= 2 and k.n_ir == 0):
                k.state = "deleted"
            elif k.state != "confirmed" and k.n_cam >= self.min_det and k.ratio >= self.confirm_ratio and \
                    (k.n_ir >= 1 or occ_near(k.xy)):
                k.state = "confirmed"

    def merge(self):
        live = self.live()
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                A, B = live[a], live[b]
                if A.state not in ("tentative", "confirmed") or B.state not in ("tentative", "confirmed"):
                    continue
                e = A.xy - B.xy
                if float(np.hypot(*e)) < self.merge_m or float(e @ np.linalg.solve(A.P + B.P, e)) < self.gate:
                    keep, drop = (A, B) if A.n_cam + 3 * A.n_ir >= B.n_cam + 3 * B.n_ir else (B, A)
                    keep.L, keep.v = keep.L + drop.L, keep.v + drop.v
                    keep.n_cam += drop.n_cam
                    keep.looks = max(keep.looks, drop.looks)
                    keep.n_ir += drop.n_ir
                    keep.ir_miss += drop.ir_miss
                    keep.robots |= drop.robots
                    if drop.state == "confirmed":
                        keep.state = "confirmed"
                    drop.state = "merged"
