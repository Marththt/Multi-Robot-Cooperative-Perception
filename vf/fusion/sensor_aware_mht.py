"""MHT with sensor model: the MHT of data_association.py, given the same per-detection inputs as the GNN.

The plain MHT sees every report as a point (x, y) with an isotropic error, a flat p_D inside the field of
view and no false alarms. This version reads, for every report, what the GNN reads (distance, line of sight
and viewing angle of the marker, plus the robot pose) and turns it into probabilities with the sensor model
fitted on the real runs (config/sensor_model.json, written by sim/calibrate.py):

  position     the mean error along the line of sight, m0 + m1 d + m2 cos(a), is removed from the report;
               the covariance is anisotropic and grows with d: sd along = s0 + s1 d, sd across = t0 + t1 d,
               plus the per-robot range / bearing bias (bias_sd_*_rel * d)
  orientation  a box is an equilateral triangle, so its orientation is known modulo 120 deg. A report with
               viewing angle a seen by a camera with yaw psi says the reported face points along
               psi + pi +- a; every object keeps a few candidate orientations (with weights and spread) and
               two reports of the same box must agree on them. The reported angle is noisy with heavy tails:
               a Gaussian core plus a share of outliers (VIEW_SD_RAD, VIEW_OUTLIER_SHARE)
  p_D          per robot and object, from the fitted logistic sigmoid(w . [1, a, a^2, d, occluded]) of the
               best face, a predicted from the candidate orientations, occluded = the line of sight to the
               marker passes through another object of the same hypothesis; zero outside the field of view
               and the distance limits (with a soft edge for the position error), and scaled by the share
               of lines of sight blocked by obstacles, which the method cannot see (UNMODELLED_OCCLUSION)
  second face  a robot may report the same box twice (two faces), with probability two_face_q * p_D(second face)
  clutter      the fitted false-detection model: false_rate per capture, raw distance and bearing drawn from
               the measured bands, converted to a density in the floor plane

Everything else is the MHT of data_association.py: robots processed one after the other as scans, the same
gate, the same birth prior (n_prior objects over the surveilled area), the same number of hypotheses. The
input data are the same raw records as for every other method and for the GNN.
"""

import json
import math
import os

import numpy as np

from vf.fusion.data_association import GATE_THRESHOLD, top_k_combinations

APOTHEM_M = 0.087                    # box centre -> marker (centre of a face)
OCCLUDER_RADIUS_M = 0.13             # another box blocks a line of sight passing this close to its centre
FACES = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
PERIOD = 2 * math.pi / 3             # orientation of a triangular box is defined modulo 120 deg
MAX_CANDIDATES = 4
EDGE_RAD = 0.02                      # soft edge of the field of view [rad]
EDGE_M = 0.03                        # soft edge of the distance limits [m]
# Share of lines of sight blocked by something the method cannot know about (an obstacle: only boxes are
# estimated). Measured on the training distribution of the GNN (sim/specs/train_general.json: 0-3 obstacles,
# 7,587 of 77,427 box-robot pairs in the field of view blocked = 9.8%), i.e. the same knowledge the GNN
# learns from. Every p_D is multiplied by (1 - this).
UNMODELLED_OCCLUSION = 0.10
# Noise of the reported viewing angle, fitted by maximum likelihood on the 607 real reports of L4-L11 (each
# true box with its best orientation): a Gaussian core of sd 0.10 rad plus 10% outliers spread uniformly.
# A plain Gaussian (view_noise_sd_rad of the simulator) would make one bad angle split a box in two.
VIEW_SD_RAD = 0.10
VIEW_OUTLIER_SHARE = 0.10
P_MIN = 1e-6
VIEW_NEW_DENSITY = 3.0 / math.pi     # best-face angle of a box of unknown orientation: uniform over 0-60 deg
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "config", "sensor_model.json")

_MODEL = None


def set_model(path, real_terms=True, clutter_factor=1.0):
    """Use the fitted sensor model at `path` (sim/calibrate.py output) for every later call (see FittedSensor
    for real_terms and clutter_factor)."""
    global _MODEL
    _MODEL = FittedSensor(path, real_terms=real_terms, clutter_factor=clutter_factor)


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = FittedSensor(DEFAULT_MODEL)
    return _MODEL


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _wrap_period(a):
    """Angle difference modulo 120 deg, to (-60, 60] deg."""
    return (np.asarray(a) + PERIOD / 2) % PERIOD - PERIOD / 2


def _normal(x, sd):
    return np.exp(-0.5 * (x / sd) ** 2) / (math.sqrt(2 * math.pi) * sd)


class FittedSensor:
    """The fitted sensor model, read from the JSON of sim/calibrate.py.

    real_terms: add the two terms for what the simulator does not contain (viewing-angle outliers,
    lines of sight blocked by unknown obstacles; module constants). True for real data; False in simulation,
    where the model then matches the simulator exactly (the viewing-angle noise is the file's Gaussian).
    clutter_factor multiplies the false-detection rate ("MHT, more clutter")."""

    def __init__(self, path, real_terms=True, clutter_factor=1.0):
        with open(path) as f:
            m = json.load(f)
        self.half_fov = math.radians(m.get("hfov_deg", 88.0)) / 2
        self.r_min = float(m.get("r_min_m", 0.15))
        self.r_max = float(m.get("r_max_m", 3.0))
        self.w = np.asarray(m["detect_w"], float)
        self.two_face_q = float(m.get("two_face_q", 0.0))
        self.real_terms = bool(real_terms)
        self.view_sd = VIEW_SD_RAD if real_terms else float(m.get("view_noise_sd_rad", VIEW_SD_RAD))
        self.view_outliers = VIEW_OUTLIER_SHARE if real_terms else 0.0
        self.unmodelled_occlusion = UNMODELLED_OCCLUSION if real_terms else 0.0
        self.along_mean = [float(v) for v in m["along_mean"]]
        self.along_sd = [float(v) for v in m["along_sd"]]
        self.across_sd = [float(v) for v in m["across_sd"]]
        self.bias_along = float(m.get("bias_sd_along_rel", 0.0))
        self.bias_across = float(m.get("bias_sd_across_rel", 0.0))
        self.false_rate = float(m.get("false_rate", 0.0)) * float(clutter_factor)
        self.near_share = float(m.get("false_near_share", 0.0))
        self.near = [float(v) for v in m.get("false_near_range_m", [1.6, 3.0])]
        self.far = [float(v) for v in m.get("false_far_range_m", [4.4, 7.4])]
        self.edge_share = float(m.get("false_edge_share", 0.0))
        self.edge_band = [math.radians(float(v)) for v in m.get("false_edge_band_deg", [38.0, 44.0])]
        self.range_fit = [float(v) for v in m.get("range_fit", [1.0, 0.0])]
        self.bearing_fit = [float(v) for v in m.get("bearing_fit", [1.0, 0.0])]

    # ------------------------------------------------------------ one report
    def measurement(self, cam, z, view):
        """Report z seen from camera cam=(x, y, yaw) -> (corrected position, covariance R).
        The mean error along the line of sight is removed (it depends on the viewing angle)."""
        u = np.asarray(z, float) - np.asarray(cam[:2], float)
        d = float(np.hypot(*u))
        u = u / max(d, 1e-9)
        c0, c1, c2 = self.along_mean
        along = c0 + c1 * d + (c2 * math.cos(view) if view is not None else c2 * 0.85)
        s_al = max(1e-3, self.along_sd[0] + self.along_sd[1] * d)
        s_ac = max(1e-3, self.across_sd[0] + self.across_sd[1] * d)
        var_al = s_al ** 2 + (self.bias_along * d) ** 2
        var_ac = s_ac ** 2 + (self.bias_across * d) ** 2
        if view is None:
            var_al += (c2 * 0.15) ** 2
        n = np.array([-u[1], u[0]])
        R = var_al * np.outer(u, u) + var_ac * np.outer(n, n)
        return np.asarray(z, float) - along * u, R

    def clutter_density(self, cam, z):
        """Density [1/m^2] of a lone false detection at z for a camera at cam (false_rate per capture,
        raw distance and bearing drawn from the fitted bands)."""
        x, y, yaw = cam
        d = float(math.hypot(z[0] - x, z[1] - y))
        b = float(_wrap(yaw - math.atan2(z[1] - y, z[0] - x)))
        A, B = self.range_fit
        k, b0 = self.bearing_fit
        dr, br = A * d + B, k * b + b0
        pd = 0.0
        if self.near[0] <= dr <= self.near[1]:
            pd += self.near_share / (self.near[1] - self.near[0])
        if self.far[0] <= dr <= self.far[1]:
            pd += (1 - self.near_share) / (self.far[1] - self.far[0])
        half = self.half_fov
        pb = 0.0
        if self.edge_band[0] <= abs(br) <= self.edge_band[1]:
            pb += self.edge_share / (2 * (self.edge_band[1] - self.edge_band[0]))
        if abs(br) <= half:
            pb += (1 - self.edge_share) / (2 * half)
        dens = self.false_rate * pd * pb * A * k / max(d, 1e-3)
        # floor: a small share of the false rate spread over the field of view, so that a false report
        # outside the fitted bands is never impossible
        floor = 0.02 * max(self.false_rate, 1e-3) / (half * self.r_max ** 2)
        return max(dens, floor)

    # ------------------------------------------------------------ prediction
    def predict(self, cam, C, TH, occluders, self_index=None):
        """What camera cam=(x, y, yaw) would see of objects at C (n, 2) with candidate orientations
        TH (n, K). occluders: (m, 2) centres of the objects that can block the line of sight;
        self_index[i] = row of object i in occluders (or -1).
        Returns p_best, a_best, p_second, a_second, all (n, K)."""
        n, K = TH.shape
        cx, cy, yaw = cam
        cam_xy = np.array([cx, cy])
        rel = C - cam_xy
        d = np.hypot(rel[:, 0], rel[:, 1])
        bear = _wrap(yaw - np.arctan2(rel[:, 1], rel[:, 0]))
        soft = (_sigmoid((self.half_fov - np.abs(bear)) / EDGE_RAD) * _sigmoid((self.r_max - d) / EDGE_M)
                * _sigmoid((d - self.r_min) / EDGE_M))
        N = TH[:, :, None] + FACES[None, None, :]                                     # (n, K, 3)
        M = C[:, None, None, :] + APOTHEM_M * np.stack([np.cos(N), np.sin(N)], -1)     # (n, K, 3, 2)
        to_cam = np.arctan2(cy - M[..., 1], cx - M[..., 0])
        seen = np.cos(_wrap(to_cam - N)) > 0.05
        a = np.abs(_wrap(yaw + np.pi - N))
        occ = np.zeros(N.shape)
        if len(occluders):
            v = M - cam_xy                                                             # (n, K, 3, 2)
            w = occluders - cam_xy                                                     # (m, 2)
            vv = np.maximum(np.sum(v * v, -1), 1e-9)                                   # (n, K, 3)
            t = np.einsum("nkfc,mc->nkfm", v, w) / vv[..., None]                       # (n, K, 3, m)
            dist = np.hypot(w[None, None, None, :, 0] - t * v[..., None, 0],
                            w[None, None, None, :, 1] - t * v[..., None, 1])
            hit = (t > 0) & (t < 1) & (dist < OCCLUDER_RADIUS_M)
            if self_index is not None:
                own = np.zeros((n, len(occluders)), bool)
                for i, s in enumerate(self_index):
                    if s >= 0:
                        own[i, s] = True
                hit &= ~own[:, None, None, :]
            occ = hit.any(-1).astype(float)
        p = _sigmoid(self.w[0] + self.w[1] * a + self.w[2] * a * a + self.w[3] * d[:, None, None]
                     + self.w[4] * occ) * seen
        order = np.argsort(-p, axis=-1)
        ps = np.take_along_axis(p, order, -1)
        as_ = np.take_along_axis(a, order, -1)
        keep = soft[:, None] * (1.0 - self.unmodelled_occlusion)
        return ps[..., 0] * keep, as_[..., 0], ps[..., 1] * keep, as_[..., 1]

    def p_best_from(self, cams, c, th, occluders):
        """p_D of the best face of ONE object at c (2,) with candidate orientations th (K,), for several
        cameras cams (P, 3) at once (occluders as in predict, the object itself not among them).
        Returns (P, K)."""
        cams = np.asarray(cams, float)
        cxy = cams[:, :2]
        yaw = cams[:, 2]
        rel = c[None, :] - cxy
        d = np.hypot(rel[:, 0], rel[:, 1])                                             # (P,)
        bear = _wrap(yaw - np.arctan2(rel[:, 1], rel[:, 0]))
        soft = (_sigmoid((self.half_fov - np.abs(bear)) / EDGE_RAD) * _sigmoid((self.r_max - d) / EDGE_M)
                * _sigmoid((d - self.r_min) / EDGE_M))
        N = th[:, None] + FACES[None, :]                                               # (K, 3)
        M = c[None, None, :] + APOTHEM_M * np.stack([np.cos(N), np.sin(N)], -1)       # (K, 3, 2)
        to_cam = np.arctan2(cxy[:, None, None, 1] - M[None, ..., 1], cxy[:, None, None, 0] - M[None, ..., 0])
        seen = np.cos(_wrap(to_cam - N[None])) > 0.05                                  # (P, K, 3)
        a = np.abs(_wrap(yaw[:, None, None] + np.pi - N[None]))
        occ = np.zeros(seen.shape)
        if len(occluders):
            v = M[None] - cxy[:, None, None, :]                                        # (P, K, 3, 2)
            w = occluders[None, :, :] - cxy[:, None, :]                                # (P, m, 2)
            vv = np.maximum(np.sum(v * v, -1), 1e-9)
            t = np.einsum("pkfc,pmc->pkfm", v, w) / vv[..., None]                      # (P, K, 3, m)
            dx = w[:, None, None, :, 0] - t * v[..., None, 0]
            dy = w[:, None, None, :, 1] - t * v[..., None, 1]
            occ = ((t > 0) & (t < 1) & (np.hypot(dx, dy) < OCCLUDER_RADIUS_M)).any(-1).astype(float)
        p = _sigmoid(self.w[0] + self.w[1] * a + self.w[2] * a * a + self.w[3] * d[:, None, None]
                     + self.w[4] * occ) * seen
        return p.max(-1) * soft[:, None] * (1.0 - self.unmodelled_occlusion)


# ====================================================================
# Objects: (m, P, thetas (K,), weights (K,), sds (K,), history)
# ====================================================================

def _report_orientations(cam, view):
    """The two orientations (mod 120 deg) compatible with a report of viewing angle `view`."""
    base = cam[2] + math.pi
    return np.mod(np.array([base + view, base - view]), PERIOD)


def _new_orientation(cam, view, sd):
    if view is None:
        th = np.linspace(0, PERIOD, MAX_CANDIDATES, endpoint=False)
        return th, np.full(MAX_CANDIDATES, 1.0 / MAX_CANDIDATES), np.full(MAX_CANDIDATES, PERIOD / 4)
    th = _report_orientations(cam, view)
    return th, np.array([0.5, 0.5]), np.array([sd, sd])


def _update_orientation(thetas, weights, sds, cam, view, sd, outliers=VIEW_OUTLIER_SHARE):
    """Combine the candidate orientations of an object with those of a new report of it. The reported angle
    may be an outlier (share `outliers`, uniform over the period): then the candidate is kept unchanged."""
    if view is None:
        return thetas, weights, sds
    out = []
    for th_c, w_c, s_c in zip(thetas, weights, sds):
        out.append((w_c * outliers / PERIOD, th_c, s_c))
        for th_r in _report_orientations(cam, view):
            diff = float(_wrap_period(th_r - th_c))
            s2 = s_c ** 2 + sd ** 2
            like = math.exp(-0.5 * diff * diff / s2) / math.sqrt(2 * math.pi * s2)
            g = s_c ** 2 / s2
            out.append((w_c * (1 - outliers) * 0.5 * like, (th_c + g * diff) % PERIOD,
                        math.sqrt(s_c ** 2 * sd ** 2 / s2)))
    # candidates that ended at the same orientation are merged
    out.sort(key=lambda v: -v[0])
    merged = []
    for w, th, s in out:
        for i, (w2, th2, s2) in enumerate(merged):
            if abs(float(_wrap_period(th - th2))) < 0.5 * min(s, s2):
                merged[i] = (w2 + w, th2, s2)
                break
        else:
            merged.append((w, th, s))
    out = sorted(merged, key=lambda v: -v[0])[:MAX_CANDIDATES]
    total = sum(v[0] for v in out)
    if total <= 0:
        return thetas, weights, sds
    keep = [v for v in out if v[0] >= 1e-3 * out[0][0]]
    total = sum(v[0] for v in keep)
    return (np.array([v[1] for v in keep]), np.array([v[0] / total for v in keep]),
            np.array([v[2] for v in keep]))


def _pad(objects):
    """Candidate orientations of all objects as padded (n, K) arrays (padding weight 0)."""
    n = len(objects)
    TH = np.zeros((n, MAX_CANDIDATES))
    W = np.zeros((n, MAX_CANDIDATES))
    SD = np.ones((n, MAX_CANDIDATES))
    for i, o in enumerate(objects):
        k = len(o[2])
        TH[i, :k], W[i, :k], SD[i, :k] = o[2], o[3], o[4]
    return TH, W, SD


def _gauss(z, m, S):
    d = z - m
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    q = (S[1, 1] * d[0] * d[0] - 2 * S[0, 1] * d[0] * d[1] + S[0, 0] * d[1] * d[1]) / det
    return q, math.exp(-0.5 * q) / (2 * math.pi * math.sqrt(det))


def _kalman(m, P, z, R):
    S = P + R
    K = P @ np.linalg.inv(S)
    return m + K @ (z - m), (np.eye(2) - K) @ P


# ====================================================================
# MHT with sensor model
# ====================================================================

def mht_sensor(order, reports, poses, sensor, meta=None, n_hypotheses=100, gate_threshold=GATE_THRESHOLD,
               model=None, max_nodes=None):
    """Same interface as data_association.mht, plus
    meta: {robot_id: [dict(view=viewing angle [rad] or None, cam=(x, y, yaw) or None), ...]}, one entry per
    report, in the order of reports[robot_id] (cam = camera pose of the frame, default the robot pose).
    sensor (SensorModel) only gives the birth prior (n_prior over the surveilled area), as for the plain MHT;
    everything about the camera comes from the fitted model.
    max_nodes: budget of the k-best search per hypothesis and robot (data_association.top_k_combinations);
    None = exact search. The result's extra holds search_nodes and search_cut (1 if a budget was reached)."""
    fm = model or get_model()
    meta = meta or {}
    log_prior = math.log(sensor.prior_density())
    log_view_new = math.log(VIEW_NEW_DENSITY)
    log_view_false = math.log(1.0 / 1.6)     # viewing angle of a false detection: U(0, 1.6 rad)
    hypotheses = [(0.0, [])]
    seen_cams = []
    search = {}

    for rid in order:
        cam_r = poses.get(rid)
        if cam_r is None:
            continue
        raw = [np.asarray(z, float) for z in reports.get(rid, [])]
        info = list(meta.get(rid, [])) + [{}] * (len(raw) - len(meta.get(rid, [])))
        rep = []
        for idx, (z, inf) in enumerate(zip(raw, info)):
            cam = tuple(inf.get("cam") or cam_r)
            view = inf.get("view")
            zc, R = fm.measurement(cam, z, view)
            rep.append(dict(z=zc, R=R, cam=cam, view=view, lc=math.log(fm.clutter_density(cam, z)), idx=idx))
        # best face first: a robot that reports a box twice reports its most visible face first
        rep.sort(key=lambda r: (r["view"] if r["view"] is not None else 0.0))
        has_view = [r["view"] is not None for r in rep]

        children = []
        updated = {}                  # (id of the object, report) -> updated object, shared by the hypotheses
        for log_w, objects in hypotheses:
            n = len(objects)
            centres = np.array([o[0] for o in objects]) if n else np.zeros((0, 2))
            if n:
                TH, W, SD = _pad(objects)
                pb, ab, p2, a2 = fm.predict(cam_r, centres, TH, centres, self_index=list(range(n)))
                pd_obj = np.clip(np.sum(W * pb, 1), P_MIN, 1 - P_MIN)
            else:
                pd_obj = np.zeros(0)
            base = log_w + float(np.sum(np.log(1.0 - pd_obj)))

            options, births = [], []
            for j, r in enumerate(rep):
                o = [(r["lc"] + (log_view_false if has_view[j] else 0.0), None, ("c", None))]
                # a new object, never reported by the robots before
                th_n, w_n, sd_n = _new_orientation(r["cam"], r["view"], fm.view_sd)
                pb1 = fm.p_best_from([r["cam"]] + seen_cams, r["z"], th_n, centres) @ w_n
                p_own = float(pb1[0])
                if p_own > P_MIN:
                    log_new = (log_prior + math.log(p_own) + (log_view_new if has_view[j] else 0.0)
                               + float(np.sum(np.log(np.maximum(1.0 - pb1[1:], P_MIN)))))
                    o.append((log_new, None, ("n", None)))
                births.append((th_n, w_n, sd_n))
                # an object of the hypothesis: first report of it by this robot, or its second face
                for t, (m, P, _th, _w, _s, _h) in enumerate(objects):
                    S = P + r["R"]
                    q, g = _gauss(r["z"], m, S)
                    if q > gate_threshold:
                        continue
                    if has_view[j]:
                        sv = np.sqrt(fm.view_sd ** 2 + SD[t] ** 2)
                        eps = fm.view_outliers
                        l1 = (1 - eps) * _normal(r["view"] - ab[t], sv) + eps * VIEW_NEW_DENSITY
                        l2 = (1 - eps) * _normal(r["view"] - a2[t], sv) + eps * VIEW_NEW_DENSITY
                        det1 = float(np.sum(W[t] * pb[t] * l1))
                        det2 = float(np.sum(W[t] * fm.two_face_q * p2[t] * l2))
                    else:
                        det1 = float(np.sum(W[t] * pb[t]))
                        det2 = float(np.sum(W[t] * fm.two_face_q * p2[t]))
                    lg = math.log(g + 1e-300)
                    if det1 > 0:
                        o.append((math.log(det1) + lg - math.log(1.0 - pd_obj[t]), t, ("t", t)))
                    if det2 > 0:
                        o.append((math.log(det2) + lg, ("f", t), ("f", t), t))
                options.append(o)

            for total, choices in top_k_combinations(options, n_hypotheses, max_nodes=max_nodes, stats=search):
                new_objects = list(objects)
                for j, (kind, t) in enumerate(choices):
                    r = rep[j]
                    if kind in ("t", "f"):
                        obj = new_objects[t]
                        key = (id(obj), j)
                        if key not in updated:
                            m, P, th, w, s, hist = obj
                            m2, P2 = _kalman(m, P, r["z"], r["R"])
                            th2, w2, s2 = _update_orientation(th, w, s, r["cam"], r["view"], fm.view_sd,
                                                          fm.view_outliers)
                            updated[key] = ((m2, P2, th2, w2, s2, hist + [(rid, r["idx"])]), obj)
                        new_objects[t] = updated[key][0]
                    elif kind == "n":
                        th_n, w_n, sd_n = births[j]
                        new_objects.append((r["z"].copy(), r["R"].copy(), th_n, w_n, sd_n, [(rid, r["idx"])]))
                children.append((base + total, new_objects))
        if not children:
            continue
        children.sort(key=lambda v: -v[0])
        hypotheses = [h for h in children[:n_hypotheses] if h[0] >= children[0][0] - 14.0]
        seen_cams.append(cam_r)

    maximum = hypotheses[0][0]
    weights = np.array([math.exp(w - maximum) for w, _ in hypotheses])
    weights /= weights.sum()
    n_distr = {}
    for p, (_, objs) in zip(weights, hypotheses):
        n_distr[len(objs)] = n_distr.get(len(objs), 0.0) + float(p)
    best = hypotheses[0][1]
    return dict(
        objects=[tuple(float(v) for v in o[0]) for o in best],
        n=len(best),
        extra=dict(p_best_hypothesis=float(weights[0]), n_distribution=n_distr,
                   reports_per_object=[o[5] for o in best], search_nodes=search.get("nodes", 0),
                   search_cut=search.get("cut", 0)))
