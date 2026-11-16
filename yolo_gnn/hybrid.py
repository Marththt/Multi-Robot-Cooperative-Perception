"""YOLO bearing rays and their two uses on top of the marker network.

RAYS. Every box segmented by YOLO in a frame gives a horizontal ray from the lens (the median column of its
mask, undistorted and rotated with the calibrated camera pitch and yaw) and a rough monocular range: from the
floor contact of the mask bottom (z = 0) and from the angular height of the box (30 cm). When the two disagree
the bottom is hidden (an obstacle, another box) and the height alone is used. A box must be segmented in most of
the 5 readings of a stop (min_frac) to give a ray: the rays of the readings are grouped by direction.

INJECTION (boxes no marker shows). A ray of a robot is unexplained when no marker detection of that robot and no
box estimated by the marker network lie along it. Two unexplained rays of different robots that cross inside
the area surrounded by the robots, at the ranges both monocular estimates expect (tol_sym), give a box; a third
ray through the same point lets the ranges disagree more (tol_asym). Every supporting ray becomes a detection
node at that point, so the marker network clusters it like any other detection.

CANNOT-LINK (boxes the marker network merges). Two detection nodes that lie on two different rays of the same
robot (bearings at least cl_ray_diff apart) are different boxes; when this holds for cl_robots robots, their
pair probability is set to 0 before the clustering.
"""

import copy
import math
from dataclasses import dataclass

import cv2
import numpy as np

from gnn.dataset import LABEL_RADIUS_M
from yolo_gnn.camera import APOTHEM_M, BOX_HEIGHT_M, CAMERA_HEIGHT_M, camera_rotation


@dataclass
class Params:
    # rays of one frame
    min_pixels: int = 140             # smaller masks are ignored
    border_margin: int = 8            # masks closer to the image side are cut: the floor range is not trusted
    clean_gap_m: float = 0.16         # floor and height ranges this close: bottom of the box visible
    hidden_gap_m: float = 0.20        # floor range this much longer: bottom hidden, height range only
    # rays of one stop (5 readings)
    group_rad: float = 0.05           # rays of different readings closer than this are the same box
    min_frac: float = 0.8             # share of the readings in which the box must be segmented
    # unexplained rays
    marker_rad: float = 0.12          # a marker detection of the same robot within this angle explains a ray
    marker_rad_clean: float = 0.11    # ... with a visible bottom: angle and range must both agree
    marker_range_m: float = 0.25
    est_perp_m: float = 0.10          # a marker-network box within this distance of the ray explains it
    est_range_m: float = 0.30
    # triangulation
    min_sin: float = 0.20             # rays crossing at less than ~12 deg are not intersected
    t_min_m: float = 0.35
    t_max_m: float = 3.0
    clear_m: float = 0.22             # a new box must be this far from every detection and estimated box
    tol_sym: float = 0.18             # both monocular ranges within this of the intersection
    tol_asym_min: float = 0.14        # with a third ray: the better range within this, the other within tol_asym_max
    tol_asym_max: float = 0.38
    support_perp_m: float = 0.08      # a third ray passes this close to the intersection
    support_range_m: float = 0.48
    node_view: float = 0.10           # viewing-angle feature of an injected node
    # cannot-link
    cl_ray_perp_m: float = 0.08       # a node lies on a ray when this close to it
    cl_ray_diff: float = 0.12         # two rays of a robot are different boxes when their bearings differ this much
    cl_robots: int = 2                # robots that must separate the two nodes
    cl_min_sep_m: float = 0.18        # nodes closer than this are never separated
    cl_p: float = 0.3                 # pairs below this probability are left alone
    same_robot_rad: float = 0.15      # same-robot rule: two markers of one robot this far apart in bearing...
    same_robot_m: float = 0.20        # ... and this far apart on the floor are different boxes (no YOLO)
    # stages
    inject: bool = True
    cannot_link: bool = True
    same_robot: bool = False


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def frame_rays(labels, K, D, pose, pitch_rad, p):
    """Rays of the boxes segmented in one frame: list of (world angle, monocular range, bottom visible)."""
    H, W = labels.shape[:2]
    x, y, yaw = pose
    R = camera_rotation(yaw, pitch_rad, 0.0)
    out = []
    for inst in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == inst)
        if len(xs) < p.min_pixels:
            continue
        at_border = xs.min() <= p.border_margin or xs.max() >= W - 1 - p.border_margin
        u = float(np.median(xs))
        v_top, v_bot = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
        pts = np.array([[[u, v_bot]], [[u, v_top]], [[u, 0.5 * (v_top + v_bot)]]], np.float64)
        und = cv2.undistortPoints(pts, K, D).reshape(3, 2)
        bot = R.T @ np.array([und[0, 0], und[0, 1], 1.0])
        mid = R.T @ np.array([und[2, 0], und[2, 1], 1.0])
        ang = math.atan2(mid[1], mid[0])
        d_floor = None
        if bot[2] < -1e-3:
            t = -CAMERA_HEIGHT_M / bot[2]
            d_floor = math.hypot(t * bot[0], t * bot[1]) + APOTHEM_M
        h = abs(und[0, 1] - und[1, 1])
        d_height = BOX_HEIGHT_M / h + APOTHEM_M if h > 0.05 else None
        clean = False
        if d_floor is not None and d_height is not None:
            if abs(d_floor - d_height) <= p.clean_gap_m and not at_border:
                clean, d = True, 0.75 * d_floor + 0.25 * d_height
            elif d_floor > d_height + p.hidden_gap_m:
                d = d_height
            else:
                d = 0.5 * (d_floor + d_height)
        else:
            d = d_floor or d_height or 1.5
        if 0.30 <= d <= 3.2:
            out.append((ang, d, clean))
    return out


def group_rays(per_reading, origin, robot, p):
    """Rays of one robot at one stop from the rays of its readings (a list per reading): a ray is kept when
    the box is segmented in at least min_frac of the readings."""
    flat = [r for rays in per_reading for r in rays]
    n = len(per_reading)
    used = [False] * len(flat)
    out = []
    for i, (a0, _, _) in enumerate(flat):
        if used[i]:
            continue
        grp = []
        for j in range(i, len(flat)):
            if not used[j] and abs(wrap(flat[j][0] - a0)) < p.group_rad:
                grp.append(flat[j])
                used[j] = True
        if len(grp) < max(1, math.ceil(n * p.min_frac - 1e-9)):
            continue
        ang = math.atan2(np.mean([math.sin(g[0]) for g in grp]), np.mean([math.cos(g[0]) for g in grp]))
        out.append(dict(robot=robot, origin=np.asarray(origin, float), ang=ang,
                        u=np.array([math.cos(ang), math.sin(ang)]),
                        d=float(np.median([g[1] for g in grp])),
                        clean=float(np.mean([g[2] for g in grp])) >= 0.8))
    return out


def along(ray, pt):
    """(range along the ray, distance from the ray) of a point."""
    v = np.asarray(pt, float) - ray["origin"]
    t = float(v @ ray["u"])
    return t, float(np.linalg.norm(v - t * ray["u"]))


def explained(ray, scene, est, p):
    """A marker detection of the same robot, or a box of the marker network, lies along the ray."""
    mine = scene["det_robot"] == ray["robot"]
    for pos in scene["pos"][mine]:
        v = pos - ray["origin"]
        dang = abs(wrap(math.atan2(v[1], v[0]) - ray["ang"]))
        if ray["clean"]:
            if dang < p.marker_rad_clean and abs(float(np.linalg.norm(v)) - ray["d"]) < p.marker_range_m:
                return True
        elif dang < p.marker_rad:
            return True
    for b in est:
        t, perp = along(ray, b)
        if t > 0.2 and perp < p.est_perp_m and abs(t - ray["d"]) < p.est_range_m:
            return True
    return False


def intersect(r1, r2, p):
    cross = r1["u"][0] * r2["u"][1] - r1["u"][1] * r2["u"][0]
    if abs(cross) < p.min_sin:
        return None
    dp = r2["origin"] - r1["origin"]
    t1 = (dp[0] * r2["u"][1] - dp[1] * r2["u"][0]) / cross
    t2 = (dp[0] * r1["u"][1] - dp[1] * r1["u"][0]) / cross
    if not (p.t_min_m <= t1 <= p.t_max_m and p.t_min_m <= t2 <= p.t_max_m):
        return None
    return r1["origin"] + t1 * r1["u"], t1, t2


def inject(scene, rays, est, p):
    """The scene with one extra detection node per supporting ray of every triangulated box (a copy), and the
    number of boxes added."""
    est = np.asarray(est, float).reshape(-1, 2)
    dets = scene["pos"].reshape(-1, 2)
    free = [r for r in rays if not explained(r, scene, est, p)]
    centre = scene["poses"][:, :2].mean(0)
    radius = float(np.max(np.linalg.norm(scene["poses"][:, :2] - centre, axis=1)))   # robots surround the boxes

    def clear(pt, boxes):
        return all(len(x) == 0 or np.min(np.linalg.norm(x - pt, axis=1)) >= p.clear_m for x in (dets, boxes))

    cands = []
    for i in range(len(free)):
        for j in range(i + 1, len(free)):
            r1, r2 = free[i], free[j]
            if r1["robot"] == r2["robot"]:
                continue
            hit = intersect(r1, r2, p)
            if hit is None:
                continue
            pt, t1, t2 = hit
            if np.linalg.norm(pt - centre) > radius or not clear(pt, est):
                continue
            e1, e2 = abs(t1 - r1["d"]), abs(t2 - r2["d"])
            supp, robots = [i, j], {r1["robot"], r2["robot"]}
            for k, rk in enumerate(free):
                if k in (i, j) or rk["robot"] in robots:
                    continue
                t, perp = along(rk, pt)
                if p.t_min_m - 0.05 <= t <= p.t_max_m and perp <= p.support_perp_m and abs(t - rk["d"]) <= p.support_range_m:
                    supp.append(k)
                    robots.add(rk["robot"])
            sym = e1 <= p.tol_sym and e2 <= p.tol_sym
            asym = min(e1, e2) <= p.tol_asym_min and max(e1, e2) <= p.tol_asym_max and len(supp) >= 3
            if sym or asym:
                cands.append((e1 + e2 - 0.35 * len(supp), supp, pt))

    cands.sort(key=lambda c: c[0])
    used = set()
    boxes = est.copy()
    nodes = []
    for _, supp, pt in cands:
        if used & set(supp) or not clear(pt, boxes):
            continue
        used |= set(supp)
        boxes = np.vstack([boxes, pt])
        for k in supp:
            q = free[k]["robot"]
            cx, cy, cyaw = scene["poses"][q]
            v = pt - [cx, cy]
            nodes.append((q, float(np.linalg.norm(v)), wrap(cyaw - math.atan2(v[1], v[0])), pt))
    if not nodes:
        return scene, 0
    s = copy.deepcopy(scene)
    new = np.array([n[3] for n in nodes], float)
    gaps = np.linalg.norm(new[:, None, :] - s["truth"][None, :, :], axis=2) if len(s["truth"]) else None
    lab = (np.where(gaps.min(1) <= LABEL_RADIUS_M, gaps.argmin(1), -1) if gaps is not None
           else np.full(len(new), -1))
    s["det_robot"] = np.concatenate([s["det_robot"], [n[0] for n in nodes]]).astype(int)
    s["distance"] = np.concatenate([s["distance"], [n[1] for n in nodes]])
    s["bearing"] = np.concatenate([s["bearing"], [n[2] for n in nodes]])
    s["view"] = np.concatenate([s["view"], np.full(len(nodes), p.node_view)])
    s["pos"] = np.vstack([s["pos"].reshape(-1, 2), new])
    s["label"] = np.concatenate([s["label"], lab]).astype(int)
    return s, len(boxes) - len(est)


def cannot_link(scene, pairs, p_same, rays, p):
    """Pair probabilities with the pairs separated by the rays (and, with same_robot, by the markers) set to 0."""
    pos = scene["pos"].reshape(-1, 2)
    out = np.array(p_same, float).copy()
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if not len(pairs):
        return out
    pi, pj = pairs[:, 0], pairs[:, 1]
    gap = np.linalg.norm(pos[pi] - pos[pj], axis=1)
    if p.same_robot:
        r_of, b_of = np.asarray(scene["det_robot"]), np.asarray(scene["bearing"], float)
        db = np.abs(np.arctan2(np.sin(b_of[pi] - b_of[pj]), np.cos(b_of[pi] - b_of[pj])))
        out[(r_of[pi] == r_of[pj]) & (db > p.same_robot_rad) & (gap >= p.same_robot_m)] = 0.0
    if not p.cannot_link or not rays:
        return out
    robots = sorted({r["robot"] for r in rays})
    on_idx = np.full((len(pos), len(robots)), -1, int)     # per node and robot: the ray the node lies on
    on_ang = np.zeros((len(pos), len(robots)))
    for c, q in enumerate(robots):
        rq = [r for r in rays if r["robot"] == q]
        O = np.array([r["origin"] for r in rq], float)
        U = np.array([r["u"] for r in rq], float)
        V = pos[:, None, :] - O[None]                      # (nodes, rays, 2)
        t = (V * U[None]).sum(2)
        perp = np.linalg.norm(V - t[..., None] * U[None], axis=2)
        perp = np.where((t >= 0.25) & (t <= 3.0) & (perp < p.cl_ray_perp_m), perp, np.inf)
        best = perp.argmin(1)                              # the closest ray (the first one on a tie)
        hit = np.isfinite(perp[np.arange(len(pos)), best])
        on_idx[hit, c] = best[hit]
        on_ang[hit, c] = np.array([r["ang"] for r in rq])[best[hit]]
    a_i, a_j = on_idx[pi], on_idx[pj]
    dang = np.abs(np.arctan2(np.sin(on_ang[pi] - on_ang[pj]), np.cos(on_ang[pi] - on_ang[pj])))
    sep = ((a_i >= 0) & (a_j >= 0) & (a_i != a_j) & (dang >= p.cl_ray_diff)).sum(1)
    out[(out > p.cl_p) & (gap >= p.cl_min_sep_m) & (sep >= p.cl_robots)] = 0.0
    return out
