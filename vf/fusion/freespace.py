"""Free space from the IR beams: the negative evidence the fusion does not use.

Every IR reading is a ray on the floor plane: origin (the IR), heading (where the gimbal points) and the
horizontal range at which the beam met something. It says two things:

  hit    something is at the end of the ray
  free   NOTHING is along the ray before it (the beam went through)

The fusion (MHT, graph network) only uses the first. This module adds the second, without touching them:

  ray_from_state(s), rays_from_states(states)   rays from IR live states / log rows (acquisition/ir_scan.py format)
  evidence(p, rays)                 at a point: n_free (rays that went through it), n_hit (rays ending there)
  log_ratio(n_hit, n_free)          log P(evidence | box at p) / P(evidence | nothing at p)
  filter_detections(dets, rays)     drops detections that other rays went through (before the fusion: the
                                    MHT and the graph network then run unchanged on what is left)
  filter_objects(objs, rays)        drops fused boxes the rays contradict (after the fusion)
  detection_features(dets, rays)    [n_free, n_hit, log_ratio] per detection: extra input of a graph network
  in_arena(p, arena, margin)        a detection on the border of the arena is a wall, not a box (no free space)
  occupancy_grid(rays, arena)       log-odds occupancy grid, for the picture

The test is on box CENTRES: a detection (the point of a face the beam hit) is moved one apothem along its beam
first. A ray "goes through" a centre c when c is within FREE_HALF_WIDTH_M of the beam axis (the beam would surely
meet a box there) and the beam ended at least FREE_MARGIN_M beyond c (a box there would have stopped it about one
apothem BEFORE c; the margin covers that, the range noise and a fused position that is a face rather than a
centre). A ray "hits" c when its end point is within HIT_RADIUS_M of c. Other rays count for neither.
"""

import math

import numpy as np

APOTHEM_M = 0.068              # box centre -> face (boxes of acquisition/ir_ranging.py)
FREE_HALF_WIDTH_M = 0.05       # a box centred this close to the beam axis would surely be hit (< apothem)
FREE_MARGIN_M = 0.10           # the beam must end this far beyond the centre (it would stop ~1 apothem before)
HIT_RADIUS_M = 0.20            # an end point this close to p supports a box at p (the fusion's grouping)
P_HIT = 0.90                   # P(a ray aimed near a box ends on it)
P_FALSE_HIT = 0.10             # P(a ray ends at p with no box there): edge mix, a slipped beam
P_FREE_GIVEN_BOX = 0.03        # P(a ray goes through p although a box is there): aim / pose error


class Ray:
    __slots__ = ("o", "u", "d", "robot", "t")

    def __init__(self, origin, heading, d, robot=None, t=None):
        self.o = np.asarray(origin[:2], float)
        self.u = np.array([math.cos(heading), math.sin(heading)])
        self.d = float(d)
        self.robot, self.t = robot, t

    @property
    def end(self):
        return self.o + self.d * self.u


def ray_from_state(s, max_free_m=None):
    """The ray of one IR state (one reading), or None. The range is the horizontal distance of the detection,
    or, for a valid reading that gave no detection (the beam met the floor, or went above the boxes),
    slant x cos(pitch): the beam went at least that far. A reading with no valid range gives no ray, unless
    max_free_m: then the beam went through max_free_m without meeting anything."""
    pose = s.get("pose")
    if not pose:
        return None
    dets = s.get("detections") or []
    if dets and dets[0].get("distance") is not None:
        d = float(dets[0]["distance"])
    elif s.get("ir_slant_m") is not None:
        pitch = math.radians((s.get("gimbal") or [0, 0])[1] or 0.0)
        d = float(s["ir_slant_m"]) * math.cos(pitch)
    elif max_free_m:
        d = float(max_free_m)
    else:
        return None
    return Ray(pose, pose[2], d, s.get("robot_id"), s.get("t"))


def rays_from_states(states, max_free_m=None):
    """Rays of IR states (acquisition/ir_scan.py states or log rows), the ones with a range."""
    return [r for r in (ray_from_state(s, max_free_m) for s in states) if r is not None]


def _along_across(p, rays):
    if not rays:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    O = np.array([r.o for r in rays])
    U = np.array([r.u for r in rays])
    D = np.array([r.d for r in rays])
    w = np.asarray(p, float)[None, :2] - O
    t = np.sum(w * U, 1)
    c = np.abs(w[:, 0] * U[:, 1] - w[:, 1] * U[:, 0])
    return t, c, D


def evidence(p, rays, exclude=None):
    """(n_free, n_hit) at point p. exclude: a ray (by identity) left out, e.g. the one that made the detection."""
    rs = [r for r in rays if r is not exclude]
    t, c, D = _along_across(p, rs)
    if not len(t):
        return 0, 0
    free = (t > 0) & (c < FREE_HALF_WIDTH_M) & (D > t + FREE_MARGIN_M)
    ends = np.array([r.end for r in rs])
    hit = np.hypot(*(ends - np.asarray(p, float)[:2]).T) < HIT_RADIUS_M
    return int(free.sum()), int(hit.sum())


def log_ratio(n_hit, n_free):
    """log P(n_hit, n_free | box at p) / P(... | nothing at p), rays independent."""
    return (n_hit * math.log(P_HIT / P_FALSE_HIT)
            + n_free * math.log(P_FREE_GIVEN_BOX / 1.0))


def contradicted(p, rays, exclude=None, threshold=0.0):
    nf, nh = evidence(p, rays, exclude)
    return nf > 0 and log_ratio(nh, nf) < threshold, nf, nh


def centre_of(d):
    """Box centre implied by a detection: its position moved one apothem along its beam (if it has one)."""
    p = np.asarray(d["position"], float)[:2]
    r = d.get("ray")
    return p if r is None else p + APOTHEM_M * r.u


def filter_detections(dets, rays, threshold=0.0):
    """dets: [{'position': [x, y], 'ray': Ray or None, ...}]. Keeps the ones not contradicted by the OTHER rays
    (the ray that made a detection supports it, so it is left out of its own test). Returns (kept, dropped)."""
    kept, dropped = [], []
    for d in dets:
        bad, nf, nh = contradicted(centre_of(d), rays, exclude=d.get("ray"), threshold=threshold)
        (dropped if bad else kept).append(dict(d, n_free=nf, n_hit=nh))
    return kept, dropped


def filter_objects(objects, rays, threshold=0.0):
    """objects: [[x, y], ...] from a fusion. Keeps the ones the rays do not contradict. Returns (kept, dropped)."""
    kept, dropped = [], []
    for o in objects:
        bad, _, _ = contradicted(o, rays, threshold=threshold)
        (dropped if bad else kept).append(o)
    return kept, dropped


def detection_features(dets, rays):
    """(n, 3) array: n_free, n_hit (other rays) and log_ratio for every detection (input of a graph network)."""
    out = []
    for d in dets:
        nf, nh = evidence(centre_of(d), rays, exclude=d.get("ray"))
        out.append([nf, nh, log_ratio(nh, nf)])
    return np.array(out, float).reshape(-1, 3)


def in_arena(p, arena, margin):
    """False for a point within margin of the arena border or outside it: the beam read a wall, not a box."""
    x0, x1, y0, y1 = arena
    return x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin


def occupancy_grid(rays, arena, res=0.05, l_free=-0.4, l_hit=0.9):
    """Log-odds grid over arena (x0, x1, y0, y1): every ray lowers the cells it crosses (up to FREE_MARGIN_M / 2
    before its end) and raises the cell of its end. Returns (grid (ny, nx), extent)."""
    x0, x1, y0, y1 = arena
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    g = np.zeros((ny, nx))

    def cell(q):
        i, j = int((q[1] - y0) / res), int((q[0] - x0) / res)
        return (i, j) if 0 <= i < ny and 0 <= j < nx else None

    for r in rays:
        seen = set()
        for s in np.arange(0.0, max(0.0, r.d - FREE_MARGIN_M / 2), res / 2):
            c = cell(r.o + s * r.u)
            if c and c not in seen:
                seen.add(c)
                g[c] += l_free
        c = cell(r.end)
        if c:
            g[c] += l_hit
    return g, (x0, x1, y0, y1)
