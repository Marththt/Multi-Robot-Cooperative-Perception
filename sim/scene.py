"""Scenes: true boxes (centre + orientation), obstacles, and the camera poses of every capture.

A scene can be
  recreated from a real run   boxes from OptiTrack, box orientations estimated from the viewing angles of
                              the real detections, camera poses taken from the logs of every round
  generated at random         from a ScenarioSpec (number of boxes, arrangement, spacing, robots,
                              obstacles), with its own seed
"""

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .geometry import APOTHEM_M, MIN_CENTER_DISTANCE_M, box_polygon, polygons_overlap, rectangle, wrap

ROTATION_STEP_DEG = 15.0          # ring rotation between rounds on the real setup


@dataclass
class Capture:
    round: int
    robot: str
    pose: tuple                    # camera pose (x, y, yaw)
    radius: float = 1.3            # survey metadata copied into the records
    angle_deg: float = 0.0


@dataclass
class Scene:
    layout_id: str
    run_id: str
    boxes: np.ndarray              # (M, 2) box centres [m]
    thetas: np.ndarray             # (M,) orientation of face 0 [rad]
    obstacles: List[np.ndarray] = field(default_factory=list)    # footprints (K_i, 2)
    captures: List[Capture] = field(default_factory=list)

    def box_polygons(self):
        return [box_polygon(c, t) for c, t in zip(self.boxes, self.thetas)]


# ====================================================================
# Recreating real runs
# ====================================================================

def estimate_orientations(n_boxes, observations, step_deg=0.25):
    """Box orientations (mod 120 deg) that best explain the reported viewing angles.

    observations: (box index, camera yaw, reported viewing angle) of the detections of one run.
    The reported angle is compared with the closest face, so the orientation is found up to the
    symmetry of the triangle. Boxes without detections get a random orientation (None)."""
    grid = np.radians(np.arange(0.0, 120.0, step_deg))
    faces = np.radians([0.0, 120.0, 240.0])
    out = []
    for j in range(n_boxes):
        obs = np.array([(yaw, va) for b, yaw, va in observations if b == j])
        if not len(obs):
            out.append(None)
            continue
        best, best_err = 0.0, None
        for t0 in grid:
            a = np.min(np.abs(wrap(obs[:, 0][:, None] + math.pi - (t0 + faces)[None])), axis=1)
            err = float(np.median(np.abs(a - obs[:, 1])))
            if best_err is None or err < best_err:
                best, best_err = t0, err
        out.append(best)
    return out


def scene_from_real(run_scenes, rng):
    """One Scene from the rounds of one real run (gnn.dataset.build_scenes output, one dict per round)."""
    first = run_scenes[0]
    truth = first["truth"]
    obs = []
    for s in run_scenes:
        for i, lab in enumerate(s["label"]):
            if lab >= 0:
                obs.append((int(lab), float(s["poses"][s["det_robot"][i]][2]), float(s["view"][i])))
    thetas = estimate_orientations(len(truth), obs)
    thetas = np.array([t if t is not None else rng.uniform(0, 2 * math.pi) for t in thetas])
    captures = []
    for s in sorted(run_scenes, key=lambda q: q["round"]):
        for r, robot in enumerate(s["robots"]):
            captures.append(Capture(round=int(s["round"]), robot=robot, pose=tuple(map(float, s["poses"][r]))))
    return Scene(layout_id=first["layout"], run_id=first["run"], boxes=np.array(truth, float), thetas=thetas,
                 captures=captures)


# ====================================================================
# Random scenes
# ====================================================================

@dataclass
class ScenarioSpec:
    """Ranges of a random scene family. Lists of choices are drawn uniformly; [lo, hi] ranges uniformly."""
    n_boxes: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 6, 8, 9, 10])
    arrangements: List[str] = field(default_factory=lambda: ["uniform"])   # uniform | pairs | line | cluster
    spacing_m: List[float] = field(default_factory=lambda: [0.20, 0.60])     # min spacing (uniform) or pitch
    area_half_m: float = 0.8                 # boxes inside a square of this half-size around the centre
    n_robots: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])
    robot_radius_m: List[float] = field(default_factory=lambda: [1.0, 2.0])
    placements: List[str] = field(default_factory=lambda: ["ring", "ring_gap"])  # ring | ring_gap | one_side | random
    yaw_jitter_deg: float = 5.0
    rounds: int = 1                          # captures per robot, the ring rotating ROTATION_STEP_DEG each round
    n_obstacles: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    targeted_obstacles: bool = False         # obstacles placed to hide boxes from some robots
    hide_per_box: List[int] = field(default_factory=lambda: [1, 3])          # robots a targeted obstacle hides from
    noise_jitter: float = 0.0                # per run, noise sds scaled by a factor in [1 - j, 1 + j]
    false_factor: List[float] = field(default_factory=lambda: [1.0, 1.0])    # per run, false-detection rate x [lo, hi]
    two_face_factor: List[float] = field(default_factory=lambda: [1.0, 1.0]) # per run, two-face rate x [lo, hi]


def _spaced(points, p, spacing):
    return all(math.hypot(p[0] - q[0], p[1] - q[1]) >= spacing for q in points)


def random_boxes(spec, n, rng, tries=2000, stretch=0.0, kind=None):
    """Box centres of one arrangement. stretch (m) widens the spacing and the area, used by random_scene
    when the boxes could not be placed or oriented without touching."""
    kind = kind or str(rng.choice(spec.arrangements))
    base = max(rng.uniform(*spec.spacing_m), MIN_CENTER_DISTANCE_M)
    s = base + stretch
    half = spec.area_half_m + stretch
    pts = []
    if kind == "line":
        ang = rng.uniform(0, math.pi)
        s = max(min(s, 2 * half / max(n - 1, 1)), spec.spacing_m[0] + stretch)
        for k in range(n):
            t = (k - (n - 1) / 2) * s
            pts.append((t * math.cos(ang), t * math.sin(ang)))
    elif kind == "pairs":
        n_pairs = (n + 1) // 2
        centres = []
        for k in range(tries):
            if len(centres) == n_pairs:
                break
            if k % 100 == 99:                # stuck: start again
                centres = []
            c = rng.uniform(-half, half, 2)
            if _spaced(centres, c, base + 0.35):     # the boxes of two pairs are at least 35 cm apart
                centres.append(c)
        for c in centres:
            ang = rng.uniform(0, math.pi)
            off = np.array([math.cos(ang), math.sin(ang)]) * s / 2
            pts.append(tuple(c + off))
            if len(pts) < n:
                pts.append(tuple(c - off))
    else:                                    # uniform or cluster
        if kind == "uniform":                # too many boxes for the area at this spacing: wider area
            s, lim = base, spec.area_half_m + 3 * stretch
        else:
            lim = min(half, max(0.25, s * math.sqrt(n) * 0.75) + stretch)
        centre = np.zeros(2) if kind == "uniform" else rng.uniform(-half + lim, half - lim, 2)
        for k in range(tries):
            if len(pts) == n:
                break
            if k % 200 == 199:               # stuck: start again
                pts = []
            p = centre + rng.uniform(-lim, lim, 2)
            if _spaced(pts, p, s):
                pts.append(tuple(p))
    return np.array(pts[:n], float), kind, s


def robot_poses(spec, n, centre, rng):
    r = rng.uniform(*spec.robot_radius_m)
    kind = rng.choice(spec.placements)
    start = rng.uniform(0, 2 * math.pi)
    if kind == "ring":
        angles = start + np.arange(n) * 2 * math.pi / n + rng.normal(0, math.radians(3), n)
    elif kind == "ring_gap":
        arc = rng.uniform(math.radians(180), math.radians(300))
        angles = start + np.linspace(0, arc, n)
    elif kind == "one_side":
        arc = rng.uniform(math.radians(60), math.radians(150))
        angles = start + np.linspace(0, arc, n)
    else:
        angles = rng.uniform(0, 2 * math.pi, n)
    radii = np.full(n, r) if kind != "random" else rng.uniform(*spec.robot_radius_m, n)
    poses = []
    for a, rr in zip(angles, radii):
        x, y = centre[0] + rr * math.cos(a), centre[1] + rr * math.sin(a)
        yaw = math.atan2(centre[1] - y, centre[0] - x) + math.radians(rng.normal(0, spec.yaw_jitter_deg))
        poses.append((x, y, yaw, rr, math.degrees(a)))
    return poses, kind


def random_obstacles(spec, boxes, robots, rng, tries=200):
    """Free-standing obstacles away from the boxes; with targeted_obstacles, one obstacle per hidden
    (box, robot) pair is placed on their line of sight."""
    obstacles = []
    n = int(rng.choice(spec.n_obstacles))
    for _ in range(tries):
        if len(obstacles) >= n:
            break
        c = rng.uniform(-spec.area_half_m, spec.area_half_m, 2)
        if all(math.hypot(*(c - b)) > 0.35 for b in boxes):
            obstacles.append(rectangle(c, rng.uniform(0.15, 0.5), rng.uniform(0.05, 0.15), rng.uniform(0, math.pi)))
    if spec.targeted_obstacles and len(robots):
        for b in boxes:
            k = int(rng.integers(spec.hide_per_box[0], spec.hide_per_box[1] + 1))
            for ri in rng.choice(len(robots), size=min(k, len(robots)), replace=False):
                rx, ry = robots[ri][:2]
                t = rng.uniform(0.35, 0.6)
                c = np.array([b[0] + t * (rx - b[0]), b[1] + t * (ry - b[1])])
                if all(math.hypot(*(c - bb)) > 0.25 for bb in boxes):
                    ang = math.atan2(ry - b[1], rx - b[0]) + math.pi / 2
                    obstacles.append(rectangle(c, rng.uniform(0.2, 0.35), 0.06, ang))
    return obstacles


def box_orientations(boxes, rng, gap=0.005, step_deg=3.0):
    """Random orientations with no two boxes overlapping (at least `gap` apart); None if not found.
    Boxes are oriented one after the other: each takes a random orientation among those (on a grid of
    step_deg over the 120 deg of the triangle's symmetry) that do not touch the boxes already placed.
    Only boxes closer than two vertex radii can touch, so only those pairs are tested."""
    n = len(boxes)
    thetas = rng.uniform(0, 2 * math.pi, n)
    if n < 2:
        return thetas
    d = np.linalg.norm(boxes[:, None, :] - boxes[None, :, :], axis=2)
    if (d[np.triu_indices(n, 1)] < MIN_CENTER_DISTANCE_M + gap).any():
        return None
    reach = 4 * APOTHEM_M + gap                   # two vertex radii
    grid = np.radians(np.arange(0.0, 120.0, step_deg))
    polys = [box_polygon(boxes[k], thetas[k]) for k in range(n)]
    for j in range(1, n):
        near = [k for k in range(j) if d[j, k] < reach]
        if not near:
            continue
        for t in rng.permutation(grid + rng.uniform(0, math.radians(step_deg))):
            p = box_polygon(boxes[j], t)
            if not any(polygons_overlap(p, polys[k], gap) for k in near):
                thetas[j] = t + 2 * math.pi / 3 * rng.integers(3)
                polys[j] = p
                break
        else:
            return None
    return thetas


def random_scene(spec, rng, layout_id, run_id, attempts=60):
    """A random scene of the family `spec`. The boxes are real prisms: two boxes never overlap. When the
    drawn arrangement cannot hold the boxes without overlapping (e.g. a line of three or more boxes needs
    a pitch of about 22 cm), it is drawn again, and after every 3 failures the spacing is widened by 1 cm."""
    n = int(rng.choice(spec.n_boxes))
    kind = str(rng.choice(spec.arrangements))          # kept through the attempts, so no family is favoured
    for attempt in range(attempts):
        boxes, kind, spacing = random_boxes(spec, n, rng, stretch=0.01 * (attempt // 3), kind=kind)
        thetas = None
        if len(boxes) == n:
            for _ in range(10):
                thetas = box_orientations(boxes, rng)
                if thetas is not None:
                    break
        if thetas is not None:
            break
    else:
        raise RuntimeError(f"no non-overlapping layout of {n} boxes for {run_id}")
    centre = boxes.mean(axis=0) if len(boxes) else np.zeros(2)
    n_rob = int(rng.choice(spec.n_robots))
    poses, placement = robot_poses(spec, n_rob, centre, rng)
    obstacles = random_obstacles(spec, boxes, poses, rng)
    captures = []
    for rnd in range(spec.rounds):
        rot = math.radians(ROTATION_STEP_DEG * rnd)
        for k, (x, y, yaw, rr, ang) in enumerate(poses):
            dx, dy = x - centre[0], y - centre[1]
            xr = centre[0] + dx * math.cos(rot) - dy * math.sin(rot)
            yr = centre[1] + dx * math.sin(rot) + dy * math.cos(rot)
            captures.append(Capture(round=rnd, robot=f"P{k}", pose=(xr, yr, yaw + rot), radius=rr,
                                    angle_deg=(ang + ROTATION_STEP_DEG * rnd) % 360))
    scene = Scene(layout_id=layout_id, run_id=run_id, boxes=boxes, thetas=thetas, obstacles=obstacles,
                  captures=captures)
    scene.meta = dict(arrangement=kind, spacing_m=round(float(spacing), 3), placement=placement,
                      n_robots=n_rob, n_obstacles=len(obstacles))
    return scene