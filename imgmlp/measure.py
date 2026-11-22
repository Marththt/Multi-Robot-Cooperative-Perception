"""Scenes for the image + measurement model: for every robot its pose, the (rho, theta) of the objects it measured
and the index of its image in a frame cache (imgmlp/frames.py).

The model does not care which sensor gave rho and theta. Three sources:

  ideal   from the scene geometry: every box inside the field of view and range of a robot, whose centre is not
          hidden by another box, an obstacle or another robot, measured exactly (distance and bearing of the
          box centre)
  noisy   the ideal measurements with a simulated error: a range error that grows with the distance, a bearing
          error, a range scale error and a bearing offset per robot, missed boxes and false measurements
          (NoiseModel)
  noisyrand  as noisy, but every scene draws its own error from broad ranges (NoiseRanges, domain
          randomization): no sensor model is needed
  sensor  the detections stored in the set (P<k>.jsonl: the fitted ArUco sensor model in simulation, the ArUco
          detector on the real runs), read by gnn/dataset.py with the range / bearing calibration of the sensor
          (--sensor-calib), i.e. what a calibrated sensor delivers

A scene is a dict as in gnn/dataset.py (robots, poses, det_robot, distance, bearing, label, truth, run, layout,
round), without any hand-made feature, plus frame (index of the image of every robot in the cache, -1 if none).
"""

import math
from dataclasses import dataclass, fields

import numpy as np

from sim.geometry import blocked, box_polygon, rectangle, wrap

HFOV_DEG = 88.0
RANGE_M = (0.15, 3.0)
ROBOT_SIZE_M = (0.32, 0.24)          # body footprint (imgsim.render.Appearance.robot_size_m)
ROBOT_BODY_BEHIND_M = 0.15           # body centre behind the camera


def robot_polygon(pose):
    x, y, yaw = pose
    c = (x - ROBOT_BODY_BEHIND_M * math.cos(yaw), y - ROBOT_BODY_BEHIND_M * math.sin(yaw))
    return rectangle(c, ROBOT_SIZE_M[0], ROBOT_SIZE_M[1], yaw)


def box_visibility(k, poses, boxes, thetas, obstacles=(), fov_deg=HFOV_DEG, range_m=RANGE_M):
    """Boxes in the field of view and range of robot k: list of (box, distance, bearing, hidden). hidden: the
    line from the camera to the box centre crosses another box, an obstacle or another robot.
    Bearing positive to the right: world direction = yaw - bearing (as the real detector)."""
    x, y, yaw = poses[k]
    cam = np.array([x, y])
    if thetas is None:
        thetas = np.zeros(len(boxes))
    polys = [box_polygon(c, t) for c, t in zip(boxes, thetas)]
    robots = [robot_polygon(p) for q, p in enumerate(poses) if q != k]
    half = math.radians(fov_deg) / 2
    out = []
    for j, c in enumerate(boxes):
        d = float(math.hypot(c[0] - x, c[1] - y))
        b = float(wrap(yaw - math.atan2(c[1] - y, c[0] - x)))
        if d < range_m[0] or d > range_m[1] or abs(b) > half:
            continue
        others = [p for q, p in enumerate(polys) if q != j] + list(obstacles) + robots
        out.append((j, d, b, bool(blocked(cam, np.asarray(c, float), others))))
    return out


@dataclass
class NoiseModel:
    range_sd_m: float = 0.02             # range error: range_sd_m + range_sd_rel * distance
    range_sd_rel: float = 0.03
    bearing_sd_deg: float = 1.5
    scale_bias_sd: float = 0.02          # per robot: range scale error (relative)
    bearing_bias_sd_deg: float = 1.0     # per robot: bearing offset
    miss_prob: float = 0.10              # a visible box is not measured
    false_rate: float = 0.15             # false measurements per robot (Poisson), uniform in range and field of view
    range_bias_sd_m: float = 0.0         # per robot: range offset

    @classmethod
    def ideal(cls):
        return cls(0, 0, 0, 0, 0, 0, 0, 0)

    def sd_range(self, d):
        """Total range spread at distance d [m] (noise, scale and offset of the robot)."""
        d = np.asarray(d, float)
        return np.sqrt((self.range_sd_m + self.range_sd_rel * d) ** 2 + (self.scale_bias_sd * d) ** 2
                       + self.range_bias_sd_m ** 2)

    def sd_bearing_deg(self):
        return math.hypot(self.bearing_sd_deg, self.bearing_bias_sd_deg)


@dataclass
class NoiseRanges:
    """Domain randomization of the measurement error: every simulated scene draws its own NoiseModel, each field
    uniform in [lo, hi]. The ranges are broad on purpose and are not fitted to any real sensor, so that the model
    needs no sensor model: only rho and theta."""
    range_sd_m: tuple = (0.0, 0.08)
    range_sd_rel: tuple = (0.0, 0.06)
    bearing_sd_deg: tuple = (0.0, 4.0)
    scale_bias_sd: tuple = (0.0, 0.08)
    bearing_bias_sd_deg: tuple = (0.0, 3.0)
    miss_prob: tuple = (0.0, 0.7)
    false_rate: tuple = (0.0, 0.4)
    range_bias_sd_m: tuple = (0.0, 0.10)

    def sample(self, rng):
        return NoiseModel(**{f.name: float(rng.uniform(*getattr(self, f.name))) for f in fields(self)})


def measured_scene(g, noise, rng, layout="sim"):
    """Scene of one geometry.jsonl row (sim/plan.py, sim/scaling.py) with ideal or noisy measurements. noise is a
    NoiseModel, or a NoiseRanges (one NoiseModel drawn for this scene). The NoiseModel used is kept in
    scene["declared"] (the uncertainty the sensor declares, imgmlp/model.py unc=True)."""
    if isinstance(noise, NoiseRanges):
        noise = noise.sample(rng)
    poses = [tuple(map(float, p)) for p in g["poses"]]
    boxes = np.array(g["boxes"], float).reshape(-1, 2)
    thetas = np.array(g.get("yaws") or [0.0] * len(boxes), float)
    obstacles = [np.array(o, float) for o in g.get("obstacles", [])]
    robots = [f"P{k}" for k in range(len(poses))]
    det_robot, dist, bear, label = [], [], [], []
    half = math.radians(HFOV_DEG) / 2
    for k in range(len(poses)):
        scale = 1 + rng.normal(0, noise.scale_bias_sd) if noise.scale_bias_sd else 1.0
        shift = rng.normal(0, noise.range_bias_sd_m) if noise.range_bias_sd_m else 0.0
        offset = math.radians(rng.normal(0, noise.bearing_bias_sd_deg)) if noise.bearing_bias_sd_deg else 0.0
        for j, d, b, hidden in box_visibility(k, poses, boxes, thetas, obstacles):
            if hidden or (noise.miss_prob and rng.random() < noise.miss_prob):
                continue
            d = d * scale + shift + (rng.normal(0, noise.range_sd_m + noise.range_sd_rel * d)
                             if noise.range_sd_m or noise.range_sd_rel else 0.0)
            b = b + offset + (math.radians(rng.normal(0, noise.bearing_sd_deg)) if noise.bearing_sd_deg else 0.0)
            det_robot.append(k), dist.append(max(d, 0.05)), bear.append(b), label.append(j)
        for _ in range(int(rng.poisson(noise.false_rate)) if noise.false_rate else 0):
            det_robot.append(k), dist.append(rng.uniform(*RANGE_M)), bear.append(rng.uniform(-half, half))
            label.append(-1)
    return dict(run=g["run"], layout=layout, round=0, robots=robots, poses=np.array(poses, float),
                det_robot=np.array(det_robot, int), distance=np.array(dist, float), bearing=np.array(bear, float),
                label=np.array(label, int), truth=boxes, declared=noise)


def sensor_scenes(set_dir, gt_path, calib_path):
    """Scenes with the detections stored in the set (gnn/dataset.py), keeping only what the model reads."""
    from gnn.dataset import build_scenes
    keep = ("run", "layout", "round", "robots", "poses", "det_robot", "distance", "bearing", "label", "truth")
    return [{k: s[k] for k in keep} for s in build_scenes(set_dir, gt_path, calib_path)]


def attach_frames(scenes, index):
    """frame[k] = row of the image of robot k in the cache (index: {(run, round, robot): row}), -1 if none."""
    missing = 0
    for s in scenes:
        s["frame"] = np.array([index.get((s["run"], int(s["round"]), q), -1) for q in s["robots"]], int)
        missing += int((s["frame"] < 0).sum())
    return missing


def add_silent_robots(scenes, rows):
    """Robots that have a frame but measured nothing are absent from the sensor scenes (gnn/dataset.py builds the
    robots from the reports). Graph B gives every robot a node, so they are added with their pose from the frame
    cache (frames.csv x, y, yaw) and no report. Returns how many were added."""
    by_stop = {}
    for r in rows:
        if r.get("x") not in (None, ""):
            by_stop.setdefault((r["run"], int(r["round"])), []).append(r)
    added = 0
    for s in scenes:
        have = set(s["robots"])
        extra = sorted((r for r in by_stop.get((s["run"], int(s["round"])), []) if r["robot"] not in have),
                       key=lambda r: r["robot"])
        if extra:
            s["robots"] = list(s["robots"]) + [r["robot"] for r in extra]
            s["poses"] = np.vstack([np.asarray(s["poses"], float).reshape(-1, 3),
                                    [[float(r["x"]), float(r["y"]), float(r["yaw"])] for r in extra]])
            added += len(extra)
    return added


def detection_positions(s):
    """World position of every measurement from the robot pose, rho and theta (the fixed geometric layer)."""
    p = s["poses"][s["det_robot"]]
    ray = p[:, 2] - s["bearing"]
    return np.column_stack([p[:, 0] + s["distance"] * np.cos(ray), p[:, 1] + s["distance"] * np.sin(ray)])
