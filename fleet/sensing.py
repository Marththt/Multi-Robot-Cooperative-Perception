"""Simulated sensing of one robot of the fleet.

Camera (YOLO on the camera, range refined by the IR as in acquisition/ir_yolo_scan.py), facing the chassis heading:
  - field of view hfov_deg, detections between r_min_m and r_max_m;
  - one ray per ray_step_deg from the lens: the ray goes on until it meets an obstacle, a target or another robot
    (true scene, fine grid). The cells it crosses are SEEN (free space: the coverage of the arena), the cell where
    it stops is seen OCCUPIED (the map of the obstacles), unless it is a robot (the brain knows where the robots
    are and does not map them);
  - a target is detected when its centre is in the field of view and in range, the line to it meets nothing else
    first, and with probability p_d (p_d_near up to 70% of the range, falling to p_d_far at r_max_m);
  - range error sd = sd_range_m[0] + sd_range_m[1] * d, bearing error sd_bearing_deg;
  - false detections: Poisson with fp_rate per frame, on a seen free cell.

Proximity: cells within prox_r_m of the robot are known (occupied or free) whatever the heading, as a bumper /
the IR of the gimbal would tell: a robot moving sideways does not drive into an obstacle it has not looked at.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from fleet.world import ROBOT_R_M


@dataclass
class CameraModel:
    hfov_deg: float = 88.0
    r_min_m: float = 0.3
    r_max_m: float = 2.0
    p_d_near: float = 0.9
    p_d_far: float = 0.6
    sd_range_m: tuple = (0.02, 0.03)
    sd_bearing_deg: float = 1.0
    fp_rate: float = 0.03
    ray_step_deg: float = 1.0
    forward_m: float = 0.10          # lens ahead of the robot centre
    prox_r_m: float = 0.45

    def p_d(self, d):
        knee = 0.7 * self.r_max_m
        if d <= knee:
            return self.p_d_near
        return self.p_d_near + (self.p_d_far - self.p_d_near) * (d - knee) / (self.r_max_m - knee)


@dataclass
class Frame:
    robot: str
    t: float
    lens: tuple                       # (x, y, yaw) of the camera
    detections: list = field(default_factory=list)   # dict(xy, range, bearing, true)  true = target index / None
    seen: np.ndarray = None           # flat indices (fine grid) of the cells seen free
    hits: np.ndarray = None           # flat indices (fine grid) of the cells seen occupied


def _march(lens_xy, angles, r_max, grid, ids, robots_xy):
    """For every ray: flat indices of the free cells it crosses (in order) and the cell where it stops (or -1)."""
    step = grid.res / 2
    s = np.arange(step, r_max + 1e-9, step)
    px = lens_xy[0] + np.cos(angles)[:, None] * s[None, :]
    py = lens_xy[1] + np.sin(angles)[:, None] * s[None, :]
    out = (px < grid.x0) | (px >= grid.x1) | (py < grid.y0) | (py >= grid.y1)
    ix = np.clip(((px - grid.x0) / grid.res).astype(int), 0, grid.nx - 1)
    iy = np.clip(((py - grid.y0) / grid.res).astype(int), 0, grid.ny - 1)
    wall = ids[ix, iy] != 0
    robot = np.zeros_like(wall)
    for rx, ry in robots_xy:
        robot |= np.hypot(px - rx, py - ry) < ROBOT_R_M
    block = wall | out | robot
    first = np.where(block.any(1), block.argmax(1), len(s))
    flat = ix * grid.ny + iy
    seen, hits = [], []
    for k, f in enumerate(first):
        seen.append(flat[k, :f])
        if f < len(s) and wall[k, f] and not out[k, f]:     # a ray stopped by a robot maps nothing there
            hits.append(flat[k, f])
    seen = np.unique(np.concatenate(seen)) if seen else np.zeros(0, int)
    return seen, np.unique(np.asarray(hits, int))


def look(cam, world, grid, ids, robot, pose, others_xy, rng, t=0.0, depth=True):
    """One camera frame of `robot` at pose (x, y, yaw of the camera): what it sees (Frame). depth=False: the
    camera gives only its detections (YOLO), not the free space (seen / hits empty)."""
    x, y, yaw = pose
    lens = (x + cam.forward_m * math.cos(yaw), y + cam.forward_m * math.sin(yaw), yaw)
    half = math.radians(cam.hfov_deg) / 2
    if depth:
        n = int(round(2 * half / math.radians(cam.ray_step_deg))) + 1
        angles = yaw + np.linspace(-half, half, n)
        seen, hits = _march(lens[:2], angles, cam.r_max_m, grid, ids, others_xy)
    else:
        seen = hits = np.zeros(0, int)
    fr = Frame(robot, t, lens, seen=seen, hits=hits)
    for j, c in enumerate(world.targets):
        v = c - np.asarray(lens[:2])
        d = float(np.hypot(*v))
        a = math.atan2(v[1], v[0])
        if not (cam.r_min_m <= d <= cam.r_max_m) or abs(_wrap(a - yaw)) > half:
            continue
        if not _line_of_sight(lens[:2], c, -(j + 1), grid, ids, others_xy):
            continue
        if rng.random() > cam.p_d(d):
            continue
        dm = d + rng.normal(0, cam.sd_range_m[0] + cam.sd_range_m[1] * d)
        am = a + rng.normal(0, math.radians(cam.sd_bearing_deg))
        fr.detections.append(dict(xy=(lens[0] + dm * math.cos(am), lens[1] + dm * math.sin(am)), range=dm,
                                  bearing=_wrap(am - yaw), true=j))
    free = seen[np.hypot(grid.P[seen, 0] - lens[0], grid.P[seen, 1] - lens[1]) >= cam.r_min_m] if len(seen) else seen
    for _ in range(rng.poisson(cam.fp_rate)):
        if len(free):
            q = grid.P[free[rng.integers(len(free))]] + rng.normal(0, grid.res / 3, 2)
        else:                                   # anywhere in the field of view
            b, r = yaw + rng.uniform(-half, half), rng.uniform(cam.r_min_m, cam.r_max_m)
            q = np.array([lens[0] + r * math.cos(b), lens[1] + r * math.sin(b)])
        d = float(np.hypot(*(q - np.asarray(lens[:2]))))
        fr.detections.append(dict(xy=(float(q[0]), float(q[1])), range=d,
                                  bearing=_wrap(math.atan2(q[1] - lens[1], q[0] - lens[0]) - yaw), true=None))
    return fr


@dataclass
class GimbalIR:
    """The ToF on the gimbal (acquisition/ir_ranging.py): one beam along the gimbal direction, 20 readings per
    second, range error 2.4 cm (real run, 3 robots), pitched just enough to stay on the box faces up to r_max_m
    (beyond: the floor or nothing, no reading). The gimbal fans from -fan_deg to +fan_deg around the chassis
    heading at fan_dps, or turns to an object at aim_dps, waits settle_s and takes n_aim readings (the median)."""
    rate_hz: float = 20.0
    r_max_m: float = 2.5
    sd_m: float = 0.024
    sd_dir_deg: float = 0.5           # where the beam really goes vs the gimbal angle read back
    aim_sd_deg: float = 1.5           # across-beam error of an aimed reading (aim + footprint)
    arm_m: float = 0.10               # IR ahead of the gimbal axis
    fan_deg: float = 40.0
    fan_dps: float = 30.0
    aim_dps: float = 120.0
    settle_s: float = 0.3
    n_aim: int = 5
    apothem_m: float = 0.068          # face -> box centre


def ir_beam(grid, ids, origin, heading, r_max, robots_xy, rng, ir):
    """One IR reading from origin along heading: (distance [m] or None for no echo within r_max, what it hit:
    'world' or 'robot' or None). The ray stops on the first occupied cell of the true scene or on a robot."""
    step = grid.res / 2
    h = heading + rng.normal(0, math.radians(ir.sd_dir_deg))
    s = np.arange(step, r_max + 1e-9, step)
    px, py = origin[0] + math.cos(h) * s, origin[1] + math.sin(h) * s
    out = (px < grid.x0) | (px >= grid.x1) | (py < grid.y0) | (py >= grid.y1)
    ix = np.clip(((px - grid.x0) / grid.res).astype(int), 0, grid.nx - 1)
    iy = np.clip(((py - grid.y0) / grid.res).astype(int), 0, grid.ny - 1)
    wall = (ids[ix, iy] != 0) | out              # the arena border is a wall too
    robot = np.zeros_like(wall)
    for rx, ry in robots_xy:
        robot |= np.hypot(px - rx, py - ry) < ROBOT_R_M
    hit = wall | robot
    if not hit.any():
        return None, None
    k = int(hit.argmax())
    return float(s[k] + rng.normal(0, ir.sd_m)), ("robot" if robot[k] and not wall[k] else "world")


def proximity(grid, ids, pose, r):
    """Flat indices of the cells within r of the robot, split into (free, occupied), from the true scene."""
    x, y = pose[0], pose[1]
    near = np.flatnonzero(np.hypot(grid.P[:, 0] - x, grid.P[:, 1] - y) <= r)
    occ = ids.ravel()[near] != 0
    return near[~occ], near[occ]


def _line_of_sight(a, b, own_id, grid, ids, others_xy):
    d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    s = np.arange(grid.res / 2, d, grid.res / 2)
    px = a[0] + (b[0] - a[0]) * s / d
    py = a[1] + (b[1] - a[1]) * s / d
    ix = np.clip(((px - grid.x0) / grid.res).astype(int), 0, grid.nx - 1)
    iy = np.clip(((py - grid.y0) / grid.res).astype(int), 0, grid.ny - 1)
    v = ids[ix, iy]
    if np.any((v != 0) & (v != own_id)):
        return False
    for rx, ry in others_xy:
        if np.any(np.hypot(px - rx, py - ry) < ROBOT_R_M):
            return False
    return True


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
