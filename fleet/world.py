"""The arena of the fleet: obstacles (rectangles), targets (the boxes, 13.6 cm across), and the two grids.

Frame: the OptiTrack frame of the lab (x, y in metres). The default arena is the bounding box of
config/optitrack_regions.json (4.8 x 4.2 m).

  World.random(...)       obstacles and targets placed at random, with corridors the robots fit through
  World.load / save       the same as JSON: {"arena": [x0, x1, y0, y1], "obstacles": [[x0, x1, y0, y1], ...],
                          "targets": [[x, y], ...]}
  Grid                    a regular grid over the arena (cell centres, index <-> point)
  World.ids(grid)         per cell: 0 free, k > 0 obstacle k-1, -(j+1) target j (what a camera ray meets)
"""

import json
import math

import numpy as np

TARGET_R_M = 0.07            # a box seen from above: ~13.6 cm across (2 apothems of yolo_gnn/camera.py)
ROBOT_R_M = 0.22             # RoboMaster EP, 32 x 24 cm: circle around it
DEFAULT_REGIONS = "config/optitrack_regions.json"


class Grid:
    def __init__(self, arena, res):
        self.x0, self.x1, self.y0, self.y1 = arena
        self.res = res
        self.nx = int(math.ceil((self.x1 - self.x0) / res))
        self.ny = int(math.ceil((self.y1 - self.y0) / res))
        xs = self.x0 + (np.arange(self.nx) + 0.5) * res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * res
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")         # [ix, iy]
        self.P = np.stack([self.X.ravel(), self.Y.ravel()], 1)      # cell centres, row = ix * ny + iy

    @property
    def shape(self):
        return (self.nx, self.ny)

    def index(self, x, y):
        """(ix, iy) of the cell of (x, y), clipped to the grid."""
        ix = min(self.nx - 1, max(0, int((x - self.x0) / self.res)))
        iy = min(self.ny - 1, max(0, int((y - self.y0) / self.res)))
        return ix, iy

    def inside(self, x, y):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def centre(self, ix, iy):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res


def lab_arena(path=DEFAULT_REGIONS):
    from vf.fusion.sensor_model import arena_from_regions
    return tuple(float(v) for v in arena_from_regions(path))


class World:
    def __init__(self, arena, obstacles, targets):
        self.arena = tuple(float(v) for v in arena)
        self.obstacles = [tuple(float(v) for v in o) for o in obstacles]
        self.targets = np.asarray(targets, float).reshape(-1, 2)

    # ---------------------------------------------------------------- files
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["arena"], d["obstacles"], d["targets"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(arena=list(self.arena), obstacles=[list(o) for o in self.obstacles],
                           targets=self.targets.tolist()), f, indent=1)

    # ---------------------------------------------------------------- random scene
    @classmethod
    def random(cls, rng, arena, n_targets=10, n_obstacles=4, keep_free=(), keep_free_r=0.6, size=(0.25, 0.7),
               corridor_m=0.8, target_sep_m=0.6, target_clear_m=0.6, border_m=0.35, tries=20000):
        """Obstacles: axis-aligned rectangles with sides in `size`, at least corridor_m apart from each other and
        from the border (a robot passes between them). Targets: at least target_sep_m apart, target_clear_m from
        the obstacles, border_m from the border. Nothing within keep_free_r of the points keep_free (the starts)."""
        x0, x1, y0, y1 = arena
        obs = []
        for _ in range(tries):
            if len(obs) == n_obstacles:
                break
            w, h = rng.uniform(*size, 2)
            cx = rng.uniform(x0 + corridor_m + w / 2, x1 - corridor_m - w / 2)
            cy = rng.uniform(y0 + corridor_m + h / 2, y1 - corridor_m - h / 2)
            o = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
            if any(rect_gap(o, q) < corridor_m for q in obs):
                continue
            if any(rect_dist(o, p) < keep_free_r for p in keep_free):
                continue
            obs.append(o)
        tg = []
        for _ in range(tries):
            if len(tg) == n_targets:
                break
            p = np.array([rng.uniform(x0 + border_m, x1 - border_m), rng.uniform(y0 + border_m, y1 - border_m)])
            if any(np.hypot(*(p - q)) < target_sep_m for q in tg):
                continue
            if any(rect_dist(o, p) < target_clear_m for o in obs):
                continue
            if any(np.hypot(*(p - np.asarray(q))) < keep_free_r for q in keep_free):
                continue
            tg.append(p)
        return cls(arena, obs, tg)

    # ---------------------------------------------------------------- grids
    def ids(self, grid):
        """Per cell of grid: 0 free, k > 0 obstacle k-1, -(j+1) target j."""
        out = np.zeros(grid.shape, int)
        for k, (a, b, c, d) in enumerate(self.obstacles):
            out[(grid.X >= a) & (grid.X <= b) & (grid.Y >= c) & (grid.Y <= d)] = k + 1
        for j, (x, y) in enumerate(self.targets):
            m = np.hypot(grid.X - x, grid.Y - y) <= max(TARGET_R_M, grid.res * 0.75)
            out[m] = -(j + 1)
        return out

    def collides(self, x, y, r=ROBOT_R_M):
        """What a robot of radius r at (x, y) touches: 'border', 'obstacle', 'target', or None."""
        a = self.arena
        if x - r < a[0] or x + r > a[1] or y - r < a[2] or y + r > a[3]:
            return "border"
        if any(rect_dist(o, (x, y)) < r for o in self.obstacles):
            return "obstacle"
        if len(self.targets) and np.min(np.hypot(self.targets[:, 0] - x, self.targets[:, 1] - y)) < r + TARGET_R_M:
            return "target"
        return None


def rect_dist(o, p):
    """Distance from point p to rectangle o = (x0, x1, y0, y1) (0 inside)."""
    dx = max(o[0] - p[0], 0.0, p[0] - o[1])
    dy = max(o[2] - p[1], 0.0, p[1] - o[3])
    return math.hypot(dx, dy)


def rect_gap(o, q):
    dx = max(o[0] - q[1], q[0] - o[1], 0.0)
    dy = max(o[2] - q[3], q[2] - o[3], 0.0)
    return math.hypot(dx, dy)
