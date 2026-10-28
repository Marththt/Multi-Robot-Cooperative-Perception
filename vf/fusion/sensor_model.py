"""Sensor model (camera + ArUco of ONE robot) shared by all the
multi-object localization methods: MHT, PHD, CPHD
(data_association.py, phd.py) and by the comparison in localize_trials.py.

Same assumptions for every method, so the comparison measures the method
and not different parameters. The assumptions are those of Section 3.5 of
the professor's document:

  p_D(x)      probability that the robot detects an object at x.
              With the robot pose: 0 outside the field of view (horizontal
              angle hfov, distance r_min..r_max), p_d inside -- or the
              given p_D(distance) function.
              Without a pose: p_d everywhere (every robot sees the whole
              arena).
  R           covariance of the position error of a report:
              sigma_pos^2 * I (isotropic). To be estimated by comparing
              the ArUco positions with OptiTrack on the objects (Section
              3.5, point 3); until then it is a DECLARED value.
  clutter     false alarms: on average `clutter` false reports per robot
              per shot, uniform over the arena (density clutter/area).
  prior       on average `n_prior` objects per class, uniform over the
              arena (density n_prior/area). It is the "birth" of MHT
              and the initial intensity of PHD/CPHD.

All densities are per square metre, so they compare directly with the
Gaussian N(z; x, R) (also in 1/m^2).
"""

import math

import numpy as np

P_MIN = 1e-6  # p_D never exactly 0 or 1 inside the logarithms

# 3x3 points at -1.5, 0, +1.5 sigma with Gaussian weights (sum = 1), see
# SensorModel.p_d_measured
_K = (-1.5, 0.0, 1.5)
_W = [math.exp(-0.5 * k * k) for k in _K]
_SMOOTHING_POINTS = [(kx, ky, wx * wy / sum(_W) ** 2)
                     for kx, wx in zip(_K, _W) for ky, wy in zip(_K, _W)]


class SensorModel:
    """Defaults = ideal conditions: every object in the field of view is
    seen, no false alarms."""

    def __init__(self, p_d=1.0, sigma_pos=0.10, clutter=0.0, n_prior=2.0,
                 hfov_deg=88.0, r_min=0.15, r_max=3.0, arena=None,
                 fault_model=None, lighting="normal"):
        if arena is None:
            raise ValueError("arena = (x_min, x_max, y_min, y_max) required")
        self.p_d_base = float(p_d)
        self.sigma_pos = float(sigma_pos)
        self.clutter = float(clutter)
        self.n_prior = float(n_prior)
        self.hfov = math.radians(hfov_deg)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.arena = tuple(float(v) for v in arena)
        self.fault_model = fault_model
        self.lighting = lighting
        self._covered_area = None

    # ------------------------------------------------------------ geometry
    def area(self):
        x0, x1, y0, y1 = self.arena
        return (x1 - x0) * (y1 - y0)

    def clutter_density(self):
        return max(self.clutter, 1e-9) / self.observed_area()

    def prior_density(self):
        return self.n_prior / self.observed_area()

    # ------------------------------------------------------------ coverage
    def coverage(self, robot_poses, X):
        """True for the points X that at least one robot can see. If even
        a single pose is missing, the whole arena."""
        robot_poses = list(robot_poses)
        if not robot_poses or any(p is None for p in robot_poses):
            return np.ones(len(X), dtype=bool)
        m = np.zeros(len(X), dtype=bool)
        for p in robot_poses:
            m |= self.p_d(p, X) > 0
        return m

    def set_coverage(self, robot_poses, step=0.05):
        """The surveilled region is the one that at least one robot sees:
        prior and clutter are spread over THAT, not over the whole arena.
        An object that no robot can see cannot be estimated by any method;
        keeping it in the prior would only make PHD/CPHD "invent" objects
        in the blind zones or unbalance birth and clutter (MHT)."""
        x0, x1, y0, y1 = self.arena
        gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step),
                             np.arange(y0 + step / 2, y1, step))
        X = np.column_stack([gx.ravel(), gy.ravel()])
        m = self.coverage(robot_poses, X)
        self._covered_area = max(float(m.sum()) * step * step, 0.01)

    def observed_area(self):
        return self._covered_area if self._covered_area is not None else self.area()

    # ------------------------------------------------------------ p_D
    def p_d(self, pose, xy):
        """xy: array (N, 2) or (2,) -> array (N,) of p_D. pose=(x, y, yaw)
        of the CAMERA, or None."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        if pose is None:
            return np.full(len(xy), self.p_d_base)
        px, py, yaw = pose
        dx, dy = xy[:, 0] - px, xy[:, 1] - py
        d = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        inside = (np.abs(ang) <= self.hfov / 2) & (d >= self.r_min)
        if self.fault_model is not None:
            p = np.array([self.fault_model.p_d(float(di), 0.0, 0.0, self.lighting)
                          for di in d])
        else:
            p = np.where(d <= self.r_max, self.p_d_base, 0.0)
        return np.where(inside, p, 0.0)

    def p_d_one(self, pose, z):
        return float(self.p_d(pose, z)[0])

    def p_d_measured(self, pose, xy):
        """p_D averaged over the position error around xy (Gaussian with
        sigma_pos, 3x3 points at -1.5, 0, +1.5 sigma).

        Use it when xy is a MEASURED or ESTIMATED position, not a true
        one. A report of an object near the edge of the field of view (or
        near r_max) can land just outside it because of the position
        noise: with the plain p_d(xy) the robot "could not have seen" its
        own report, so the report could only be a false alarm. Averaging
        over the uncertainty keeps a small, non-zero p_D there. For TRUE
        positions (the PHD/CPHD grid cells) use p_d()."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        total = np.zeros(len(xy))
        for dx, dy, w in _SMOOTHING_POINTS:
            total += w * self.p_d(pose, xy + self.sigma_pos * np.array([dx, dy]))
        return total

    def p_d_measured_one(self, pose, z):
        return float(self.p_d_measured(pose, z)[0])

    # ------------------------------------------------------------ noise
    def cov(self, pose=None, z=None):
        return (self.sigma_pos ** 2) * np.eye(2)

    def describe(self):
        x0, x1, y0, y1 = self.arena
        pd = ("p_D from fault model (distance)" if self.fault_model is not None
              else f"p_D={self.p_d_base:g} up to {self.r_max:g} m")
        return (f"{pd}, field of view {math.degrees(self.hfov):.0f} degrees, "
                f"position sigma={self.sigma_pos:g} m, clutter={self.clutter:g}/robot, "
                f"prior={self.n_prior:g} objects/class, arena x[{x0:.2f},{x1:.2f}] "
                f"y[{y0:.2f},{y1:.2f}] ({self.area():.1f} m^2)")


def gauss2(z, m, S):
    """N(z; m, S) for z (2,) and m (N,2) or (2,) -> array (N,)."""
    m = np.atleast_2d(m)
    Si = np.linalg.inv(S)
    d = z[None, :] - m
    q = np.einsum("ni,ij,nj->n", d, Si, d)
    return np.exp(-0.5 * q) / (2 * math.pi * math.sqrt(np.linalg.det(S)))


def mahalanobis2(z, m, S):
    d = np.asarray(z, float) - np.asarray(m, float)
    return float(d @ np.linalg.solve(S, d))


def arena_from_regions(path):
    import json
    with open(path) as f:
        reg = json.load(f)["regions"]
    return (min(r["x_min"] for r in reg), max(r["x_max"] for r in reg),
            min(r["y_min"] for r in reg), max(r["y_max"] for r in reg))


# Zones used by the fusion (zone + class): a regular grid
# of cells over the arena, each cell at least ZONE_MIN_SIDE_M per side.
# A cell must be clearly larger than a target: with 0.20 m boxes a 0.5 m
# cell has more than 4 times the area of a box (> 2 box sides per side),
# so two boxes rarely share a cell and a box rarely spans many cells.
ZONE_MIN_SIDE_M = 0.5


def grid_zones(arena, min_side=ZONE_MIN_SIDE_M):
    """Regular grid of zones covering arena = (x_min, x_max, y_min, y_max):
    n_x = floor(width / min_side) columns and n_y = floor(height / min_side)
    rows (at least 1 each), so every cell is at least min_side x min_side
    and the cells tile the arena exactly. Returns the zones in the format
    of the regions file: [dict(name="r<row>c<col>", x_min, x_max, y_min,
    y_max), ...], row 0 = lowest y, col 0 = lowest x."""
    x0, x1, y0, y1 = arena
    nx = max(1, int((x1 - x0) // min_side))
    ny = max(1, int((y1 - y0) // min_side))
    wx, wy = (x1 - x0) / nx, (y1 - y0) / ny
    zones = []
    for r in range(ny):
        for c in range(nx):
            zones.append(dict(name=f"r{r}c{c}",
                              x_min=x0 + c * wx, x_max=x0 + (c + 1) * wx if c < nx - 1 else x1,
                              y_min=y0 + r * wy, y_max=y0 + (r + 1) * wy if r < ny - 1 else y1))
    return zones


def zones_from_arguments(args, arena):
    return grid_zones(arena, args.zone_side)


def arena_from_data(points, margin=1.0):
    p = np.asarray(points, float)
    return (p[:, 0].min() - margin, p[:, 0].max() + margin,
            p[:, 1].min() - margin, p[:, 1].max() + margin)


# ====================================================================
# Shared command-line arguments (localize_trials.py, dashboard.py,
# map_server.py): same names, same defaults
# ====================================================================

def add_sensor_arguments(parser):
    """Default = NO FAULTS (fault-free stage of the document): every
    object in the field of view is seen (p_D=1), no false alarms.
    Only the position error remains, which is there anyway."""
    g = parser.add_argument_group(
        "sensor model for localization (same for all methods)")
    g.add_argument("--p-d", type=float, default=1.0,
                   help="probability of seeing an object in the field of view "
                        "(default 1 = no faults)")
    g.add_argument("--sigma-pos", type=float, default=0.10,
                   help="position error of a report [m] (default 0.10)")
    g.add_argument("--clutter", type=float, default=0.0,
                   help="mean false alarms per robot per shot (default 0)")
    g.add_argument("--n-prior", type=float, default=2.0,
                   help="objects expected per class before looking (default 2)")
    g.add_argument("--hfov-deg", type=float, default=88.0,
                   help="horizontal field of view of the camera (2*atan(640/fx), "
                        "fx~631-665 at 1280 px: ~88-91 degrees)")
    g.add_argument("--r-max", type=float, default=3.0,
                   help="maximum distance at which a marker is seen [m]")
    g.add_argument("--zone-side", type=float, default=ZONE_MIN_SIDE_M,
                   help="minimum side [m] of the grid cells used as zones by zone + class "
                        "(default %(default)s: a cell at least 4 times the area of a "
                        "0.20 m box)")
    return g


def sensor_from_arguments(args, arena):
    return SensorModel(p_d=args.p_d, sigma_pos=args.sigma_pos, clutter=args.clutter,
                       n_prior=args.n_prior, hfov_deg=args.hfov_deg,
                       r_max=args.r_max, arena=arena)