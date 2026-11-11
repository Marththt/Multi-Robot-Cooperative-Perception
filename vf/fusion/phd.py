"""Multi-sensor PHD and CPHD filters (point 4 of Section 3.2 of the
professor's document) for ONE static shot with several robots.

Idea (Random Finite Sets): instead of deciding which reports go
together, estimate an "object density" v(x) over the arena, the PHD
intensity. Its integral over a zone is the expected number of objects in
that zone, its peaks are the positions. No identities, no explicit
associations: it is the most "continuous" of the four methods.

  PHD:  v(x) <- v(x) * [ (1 - p_D(x)) + sum_z p_D(x) g(z|x) / (kappa + <p_D g_z, v>) ]
  CPHD: as above, but it also propagates the distribution of the NUMBER
        of objects rho(n) (not just the mean): the count is much more
        stable, at the price of the elementary symmetric functions
        (Vo, Vo, Cantoni 2007, "Analytic implementations of the CPHD
        filter").

Multi-sensor: one robot at a time ("iterated corrector"), the intensity
after robot 1 is the prior of robot 2, and so on. It is the standard
approximation; the result depends slightly on the order of the robots.
Limitations known in the literature, visible here in simulation too:
  - PHD: an object not seen by ONE robot that should have seen it loses
    most of its mass (sensitivity to missed detections): it tends to
    underestimate the number.
  - CPHD: with fields of view that overlap only partially, a report in
    one zone shifts mass even into distant zones ("spooky effect",
    Fraenken-Schmitt-Ulmke 2009): even without faults it can get the
    count wrong by one in some shots.

Surveilled region: only the cells that at least one robot can see
(SensorModel.coverage). Without known poses, the whole arena.

Implemented on a GRID (cells of `step` metres) instead of Gaussian
mixtures: with a p_D(x) that depends on position (field of view,
distance) the mixture version needs approximations, while on the grid
p_D(x) is applied exactly cell by cell. A 5x4 m arena at 5 cm is 8000
cells: negligible cost.

Same sensor model as MHT (sensor_model.py): uniform prior with
n_prior expected objects, uniform clutter, p_D(x), R.
"""

import itertools
import math

import numpy as np

from vf.fusion.sensor_model import gauss2


class Grid:
    """Cells of the arena that at least one robot can see (see
    SensorModel.set_coverage)."""

    def __init__(self, sensor, robot_poses, step=0.05):
        x0, x1, y0, y1 = sensor.arena
        xs = np.arange(x0 + step / 2, x1, step)
        ys = np.arange(y0 + step / 2, y1, step)
        gx, gy = np.meshgrid(xs, ys)
        X = np.column_stack([gx.ravel(), gy.ravel()])
        self.X = X[sensor.coverage(robot_poses, X)]
        self.dA = step * step
        self.area = len(self.X) * self.dA


def extract_peaks(v, grid, n, radius, min_mass=0.0):
    """The n "heaviest" positions of v: take the maximum cell, compute
    the weighted mean of v within `radius`, remove that mass, repeat.
    Stops earlier if a peak has mass < min_mass."""
    v = v.copy()
    objects, masses = [], []
    for _ in range(n):
        if len(v) == 0:
            break
        i = int(np.argmax(v))
        if v[i] <= 0:
            break
        near = np.hypot(*(grid.X - grid.X[i]).T) <= radius
        mass = float(v[near].sum() * grid.dA)
        if mass < min_mass:
            break
        center = (v[near, None] * grid.X[near]).sum(0) / v[near].sum()
        objects.append((float(center[0]), float(center[1])))
        masses.append(mass)
        v[near] = 0.0
    return objects, masses


def _likelihoods(sensor, pose, Z, grid):
    R = sensor.cov(pose)
    return [gauss2(np.asarray(z, float), grid.X, R) for z in Z]


def _phd_intensity(order, reports, poses, sensor, grid):
    """PHD intensity on the grid after the robots of `order`, one at a time (iterated corrector)."""
    v = np.full(len(grid.X), sensor.n_prior / grid.area)
    kappa = sensor.clutter_density()
    for rid in order:
        pose = poses.get(rid)
        Z = reports.get(rid, [])
        pd = sensor.p_d(pose, grid.X)
        updated = v * (1.0 - pd)
        for g in _likelihoods(sensor, pose, Z, grid):
            w = pd * g * v
            updated += w / (kappa + w.sum() * grid.dA)
        v = updated
    return v


def _phd_result(v, grid, sensor):
    n_expected = float(v.sum() * grid.dA)
    # one object = one peak with mass >= 0.5 (the standard GM-PHD rule:
    # components with weight > 0.5), not round(n_expected): the diffuse
    # mass left where the robots could have been wrong is not an object
    objects, masses = extract_peaks(v, grid, 50, radius=3 * sensor.sigma_pos,
                                    min_mass=0.5)
    return dict(objects=objects, n=len(objects),
                extra=dict(n_expected=n_expected, masses=[round(m, 2) for m in masses]),
                intensity=(grid, v))


def phd(order, reports, poses, sensor, step=0.05):
    grid = Grid(sensor, [poses.get(r) for r in order], step)
    return _phd_result(_phd_intensity(order, reports, poses, sensor, grid), grid, sensor)


def robot_orders(order, n_max=24, seed=0):
    """Orders in which the iterated corrector takes the robots: all of them up to 4 robots (24), otherwise
    n_max orders drawn at random (fixed seed, so that a scene always gives the same result)."""
    order = list(order)
    if math.factorial(len(order)) <= n_max:
        return [list(p) for p in itertools.permutations(order)]
    rng = np.random.default_rng(seed)
    return [list(rng.permutation(order)) for _ in range(n_max)]


def phd_avg(order, reports, poses, sensor, step=0.05):
    """PHD with the intensity averaged over the robot orders (robot_orders): the iterated corrector
    weighs a missed detection more when it comes after the detections of the same box (a box missed by
    the LAST robot loses most of its mass, one missed by the first does not); the average removes that
    dependence on an arbitrary order."""
    grid = Grid(sensor, [poses.get(r) for r in order], step)
    orders = robot_orders(order)
    v = sum(_phd_intensity(o, reports, poses, sensor, grid) for o in orders) / len(orders)
    return _phd_result(v, grid, sensor)


# ====================================================================
# CPHD
# ====================================================================

def _elementary_symmetric(values):
    """Elementary symmetric functions e_0..e_m: coefficients of
    prod_i (1 + values_i * t)."""
    e = np.zeros(len(values) + 1)
    e[0] = 1.0
    for x in values:
        e[1:] = e[1:] + x * e[:-1]
    return e


def _poisson(lam, n_max):
    return np.array([math.exp(-lam) * lam ** k / math.factorial(k)
                     for k in range(n_max + 1)])


def _permutations(n, k):
    """P(n, k) = n! / (n - k)!, 0 if k > n."""
    if k > n:
        return 0.0
    return float(math.factorial(n) // math.factorial(n - k))


def _upsilon(u, e, m, a, pK, n_max):
    """Upsilon^u(n) for n = 0..n_max, with e = symmetric functions of a
    set of m reports (Vo-Vo-Cantoni formula written with the normalized
    spatial density: a = <1 - p_D, s>)."""
    out = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        tot = 0.0
        for j in range(0, min(m, n - u) + 1):
            if n - j - u < 0:
                continue
            tot += (math.factorial(m - j) * pK[m - j] * _permutations(n, j + u)
                    * a ** (n - j - u) * e[j])
        out[n] = tot
    return out


def _cphd_state(order, reports, poses, sensor, grid, n_max):
    """Cardinality distribution rho(n) and normalised spatial density s on the grid after the robots of
    `order`, one at a time (iterated corrector)."""
    area = grid.area
    s = np.full(len(grid.X), 1.0 / area)          # spatial density, integrates to 1
    rho = _poisson(sensor.n_prior, n_max)
    rho /= rho.sum()
    for rid in order:
        pose = poses.get(rid)
        Z = reports.get(rid, [])
        m = len(Z)
        pd = sensor.p_d(pose, grid.X)
        pK = _poisson(sensor.clutter, max(m, 1))
        a = float(((1.0 - pd) * s).sum() * grid.dA)
        psi = [pd * g * area for g in _likelihoods(sensor, pose, Z, grid)]  # p_D g / c(z)
        xi = [float((p * s).sum() * grid.dA) for p in psi]
        e = _elementary_symmetric(xi)
        ups0 = _upsilon(0, e, m, a, pK, n_max)
        ups1 = _upsilon(1, e, m, a, pK, n_max)
        den = float(ups0 @ rho)
        if den <= 0:
            continue
        updated = (float(ups1 @ rho) / den) * (1.0 - pd) * s
        for k in range(m):
            e_k = _elementary_symmetric(xi[:k] + xi[k + 1:])
            ups1_k = _upsilon(1, e_k, m - 1, a, pK, n_max)
            updated += (float(ups1_k @ rho) / den) * psi[k] * s
        rho = ups0 * rho
        rho /= rho.sum()
        mass = float(updated.sum() * grid.dA)
        s = updated / mass if mass > 0 else s
    return rho, s


def _cphd_result(rho, v, grid, sensor, n_max):
    n_map = int(np.argmax(rho))
    n_mean = float(np.arange(n_max + 1) @ rho)
    objects, masses = extract_peaks(v, grid, n_map, radius=3 * sensor.sigma_pos)
    return dict(objects=objects, n=n_map,
                extra=dict(p_n=round(float(rho[n_map]), 3), n_mean=round(n_mean, 2),
                           masses=[round(mm, 2) for mm in masses]),
                intensity=(grid, v))


def cphd(order, reports, poses, sensor, step=0.05, n_max=12):
    grid = Grid(sensor, [poses.get(r) for r in order], step)
    rho, s = _cphd_state(order, reports, poses, sensor, grid, n_max)
    return _cphd_result(rho, s * float(np.arange(n_max + 1) @ rho), grid, sensor, n_max)


def cphd_avg(order, reports, poses, sensor, step=0.05, n_max=12):
    """CPHD with the cardinality distribution and the intensity averaged over the robot orders
    (robot_orders), for the same reason as phd_avg."""
    grid = Grid(sensor, [poses.get(r) for r in order], step)
    orders = robot_orders(order)
    rho_sum, v_sum = np.zeros(n_max + 1), np.zeros(len(grid.X))
    for o in orders:
        rho, s = _cphd_state(o, reports, poses, sensor, grid, n_max)
        rho_sum += rho
        v_sum += s * float(np.arange(n_max + 1) @ rho)
    return _cphd_result(rho_sum / len(orders), v_sum / len(orders), grid, sensor, n_max)
