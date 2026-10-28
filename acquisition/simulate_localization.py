"""SIMULATED comparison of all the localization methods (the same ones
used by localize_trials.py and the live map) in IDEAL conditions:
N robots (--robots, any number), random objects, and every robot reports
every object inside its field of view, at its exact position, with no
false alarms.

Robot poses: with --robots 3 (default) the three poses of the lab
(LAB_POSES below); with any other N the robots are spread evenly on an
ellipse 0.3 m inside the arena border, each looking at the arena center.

In these conditions the only difficulty left is the correspondence
problem itself: several objects of the same class, seen by several
robots with overlapping fields of view, with no identity attached to the
reports. A method that does not localize every object exactly here has a
problem in its logic, not in the data.

Objects are placed uniformly in the region that at least one robot can
see (an object nobody can see cannot be found by any method), at least
40 cm apart.

    python -m acquisition.simulate_localization
    python -m acquisition.simulate_localization --trials 200 --max-objects 8
    python -m acquisition.simulate_localization --robots 7
"""
import argparse
import json
import math
import os
import time

import numpy as np

from vf.fusion.localization_methods import METHODS, measurements_by_class, ospa, run_method
from vf.fusion.sensor_model import ZONE_MIN_SIDE_M, SensorModel, grid_zones

ARENA = (-2.690, 2.135, -2.672, 1.519)
LAB_POSES = {"P0": (-2.4, -0.6, 0.0), "P1": (1.9, -0.4, math.pi), "P2": (-0.3, 1.3, -math.pi / 2)}
POSES = dict(LAB_POSES)  # set in main() from --robots
MIN_SEPARATION_M = 0.4


def ring_poses(n_robots, margin_m=0.3):
    """n_robots camera poses evenly spread on an ellipse margin_m inside the
    arena border, each looking at the arena center. Ids P0, P1, ..."""
    x0, x1, y0, y1 = ARENA
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = (x1 - x0) / 2 - margin_m, (y1 - y0) / 2 - margin_m
    poses = {}
    for k in range(n_robots):
        th = 2 * math.pi * k / n_robots
        x, y = cx + ax * math.cos(th), cy + ay * math.sin(th)
        poses[f"P{k}"] = (x, y, math.atan2(cy - y, cx - x))
    return poses


def covered(sensor, p):
    return any(sensor.p_d_one(pose, p) > 0 for pose in POSES.values())


def random_covered_point(rng, sensor):
    while True:
        p = np.array([rng.uniform(ARENA[0], ARENA[1]), rng.uniform(ARENA[2], ARENA[3])])
        if covered(sensor, p):
            return p


def scene(rng, n_obj, sensor):
    objs = []
    while len(objs) < n_obj:
        p = random_covered_point(rng, sensor)
        if all(np.hypot(*(p - q)) > MIN_SEPARATION_M for q in objs):
            objs.append(p)
    return objs


def observe(objs, sensor, poses, class_name="box"):
    """Ideal observations, in the same form the robots produce them:
    every robot sees every object in its field of view and returns its
    exact distance and bearing (positive to the right); nothing else."""
    robots = {}
    for rid, (x, y, psi) in poses.items():
        observations = []
        for o in objs:
            if sensor.p_d_one((x, y, psi), o) <= 0:
                continue
            dx, dy = o[0] - x, o[1] - y
            bearing = (psi - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi
            observations.append(dict(class_name=class_name, distance=math.hypot(dx, dy),
                                     bearing=bearing))
        robots[rid] = dict(pose=(x, y, psi), observations=observations)
    return robots


def campaign(sensor, zones, n_trials, max_objects, seed=0, methods=None):
    methods = methods or list(METHODS)
    rng = np.random.default_rng(seed)
    res = {m: dict(ospa=[], cnt=[], t=0.0) for m in methods}
    for _ in range(n_trials):
        objs = scene(rng, int(rng.integers(1, max_objects + 1)), sensor)
        used, poses, per_class, _ = measurements_by_class(observe(objs, sensor, POSES))
        reports = per_class.get("box", {})
        for m in methods:
            t0 = time.time()
            out = run_method(m, used, reports, poses, sensor, zones=zones)
            res[m]["t"] += time.time() - t0
            res[m]["ospa"].append(ospa(out["objects"], objs))
            res[m]["cnt"].append(out["n"] == len(objs))
    return res


def print_table(res, n_trials):
    print(f"   {'method':<13} {'mean OSPA [m]':>14} {'exact count':>12} {'ms/shot':>9}")
    for m, r in res.items():
        print(f"   {m:<13} {np.mean(r['ospa']):>14.3f} {100 * np.mean(r['cnt']):>11.0f}% "
              f"{1000 * r['t'] / n_trials:>9.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--max-objects", type=int, default=5,
                    help="each trial has between 1 and this many objects")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zone-side", type=float, default=ZONE_MIN_SIDE_M,
                    help="minimum side [m] of the grid cells used as zones (default %(default)s)")
    ap.add_argument("--robots", type=int, default=3,
                    help="number of robots N (3 = the lab poses; any other N: robots on a "
                         "ring looking at the arena center)")
    a = ap.parse_args()
    if a.robots < 1:
        ap.error("--robots must be at least 1")
    POSES.clear()
    POSES.update(LAB_POSES if a.robots == 3 else ring_poses(a.robots))

    methods = list(METHODS)
    zones = grid_zones(ARENA, a.zone_side)

    # the methods know the conditions are ideal: every object in view is
    # seen, no false alarms, reports very close to the true position
    sensor = SensorModel(p_d=1.0, sigma_pos=0.03, clutter=0.0,
                         n_prior=(1 + a.max_objects) / 2, arena=ARENA, r_max=6)
    print(f"ideal conditions, {len(POSES)} robots, {a.trials} trials, "
          f"1-{a.max_objects} objects per trial")
    print_table(campaign(sensor, zones, a.trials, a.max_objects, a.seed, methods), a.trials)


if __name__ == "__main__":
    main()