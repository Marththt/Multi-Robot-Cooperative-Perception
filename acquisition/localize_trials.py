"""Compare ALL the fusion methods of Section 3.2 of the professor's
document on the static trials written by static_trial.py: for every trial
and every class, how many objects and WHERE according to each method.

    zone_class, union, mht, phd, cphd
    (see vf/fusion/localization_methods.py)

Usage (from the project root)
-----------------------------
    python -m acquisition.localize_trials --trial-dir trials/ --regions config/optitrack_regions.json

    # with the ground truth (true object positions) -> OSPA and count
    python -m acquisition.localize_trials --trial-dir trials/ --regions config/optitrack_regions.json \\
        --ground-truth config/ground_truth_positions.json

    # only some methods, one plot per trial
    python -m acquisition.localize_trials --trial-dir trials/ --regions config/optitrack_regions.json \\
        --methods mht,cphd --plots trial_plots/

ground_truth_positions.json:
    {"trial001": [{"class_name": "box", "x": 0.52, "y": -1.10},
                  {"class_name": "box", "x": -0.30, "y": 0.05}],
     "trial002": []}

Robot poses: MHT, PHD and CPHD use the field of view of every
robot (a robot that did NOT see an object it could have seen is
information). This needs the robot pose in every trial row
("robot_pose"), which static_trial.py saves itself. Trials recorded
before this change have no pose: every robot is then treated as if it
saw the whole arena (and the script says so).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from acquisition.merge_static_trials import load_trials, group_by_trial
from vf.fusion.localization_methods import METHODS, measurements_by_class, run_method, ospa
from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_data,
                                    arena_from_regions, sensor_from_arguments,
                                    zones_from_arguments)


def prepare_trial(rows):
    """[(robot_id, row)] of a trial -> order, poses, {class_name: reports}.

    The rows are the robots' observations (distance, bearing, class and
    the camera pose of the frame); measurements_by_class turns them into
    world positions, the first step of every method. Rows saved before
    the bearing was recorded fall back to their stored position."""
    robots = {}
    for rid, r in rows:
        robot = robots.setdefault(rid, dict(pose=None, observations=[]))
        if robot["pose"] is None and r.get("robot_pose") is not None:
            robot["pose"] = tuple(r["robot_pose"])
        if r.get("class_name"):
            robot["observations"].append(dict(
                class_name=r["class_name"], distance=r.get("distance"),
                bearing=r.get("bearing_rad"), position=r.get("position"),
                pose=tuple(r["robot_pose"]) if r.get("robot_pose") is not None else None))
    order, poses, per_class, n_fallback = measurements_by_class(robots, require_pose=False)
    if n_fallback:
        print(f"  ({n_fallback} observation(s) without bearing: stored position used)")
    return order, poses, per_class


def _fmt(objects):
    return " ".join(f"({x:+.2f},{y:+.2f})" for x, y in objects) or "-"


def plot_trial(trial_id, class_name, order, poses, reports, results, truth,
               sensor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(results)
    columns = min(4, len(methods))
    plot_rows = (len(methods) + columns - 1) // columns
    fig, axes = plt.subplots(plot_rows, columns, figsize=(4.2 * columns, 3.8 * plot_rows),
                             squeeze=False)
    colors = ["#2B6CB0", "#D9822B", "#3F9C6B", "#7C5CBF", "#B3402A"]
    x0, x1, y0, y1 = sensor.arena
    for k, name in enumerate(methods):
        ax = axes[k // columns][k % columns]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        for i, rid in enumerate(order):
            c = colors[i % len(colors)]
            zs = reports.get(rid, [])
            if zs:
                ax.scatter([z[0] for z in zs], [z[1] for z in zs], s=18, color=c, alpha=0.7)
            if poses.get(rid) is not None:
                px, py, yaw = poses[rid]
                ax.plot(px, py, marker=(3, 0, np.degrees(yaw) - 90), ms=11, color=c)
        if truth:
            ax.scatter([v[0] for v in truth], [v[1] for v in truth], s=160,
                       facecolors="none", edgecolors="#1A2332", linewidths=1.5)
        objs = results[name]["objects"]
        if objs:
            ax.scatter([o[0] for o in objs], [o[1] for o in objs], marker="x", s=70,
                       color="#1A2332", linewidths=2)
        ax.set_title(f"{name}: n={results[name]['n']}", fontsize=10)
        ax.tick_params(labelsize=7)
    for k in range(len(methods), plot_rows * columns):
        axes[k // columns][k % columns].axis("off")
    fig.suptitle(f"{trial_id} - {class_name}   (dots = robot reports, "
                 f"x = estimate, circles = ground truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--regions", default=None,
                   help="regions json: arena boundaries (the zones of zone_class "
                        "are a grid of --zone-side cells over it)")
    p.add_argument("--arena", type=float, nargs=4, default=None,
                   metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                   help="arena boundaries, if you do not want those of --regions")
    p.add_argument("--methods", default="all",
                   help="comma-separated list among: " + ",".join(METHODS))
    p.add_argument("--trial", default=None, help="only this trial_id")
    p.add_argument("--ground-truth", default=None, help="true positions, see above")
    p.add_argument("--ospa-c", type=float, default=0.5,
                   help="cost in metres of an extra or missing object (OSPA)")
    p.add_argument("--plots", default=None, help="folder for one png per trial/class")
    add_sensor_arguments(p)
    a = p.parse_args()

    methods = list(METHODS) if a.methods == "all" else [m.strip() for m in a.methods.split(",")]
    for m in methods:
        if m not in METHODS:
            p.error(f"unknown method {m!r}, choose among {list(METHODS)}")

    trials = group_by_trial(load_trials(a.trial_dir))
    if a.trial:
        trials = {a.trial: trials[a.trial]}
    if not trials:
        sys.exit(f"no trials in {a.trial_dir}")

    if a.arena:
        arena = tuple(a.arena)
    elif a.regions:
        arena = arena_from_regions(a.regions)
    else:
        points = [r["position"] for rr in trials.values() for _, r in rr if r.get("position")]
        arena = arena_from_data(points)
    sensor = sensor_from_arguments(a, arena)
    zones = zones_from_arguments(a, arena)
    print("sensor model:", sensor.describe())
    print(f"zones: grid of {len(zones)} cells of at least {a.zone_side:g} m x {a.zone_side:g} m")

    ground_truth = None
    if a.ground_truth:
        with open(a.ground_truth) as f:
            ground_truth = json.load(f)
    if a.plots:
        os.makedirs(a.plots, exist_ok=True)

    ospa_values = defaultdict(list)
    count_ok = defaultdict(list)
    without_poses = []
    for trial_id in sorted(trials):
        order, poses, per_class = prepare_trial(trials[trial_id])
        if any(v is None for v in poses.values()):
            without_poses.append(trial_id)
        trial_truth = ground_truth.get(trial_id) if ground_truth is not None else None
        classes = set(per_class)
        if trial_truth:
            classes |= {v["class_name"] for v in trial_truth}
        print(f"\n[{trial_id}] robots: {', '.join(order)}"
              + ("" if all(poses.values()) else "   (pose missing for some robot: "
                 "field of view NOT used)"))
        if not classes:
            print("  no objects reported")
        for class_name in sorted(classes):
            reports = per_class.get(class_name, {})
            truth = ([(v["x"], v["y"]) for v in trial_truth if v["class_name"] == class_name]
                     if trial_truth is not None else None)
            n_reports = ", ".join(f"{rid}={len(reports.get(rid, []))}" for rid in order)
            print(f"  {class_name}: reports per robot ({n_reports})"
                  + (f"   ground truth: {len(truth)} {_fmt(truth)}" if truth is not None else ""))
            results = {}
            for m in methods:
                out = run_method(m, order, reports, poses, sensor, zones=zones)
                results[m] = out
                line = f"    {m:<15} n={out['n']}  {_fmt(out['objects'])}"
                if truth is not None:
                    d = ospa(out["objects"], truth, c=a.ospa_c)
                    ospa_values[m].append(d)
                    count_ok[m].append(out["n"] == len(truth))
                    line += f"   OSPA={d:.3f} m"
                print(line)
            if a.plots:
                plot_trial(trial_id, class_name, order, poses, reports, results, truth,
                           sensor, os.path.join(a.plots, f"{trial_id}_{class_name}.png"))

    if ospa_values:
        n = len(next(iter(ospa_values.values())))
        print(f"\nSUMMARY over {n} (trial, class) pairs with ground truth:")
        print(f"  {'method':<15} {'mean OSPA [m]':>14} {'exact count':>17}")
        for m in methods:
            print(f"  {m:<15} {np.mean(ospa_values[m]):>14.3f} "
                  f"{100 * np.mean(count_ok[m]):>16.0f}%")
    if without_poses:
        print(f"\n{len(without_poses)} trials without robot poses (trials recorded before "
              f"this change have no pose): the field of view is not used for them.")


if __name__ == "__main__":
    main()