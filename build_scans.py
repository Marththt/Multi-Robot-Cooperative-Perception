"""Offline analysis of the autonomous survey (survey.py): checks that the
data make sense, composes scans from the observation points and runs the
five localization methods on them, to find WHEN the classical methods
fail.

Input
-----
<survey-dir>/*.jsonl written by survey.py (one file per robot) and the
ground truth of every layout (tools/record_ground_truth.py).

One capture = one OBSERVATION POINT: the camera pose and the
observations (class, distance d, bearing beta) taken from one stop with
one yaw offset. The scene is static, so observation points taken at
different times by different robots can be put together.

Step 1 - single measurements (does the data make sense?)
---------------------------------------------------------
Every observation is turned into its measurement z (the same first step
of every method, measurements_by_class) and compared with the nearest
true box of the same class:
  - position error of z: bias (mean error x, y) and RMS, also per
    distance band;
  - range error d - d_true and bearing error beta - beta_true (a slope of
    beta against beta_true close to -1 instead of +1 means the bearing sign
    convention is wrong; a range error that grows with d means a wrong
    marker size or calibration);
  - missed detections: true boxes inside the field of view of the sensor
    model (--hfov-deg, --r-max) that the capture did not report;
  - out-of-model detections: boxes reported although the sensor model
    says they are outside the field of view (FoV or r_max too small);
  - false measurements: z farther than --gate-m from every true box;
  - duplicates: two measurements of the same box in one capture.
All of it overall, per distance band and PER ROBOT (a wrong calibration,
camera offset or sign is usually in one robot only and is flagged there).

Step 2 - scans and methods
--------------------------
For every layout and every K in --k, --scans-per-k random sets of K
observation points from DISTINCT viewpoints (different stop positions;
seeded, reproducible) are drawn; each set is one scan with K "virtual
robots". The five methods run on every scan and class: OSPA (c = 0.5 m,
p = 1) and exact count against the ground truth. Every scan is also
described by what can make a method fail:
  - coverage of each box (how many of the K points have it in the field
    of view) -> min_cov;
  - missed detections and false measurements in the scan;
  - mean distance between the viewpoints and the boxes;
  - smallest distance between two true boxes of the layout (min_sep).

Output
------
<out>/measurements.csv  one row per measurement (step 1)
<out>/scans.csv         one row per scan, class and method (step 2)
and a summary on screen: per method, per K, per layout, per minimum
coverage, with/without missed detections, per distance band.

    python build_scans.py --survey-dir survey/ --ground-truth config/ground_truth_positions.json \\
        --regions config/optitrack_regions.json --out analysis/

    python build_scans.py --survey-dir survey/ --ground-truth config/ground_truth_positions.json \\
        --k 1 2 3 4 6 --scans-per-k 300 --methods zone_class,union,mht --plots analysis/
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

from vf.fusion.localization_methods import METHODS, measurement_from_observation, ospa, run_method
from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_regions,
                                    sensor_from_arguments, zones_from_arguments)

ARENA_DEFAULT = (-2.690, 2.135, -2.672, 1.519)
DISTANCE_BANDS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, float("inf"))


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def band(d):
    for lo, hi in zip(DISTANCE_BANDS[:-1], DISTANCE_BANDS[1:]):
        if lo <= d < hi:
            return f"{lo:.1f}-{hi:.1f}" if hi != float("inf") else f">{lo:.1f}"
    return "?"


# ====================================================================
# Loading
# ====================================================================

def load_points(survey_dir, range_fit=None):
    """All the observation points of the survey: {key: point}, point =
    dict(trial_id, layout, run, robot, stop, yaw_offset, radius, angle, pose,
    observations=[dict(class_name, distance, bearing)]).

    survey_dir can be one run folder of launch_all.py or the folder with
    all the runs: the *.jsonl files are searched in every subfolder (except
    raw/ and sync/), and every run stays separate (run_id in the key)."""
    points = {}
    bad = 0
    paths = sorted(glob.glob(os.path.join(survey_dir, "**", "*.jsonl"), recursive=True))
    paths = [q for q in paths
             if not {"raw", "sync"} & set(os.path.normpath(os.path.relpath(q, survey_dir)).split(os.sep)[:-1])]
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if "layout_id" not in r:
                    continue  # not a survey row (e.g. a static_trial.py file)
                # captures of a round-mode survey get their own key: an older
                # survey in the same file may have used the same trial_id
                key = r["trial_id"] if r.get("round") is None else f"{r['trial_id']}@r{r['round']}"
                if r.get("run_id"):
                    key += f"@{r['run_id']}"
                pt = points.setdefault(key, dict(
                    trial_id=key, layout=r["layout_id"], run=r.get("run_id") or "", robot=r["robot_id"],
                    stop=r["stop"], yaw_offset=r["yaw_offset_deg"], radius=r["radius_m"],
                    angle=r["angle_deg"], round=r.get("round"), t=r.get("t", 0.0),
                    pose=None, observations=[]))
                if pt["pose"] is None and r.get("robot_pose") is not None:
                    pt["pose"] = tuple(r["robot_pose"])
                if r.get("class_name") and r.get("distance") is not None and r.get("bearing_rad") is not None:
                    pt["observations"].append(dict(
                        class_name=r["class_name"],
                        distance=(float(r["distance"]) if range_fit is None
                                  else (float(r["distance"]) - range_fit[1]) / range_fit[0]),
                        bearing=float(r["bearing_rad"]),
                        pose=tuple(r["robot_pose"]) if r.get("robot_pose") is not None else None))
    if bad:
        print(f"warning: {bad} unreadable line(s) skipped")
    no_pose = [k for k, p in points.items() if p["pose"] is None]
    for k in no_pose:
        del points[k]
    if no_pose:
        print(f"warning: {len(no_pose)} capture(s) without camera pose skipped")
    return points


def bind_run_truth(points, truth):
    """Captures of a run whose boxes were read at its start (launch_all.py
    writes them in the ground truth under the run id) are analysed against
    THOSE boxes: their 'layout' becomes the run id (the name given with
    --layout-id is kept in 'layout_name'). Runs without their own entry
    keep the --layout-id entry."""
    for p in points.values():
        p.setdefault("layout_name", p["layout"])
        if p.get("run") and p["run"] in truth:
            p["layout"] = p["run"]
    return points


def load_truth(path):
    with open(path) as f:
        data = json.load(f)
    return {k: [dict(class_name=b["class_name"], xy=np.array([b["x"], b["y"]], float)) for b in v]
            for k, v in data.items()}


# ====================================================================
# Step 1: single measurements against the ground truth
# ====================================================================

def check_point(pt, truth, sensor, gate_m):
    """Measurements of one observation point compared with the truth.
    Returns (rows, in_fov, detected) where rows are per-measurement dicts,
    in_fov the indices of the true boxes the sensor model says are visible
    and detected the indices of the boxes matched by a measurement."""
    rows, detected = [], set()
    for ob in pt["observations"]:
        pose = ob["pose"] or pt["pose"]
        z = measurement_from_observation(pose, ob["distance"], ob["bearing"])
        same = [i for i, b in enumerate(truth) if b["class_name"] == ob["class_name"]]
        if not same:
            rows.append(dict(trial_id=pt["trial_id"], layout=pt["layout"], robot=pt["robot"],
                             class_name=ob["class_name"], box=-1, z_x=z[0], z_y=z[1], err=None,
                             verdict="false"))
            continue
        i = min(same, key=lambda j: float(np.hypot(*(z - truth[j]["xy"]))))
        t = truth[i]["xy"]
        err = float(np.hypot(*(z - t)))
        dx, dy = t[0] - pose[0], t[1] - pose[1]
        d_true = math.hypot(dx, dy)
        b_true = wrap_pi(pose[2] - math.atan2(dy, dx))
        verdict = "false" if err > gate_m else (
            "duplicate" if i in detected else
            "out_of_model" if sensor.p_d_one(pose, t) <= 0 else "ok")
        if verdict != "false":
            detected.add(i)
        rows.append(dict(trial_id=pt["trial_id"], layout=pt["layout"], robot=pt["robot"],
                         class_name=ob["class_name"], box=i, z_x=z[0], z_y=z[1],
                         err=err, err_x=z[0] - t[0], err_y=z[1] - t[1],
                         d=ob["distance"], d_true=d_true, d_err=ob["distance"] - d_true,
                         bearing_deg=math.degrees(ob["bearing"]), bearing_true_deg=math.degrees(b_true),
                         bearing_err_deg=math.degrees(wrap_pi(ob["bearing"] - b_true)),
                         verdict=verdict))
    in_fov = {i for i, b in enumerate(truth) if sensor.p_d_one(pt["pose"], b["xy"]) > 0}
    return rows, in_fov, detected


def step1(points, truth, sensor, gate_m):
    all_rows, info = [], {}
    for k, pt in points.items():
        rows, in_fov, detected = check_point(pt, truth[pt["layout"]], sensor, gate_m)
        all_rows += rows
        info[k] = dict(in_fov=in_fov, detected=detected,
                       missed=in_fov - detected,
                       n_false=sum(1 for r in rows if r["verdict"] == "false"))
    return all_rows, info


def print_step1(rows, info):
    good = [r for r in rows if r["verdict"] in ("ok", "out_of_model", "duplicate")]
    n_fov = sum(len(v["in_fov"]) for v in info.values())
    n_missed = sum(len(v["missed"]) for v in info.values())
    verdicts = defaultdict(int)
    for r in rows:
        verdicts[r["verdict"]] += 1
    print("\n=== Step 1: single measurements against the ground truth ===")
    print(f"captures {len(info)}, measurements {len(rows)}: "
          + ", ".join(f"{v} {n}" for v, n in sorted(verdicts.items())))
    if n_fov:
        print(f"boxes in the field of view {n_fov}, missed {n_missed} "
              f"({100 * n_missed / n_fov:.1f}%)")
    if not good:
        print("no measurement matched a true box: check poses, frame and units")
        return
    ex = np.array([r["err_x"] for r in good])
    ey = np.array([r["err_y"] for r in good])
    e = np.array([r["err"] for r in good])
    de = np.array([r["d_err"] for r in good])
    be = np.array([r["bearing_err_deg"] for r in good])
    print(f"position error: bias ({ex.mean():+.3f}, {ey.mean():+.3f}) m, RMS {math.sqrt((e ** 2).mean()):.3f} m, "
          f"95th percentile {np.percentile(e, 95):.3f} m")
    print(f"range error d - d_true: mean {de.mean():+.3f} m, std {de.std():.3f} m")
    print(f"bearing error: mean {be.mean():+.2f} deg, std {be.std():.2f} deg")
    bt = np.array([r["bearing_true_deg"] for r in good])
    bm = np.array([r["bearing_deg"] for r in good])
    if bt.std() > 1.0:
        slope = float(np.polyfit(bt, bm, 1)[0])
        flag = "  <-- bearing sign convention looks WRONG" if slope < 0 else ""
        print(f"bearing slope (measured vs true): {slope:+.3f} (should be close to +1){flag}")
    d = np.array([r["d_true"] for r in good])
    if d.std() > 0.1:
        slope = float(np.polyfit(d, de, 1)[0])
        print(f"range error growth: {100 * slope:+.1f}% of the distance "
              f"(far from 0 = marker size or calibration)")
        A, B = np.polyfit(d, d + de, 1)
        print(f"range model: measured d = {A:.3f} * true d {B:+.3f} m"
              + (f"  -> to correct it: --range-fit {A:.3f} {B:.3f}" if abs(A - 1) > 0.03 or abs(B) > 0.03 else ""))
    print(f"  {'distance band':<14} {'n':>5} {'RMS [m]':>8} {'range err':>10} {'bearing err':>12}")
    groups = defaultdict(list)
    for r in good:
        groups[band(r["d_true"])].append(r)
    for b in sorted(groups, key=lambda s: float(s.strip(">").split("-")[0])):
        g = groups[b]
        ee = np.array([r["err"] for r in g])
        print(f"  {b:<14} {len(g):>5} {math.sqrt((ee ** 2).mean()):>8.3f} "
              f"{np.mean([r['d_err'] for r in g]):>+10.3f} "
              f"{np.mean([r['bearing_err_deg'] for r in g]):>+11.2f}d")
    # calibration and conventions are per robot: a wrong one hides in the
    # overall numbers, so the same checks are repeated robot by robot
    print("  per robot (a problem is usually in ONE robot: calibration, camera offset, sign):")
    print(f"    {'robot':<6} {'match':>6} {'false':>6} {'missed':>7} {'bias x':>7} {'bias y':>7} "
          f"{'RMS':>6} {'d err':>7} {'d growth':>9} {'b slope':>8}")
    robots = sorted({r["robot"] for r in rows})
    rms_of = {rid: math.sqrt(np.mean([r["err"] ** 2 for r in good if r["robot"] == rid]))
              for rid in robots if any(r["robot"] == rid for r in good)}
    rms_median = float(np.median(list(rms_of.values()))) if rms_of else 0.0
    for rid in robots:
        g = [r for r in good if r["robot"] == rid]
        n_false = sum(1 for r in rows if r["robot"] == rid and r["verdict"] == "false")
        fov = sum(len(v["in_fov"]) for k, v in info.items() if k.split("/")[1] == rid)
        miss = sum(len(v["missed"]) for k, v in info.items() if k.split("/")[1] == rid)
        if not g:
            print(f"    {rid:<6} {0:>6} {n_false:>6}  no measurement matched a true box  <-- CHECK")
            continue
        ee = np.array([r["err"] for r in g])
        de = np.array([r["d_err"] for r in g])
        d = np.array([r["d_true"] for r in g])
        bt = np.array([r["bearing_true_deg"] for r in g])
        bm = np.array([r["bearing_deg"] for r in g])
        growth = 100 * float(np.polyfit(d, de, 1)[0]) if d.std() > 0.1 else float("nan")
        bslope = float(np.polyfit(bt, bm, 1)[0]) if bt.std() > 1.0 else float("nan")
        flags = []
        if n_false > 0.1 * (len(g) + n_false):
            flags.append("many false")
        if bslope < 0.8:
            flags.append("bearing sign/scale")
        if abs(growth) > 5:
            flags.append("range scale")
        if len(rms_of) > 1 and rms_of[rid] > 2 * rms_median:
            flags.append("large error")
        print(f"    {rid:<6} {len(g):>6} {n_false:>6} {100 * miss / max(fov, 1):>6.1f}% "
              f"{np.mean([r['err_x'] for r in g]):>+7.3f} {np.mean([r['err_y'] for r in g]):>+7.3f} "
              f"{math.sqrt((ee ** 2).mean()):>6.3f} {de.mean():>+7.3f} {growth:>+8.1f}% {bslope:>+8.3f}"
              + (f"  <-- {', '.join(flags)}" if flags else ""))


# ====================================================================
# Step 2: scans
# ====================================================================

def draw_scans(points, layout, k, n_scans, rng):
    """n_scans random sets of k observation points of `layout`, at most one
    per viewpoint (radius, angle). Without enough viewpoints: as many
    distinct sets as possible."""
    by_view = defaultdict(list)
    for key, pt in points.items():
        if pt["layout"] == layout:
            by_view[(pt["radius"], pt["angle"])].append(key)
    views = sorted(by_view)
    if len(views) < k:
        return []
    seen, scans = set(), []
    tries = 0
    while len(scans) < n_scans and tries < 20 * n_scans:
        tries += 1
        chosen = tuple(sorted(rng.choice(by_view[v]) for v in rng.sample(views, k)))
        if chosen in seen:
            continue
        seen.add(chosen)
        scans.append(chosen)
    return scans


def min_separation(boxes):
    s = [float(np.hypot(*(a["xy"] - b["xy"]))) for i, a in enumerate(boxes) for b in boxes[i + 1:]]
    return min(s) if s else float("nan")


def run_scan(members, points, info, truth, sensor, zones, methods):
    """One scan: rows of the CSV, one per class and method."""
    robots = {}
    for key in members:
        pt = points[key]
        robots[key] = dict(pose=pt["pose"], observations=pt["observations"])
    order = sorted(robots)
    poses = {k: robots[k]["pose"] for k in order}
    per_class = {}
    for k in order:
        for ob in robots[k]["observations"]:
            z = measurement_from_observation(ob["pose"] or poses[k], ob["distance"], ob["bearing"])
            per_class.setdefault(ob["class_name"], {}).setdefault(k, []).append(z)
    boxes = truth
    classes = sorted({b["class_name"] for b in boxes} | set(per_class))
    cov = [sum(1 for k in members if i in info[k]["in_fov"]) for i in range(len(boxes))]
    seen_by = [sum(1 for k in members if i in info[k]["detected"]) for i in range(len(boxes))]
    n_missed = sum(len(info[k]["missed"]) for k in members)
    n_false = sum(info[k]["n_false"] for k in members)
    ranges = [float(np.hypot(*(b["xy"] - np.array(points[k]["pose"][:2])))) for k in members for b in boxes]
    out = []
    for c in classes:
        idx = [i for i, b in enumerate(boxes) if b["class_name"] == c]
        true_xy = [boxes[i]["xy"] for i in idx]
        reports = per_class.get(c, {})
        for m in methods:
            if reports:
                res = run_method(m, order, reports, poses, sensor, zones=zones)
                est = res["objects"]
            else:
                est = []
            out.append(dict(
                method=m, class_name=c, n_true=len(true_xy), n_est=len(est),
                exact=int(len(est) == len(true_xy)), ospa=ospa(est, true_xy),
                min_cov=min((cov[i] for i in idx), default=0),
                min_seen=min((seen_by[i] for i in idx), default=0),
                n_missed=n_missed, n_false=n_false,
                mean_range=float(np.mean(ranges)) if ranges else float("nan")))
    return out


def summary(rows, key, title, methods, fmt=str):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r[key]][r["method"]].append(r)
    print(f"\n  by {title}")
    head = "".join(f"{m:>16}" for m in methods)
    print(f"  {'':<12}{'n':>7}{head}")
    for g in sorted(groups):
        n = len(next(iter(groups[g].values())))
        cells = ""
        for m in methods:
            rr = groups[g].get(m, [])
            if rr:
                cells += f"{100 * np.mean([r['exact'] for r in rr]):>8.0f}% {np.mean([r['ospa'] for r in rr]):>6.3f}"
            else:
                cells += f"{'-':>16}"
        print(f"  {fmt(g):<12}{n:>7}{cells}")


def print_step2(rows, methods):
    print("\n=== Step 2: methods on the composed scans ===")
    print("  every cell: exact count [%]  mean OSPA [m]")
    for r in rows:
        r["all"] = "all"
        r["missed_any"] = "missed>0" if r["n_missed"] else "missed=0"
        r["cov_group"] = min(r["min_cov"], 3)
        r["range_band"] = band(r["mean_range"])
    summary(rows, "all", "overall", methods)
    summary(rows, "K", "K (observation points per scan)", methods)
    summary(rows, "layout", "layout", methods)
    summary(rows, "cov_group", "min coverage (points that have the least covered box in view)",
            methods, fmt=lambda g: f"{g}+" if g == 3 else str(g))
    summary(rows, "missed_any", "missed detections in the scan", methods)
    summary(rows, "range_band", "mean viewpoint-box distance [m]", methods)
    # direction of the count errors
    print("\n  count errors: too many / too few (over all scans)")
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        over = sum(1 for r in rr if r["n_est"] > r["n_true"])
        under = sum(1 for r in rr if r["n_est"] < r["n_true"])
        print(f"    {m:<12} over {100 * over / len(rr):5.1f}%   under {100 * under / len(rr):5.1f}%")


def plots(meas, rows, methods, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    good = [r for r in meas if r["verdict"] != "false"]
    if good:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        ax[0].scatter([r["d_true"] for r in good], [r["err"] for r in good], s=6)
        ax[0].set(xlabel="true distance [m]", ylabel="position error [m]", title="measurement error")
        ax[1].scatter([r["d_true"] for r in good], [r["d_err"] for r in good], s=6)
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set(xlabel="true distance [m]", ylabel="d - d_true [m]", title="range error")
        ax[2].scatter([r["bearing_true_deg"] for r in good], [r["bearing_deg"] for r in good], s=6)
        lim = max(abs(r["bearing_true_deg"]) for r in good) + 5
        ax[2].plot([-lim, lim], [-lim, lim], "k", lw=0.8)
        ax[2].set(xlabel="true bearing [deg]", ylabel="measured bearing [deg]", title="bearing")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "measurements.png"), dpi=150)
        plt.close(fig)
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ks = sorted({r["K"] for r in rows})
        for m in methods:
            ax[0].plot(ks, [100 * np.mean([r["exact"] for r in rows if r["K"] == k and r["method"] == m])
                            for k in ks], "o-", label=m)
            ax[1].plot(ks, [np.mean([r["ospa"] for r in rows if r["K"] == k and r["method"] == m])
                            for k in ks], "o-", label=m)
        ax[0].set(xlabel="K (observation points per scan)", ylabel="exact count [%]")
        ax[1].set(xlabel="K (observation points per scan)", ylabel="mean OSPA [m]")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "methods_vs_k.png"), dpi=150)
        plt.close(fig)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--survey-dir", required=True, help="folder of the survey.py files")
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--regions", default=None, help="regions JSON, for the arena bounds")
    p.add_argument("--out", default="analysis/")
    p.add_argument("--layouts", nargs="*", default=None, help="only these layouts (default: all)")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 6],
                   help="observation points per scan")
    p.add_argument("--scans-per-k", type=int, default=200)
    p.add_argument("--yaw-offsets", type=float, nargs="*", default=None,
                   help="use only the captures with these yaw offsets (default: all)")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--range-fit", type=float, nargs=2, default=None, metavar=("A", "B"),
                   help="correct every measured distance with the model measured = A * true + B "
                        "(Step 1 prints A and B): distance used = (measured - B) / A")
    p.add_argument("--gate-m", type=float, default=0.5,
                   help="a measurement farther than this from every true box is a false measurement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", default=None, help="folder for the figures (optional)")
    add_sensor_arguments(p)
    a = p.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        p.error(f"unknown method(s) {unknown}; available: {list(METHODS)}")
    arena = arena_from_regions(a.regions) if a.regions else ARENA_DEFAULT
    sensor = sensor_from_arguments(a, arena)
    zones = zones_from_arguments(a, arena)
    truth = load_truth(a.ground_truth)

    points = bind_run_truth(load_points(a.survey_dir, a.range_fit), truth)
    if a.range_fit:
        print(f"distances corrected: (measured - {a.range_fit[1]:+.3f}) / {a.range_fit[0]:.3f}")
    if a.yaw_offsets is not None:
        points = {k: v for k, v in points.items() if v["yaw_offset"] in a.yaw_offsets}
    missing = sorted({v["layout"] for v in points.values()} - set(truth))
    if missing:
        print(f"warning: no ground truth for layout(s) {missing}: skipped")
    layouts = sorted({v["layout"] for v in points.values()} & set(truth))
    if a.layouts:
        names = {v["layout"]: v.get("layout_name", v["layout"]) for v in points.values()}
        layouts = [l for l in layouts if l in a.layouts or names.get(l) in a.layouts]
    points = {k: v for k, v in points.items() if v["layout"] in layouts}
    if not points:
        sys.exit("no observation point to analyse")
    print(f"{len(points)} observation point(s), layouts {layouts}; sensor: {sensor.describe()}")
    for l in layouts:
        robots = sorted({v["robot"] for v in points.values() if v["layout"] == l})
        print(f"  {l}: {len(truth[l])} box(es), min separation {min_separation(truth[l]):.2f} m, "
              f"{sum(1 for v in points.values() if v['layout'] == l)} captures by {robots}")

    meas, info = step1(points, truth, sensor, a.gate_m)
    print_step1(meas, info)

    rng = random.Random(a.seed)
    rows = []
    for l in layouts:
        for k in a.k:
            scans = draw_scans(points, l, k, a.scans_per_k, rng)
            for s, members in enumerate(scans):
                for r in run_scan(members, points, info, truth[l], sensor, zones, methods):
                    r.update(layout=l, K=k, scan=f"{l}/K{k}/{s:04d}", members=" ".join(members),
                             min_sep=min_separation(truth[l]))
                    rows.append(r)
    if not rows:
        sys.exit("no scan could be composed (too few viewpoints for the requested K)")
    print_step2(rows, methods)

    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "measurements.csv"), meas,
              ["trial_id", "layout", "robot", "class_name", "box", "verdict", "z_x", "z_y", "err",
               "err_x", "err_y", "d", "d_true", "d_err", "bearing_deg", "bearing_true_deg",
               "bearing_err_deg"])
    write_csv(os.path.join(a.out, "scans.csv"), rows,
              ["scan", "layout", "K", "members", "class_name", "method", "n_true", "n_est", "exact",
               "ospa", "min_cov", "min_seen", "n_missed", "n_false", "mean_range", "min_sep"])
    print(f"\nwritten {os.path.join(a.out, 'measurements.csv')} and {os.path.join(a.out, 'scans.csv')}")
    if a.plots:
        plots(meas, rows, methods, a.plots)
        print(f"figures in {a.plots}")


if __name__ == "__main__":
    main()