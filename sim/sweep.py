"""Failure map of the classical methods: one factor of the scene is changed at a time, everything else
stays at the nominal setting of the real setup (sim/specs/sweep_base.json: 4 boxes, 4 robots on a ring
of 1.3 m, measured sensor model, no obstacles).

For every value of a factor, --runs scenes are simulated (one round each, new seed per scene) and fused by
the five methods with pipeline/evaluate_runs.py, so the metrics are exactly those of the real evaluation.
The seeds depend only on --seed, the factor and the value, so a factor run alone gives the same scenes.

Factors (--factors, default all):
  spacing     distance between neighbouring boxes [m]; pairs, lines and clusters at that pitch
  boxes       number of boxes
  robots      number of robots on the ring
  distance    ring radius, i.e. robot-to-box distance [m]
  occlusion   robots each box is hidden from by an obstacle on the line of sight
  false_rate  lone false detections, x the measured rate
  two_face    two-face reports, x the measured rate

    python -m sim.sweep --model config/sensor_model.json --sensor-calib config/sensor_calib.json \\
        --regions config/optitrack_regions.json --runs 500 --seed 1 --out results/failure_map/

Outputs (in --out): sweep.csv (one row per factor, value and method: exact-count rate with its 95%
interval, correct rate, too few / too many, mean dn, RMSE, OSPA), failure_map.png (exact-count rate of
every method against every factor), mht_errors.png (MHT: too few / too many against every factor).
The simulated runs are kept in --data-dir/<factor>/<value>/ with the spec used for them (a new sweep of
the same value replaces them).
"""

import argparse
import contextlib
import copy
import csv
import io
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from collections import defaultdict

import numpy as np

from pipeline.evaluate_runs import detection_metrics
from sim.generate import random_runs
from sim.model import SensorModel

METHODS = ["zone_class", "union", "phd", "cphd", "mht"]
LABELS = dict(zone_class="Zone+class", union="Union", phd="PHD", cphd="CPHD", mht="MHT")
COLORS = dict(zone_class="#8c8c8c", union="#e6a23c", phd="#5b8def", cphd="#2bb3a3", mht="#d1495b")

# factor: (spec field, values, axis label, extra spec settings)
FACTORS = {
    "spacing": ("spacing_m", [0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60], "box spacing [m]",
                dict(arrangements=["pairs", "line", "cluster"])),
    "boxes": ("n_boxes", [1, 2, 3, 4, 5, 6, 8, 10, 12], "number of boxes", {}),
    "robots": ("n_robots", [2, 3, 4, 5, 6, 8, 10, 15, 20], "number of robots", {}),
    "distance": ("robot_radius_m", [0.8, 1.0, 1.3, 1.6, 2.0, 2.5], "robot-to-centre distance [m]", {}),
    "occlusion": ("hide_per_box", [0, 1, 2, 3], "robots each box is hidden from",
                  dict(targeted_obstacles=True)),
    "false_rate": ("false_factor", [1, 2, 4, 8, 16], "false detections (x measured)", {}),
    "two_face": ("two_face_factor", [1, 2, 5, 10, 20], "two-face reports (x measured)", {}),
}
FACTOR_IDS = {name: i for i, name in enumerate(FACTORS)}   # seeds do not depend on --factors
RANGE_FIELDS = {"spacing_m", "robot_radius_m", "hide_per_box", "false_factor", "two_face_factor"}


def spec_for(base, field, value, extra):
    spec = copy.deepcopy(base)
    spec.update(extra)
    spec[field] = [value, value] if field in RANGE_FIELDS else [value]
    return spec


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (mid - half), 100 * (mid + half)


def evaluate(run_dir, eval_dir, sensor_calib, regions):
    cmd = [sys.executable, "-m", "pipeline.evaluate_runs", "--survey-dir", run_dir,
           "--ground-truth", os.path.join(run_dir, "ground_truth.json"), "--sensor-calib", sensor_calib,
           "--full-only", "--quiet", "--methods", ",".join(METHODS), "--out", eval_dir]
    if regions:
        cmd += ["--regions", regions]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(eval_dir, "scenes.csv")) as f:
        return list(csv.DictReader(f))


def run_value(task):
    """Simulates and evaluates one value of one factor (runs in a worker process with --jobs)."""
    factor, value, spec, run_dir, seed, model_path, runs, sensor_calib, regions = task
    if os.path.isdir(run_dir):              # runs of an earlier sweep of this value are replaced
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    spec_path = os.path.join(run_dir, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    with contextlib.redirect_stdout(io.StringIO()):
        random_runs(Namespace(spec=spec_path, runs=runs, name=f"{factor}_{value}", seed=seed, out=run_dir),
                    SensorModel.load(model_path))
    rows = evaluate(run_dir, os.path.join(run_dir, "evaluation"), sensor_calib, regions)
    rows += empty_scenes(run_dir, rows, f"{factor}_{value}")
    return (factor, value), summarize(rows, factor, value)


def empty_scenes(run_dir, rows, layout_id, cutoff=0.5):
    """Scenes in which no robot reported anything have no feature vector, so evaluate_runs does not see
    them; they are added here as estimates with no box for every method (all boxes missed, OSPA = c)."""
    with open(os.path.join(run_dir, "ground_truth.json")) as f:
        truth = json.load(f)
    seen = {r["run"] for r in rows}
    extra = []
    for run, boxes in truth.items():
        if run == layout_id or run in seen or not boxes:
            continue
        for m in METHODS:
            extra.append(dict(n_true=len(boxes), n_est=0, dn=-len(boxes), n_matched=0, rmse="", ospa=cutoff,
                              run=run, method=m))
    return extra


def report(factor, value, res):
    mht = next((r for r in res if r["method"] == "mht"), None)
    if mht:
        print(f"  {factor + '/' + str(value):<22} MHT exact {mht['exact_pct']:5.1f}%  "
              f"(too few {mht['too_few_pct']:.1f}%, too many {mht['too_many_pct']:.1f}%)", flush=True)


def summarize(rows, factor, value):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = []
    for m in METHODS:
        rs = by_method.get(m, [])
        n = len(rs)
        if not n:
            continue
        dn = np.array([int(r["dn"]) for r in rs])
        exact = int((dn == 0).sum())
        correct = sum(1 for r in rs if int(r["dn"]) == 0 and int(r["n_matched"]) == int(r["n_true"]))
        rmse = np.array([float(r["rmse"]) for r in rs if r["rmse"] not in ("", "nan")], float)
        rmse = rmse[np.isfinite(rmse)]
        lo, hi = wilson(exact, n)
        dm = detection_metrics(rs)
        out.append(dict(factor=factor, value=value, method=m, scenes=n,
                        pd_pct=round(dm["pd_pct"], 2), rmse_all_cm=round(dm["rmse_all_cm"], 2),
                        false_per_scene=round(dm["false_per_scene"], 3),
                        exact_pct=round(100 * exact / n, 1), exact_lo=round(lo, 1), exact_hi=round(hi, 1),
                        correct_pct=round(100 * correct / n, 1),
                        too_few_pct=round(100 * float((dn < 0).mean()), 1),
                        too_many_pct=round(100 * float((dn > 0).mean()), 1),
                        mean_dn=round(float(dn.mean()), 3),
                        rmse_median=round(float(np.median(rmse)), 4) if len(rmse) else "",
                        ospa_mean=round(float(np.mean([float(r["ospa"]) for r in rs])), 4)))
    return out


def place_legend(axes, n):
    """Legend in the first empty panel, or in the first panel when the grid is full."""
    handles, labels = axes.flat[0].get_legend_handles_labels()
    spare = list(axes.flat)[n:]
    for ax in spare:
        ax.axis("off")
    if spare:
        spare[0].legend(handles, labels, loc="center", fontsize=11, frameon=False)
    else:
        axes.flat[0].legend(fontsize=9)


def figures(rows, factors, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    n = len(factors)
    cols = min(4, n)
    nrows = math.ceil(n / cols)

    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.4 * nrows), squeeze=False)
    for ax, factor in zip(axes.flat, factors):
        for m in METHODS:
            rs = [r for r in rows if r["factor"] == factor and r["method"] == m]
            if not rs:
                continue
            x = [r["value"] for r in rs]
            ax.plot(x, [r["exact_pct"] for r in rs], "o-", color=COLORS[m], label=LABELS[m], lw=1.8, ms=4)
            ax.fill_between(x, [r["exact_lo"] for r in rs], [r["exact_hi"] for r in rs],
                            color=COLORS[m], alpha=0.15, lw=0)
        ax.set_xlabel(FACTORS[factor][2])
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.set_ylabel("exact count [%]")
    place_legend(axes, n)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "failure_map.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.4 * nrows), squeeze=False)
    for ax, factor in zip(axes.flat, factors):
        rs = [r for r in rows if r["factor"] == factor and r["method"] == "mht"]
        x = [r["value"] for r in rs]
        ax.plot(x, [r["too_few_pct"] for r in rs], "v-", color="#5b8def", label="too few", lw=1.8)
        ax.plot(x, [r["too_many_pct"] for r in rs], "^-", color="#d1495b", label="too many", lw=1.8)
        ax.plot(x, [r["exact_pct"] - r["correct_pct"] for r in rs], "s--", color="#8c8c8c",
                label="right count, box missed", lw=1.4, ms=4)
        ax.set_xlabel(FACTORS[factor][2])
        ax.set_ylabel("MHT scenes [%]")
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    place_legend(axes, n)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mht_errors.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="sensor model JSON (sim/calibrate.py)")
    ap.add_argument("--sensor-calib", required=True, help="pipeline/calibrate.py --out, as for the real runs")
    ap.add_argument("--regions", default=None, help="arena regions, for zone+class")
    ap.add_argument("--base-spec", default="sim/specs/sweep_base.json")
    ap.add_argument("--factors", default=",".join(FACTORS))
    ap.add_argument("--runs", type=int, default=500, help="scenes per value")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="values simulated in parallel (default: CPU cores - 1)")
    ap.add_argument("--data-dir", default="data/sim_sweep/")
    ap.add_argument("--out", default="results/failure_map/")
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)             # Windows cannot wait on more than 63 process handles

    factors = [f.strip() for f in a.factors.split(",") if f.strip()]
    unknown = [f for f in factors if f not in FACTORS]
    if unknown:
        raise SystemExit(f"unknown factors: {unknown}; known: {list(FACTORS)}")
    with open(a.base_spec) as f:
        base = json.load(f)
    os.makedirs(a.out, exist_ok=True)

    tasks = []
    for fi, factor in enumerate(factors):
        field, values, _, extra = FACTORS[factor]
        for vi, value in enumerate(values):
            run_dir = os.path.join(a.data_dir, factor, str(value))
            seed = a.seed * 1000 + FACTOR_IDS[factor] * 50 + vi    # every factor and value has its own seeds
            tasks.append((factor, value, spec_for(base, field, value, extra), run_dir, seed, a.model, a.runs,
                           a.sensor_calib, a.regions))

    results = {}
    if a.jobs > 1:
        with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
            for (factor, value), res in pool.imap_unordered(run_value, tasks):
                results[(factor, value)] = res
                report(factor, value, res)
    else:
        for t in tasks:
            (factor, value), res = run_value(t)
            results[(factor, value)] = res
            report(factor, value, res)

    rows = [r for t in tasks for r in results[(t[0], t[1])]]
    with open(os.path.join(a.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figures(rows, factors, a.out)
    print(f"\nwritten in {a.out}: sweep.csv, failure_map.png, mht_errors.png")


if __name__ == "__main__":
    main()
