"""MHT, marker network and YOLO + marker network on the simulated scalability sets (sim/scaling.py), with the
metrics of the real scenes, before the validation on the lab captures (yolo_gnn/evaluate.py).

Every method sees the same scenes: the marker detections of the simulator (the sensor file fitted on the real runs)
and, for the YOLO stages, the rays of the frames rendered and segmented by yolo_gnn/segment_sim.py. Only the boxes
in the field of view and range of at least one robot are scored (sim/scaling.py, "covered").

Free parameters, chosen on the tune set (same distribution as the training set, own seed, never a test scene),
by the mean scene F1 (2 matched / (true + estimated)):
  MHT                  the birth density (boxes per m^2 of surveilled area) in --mht-prior-grid; H = --mht-h
  YOLO + marker        the thresholds of yolo_gnn/evaluate.py (GRID), for every variant and marker network
The marker networks (--gnn NAME=DIR, gnn/train_sim.py) and the YOLO-seg model are fixed.

Metrics per test set (mean and standard deviation over the seeds of the marker network): correct scenes (count
right and every box within 15 cm), too few / too many boxes, scene F1, detection probability P_D, false estimates
per scene, RMSE* (a missed box counts as 15 cm), median RMSE of the matched boxes, run time per scene of the
fusion (the YOLO segmentation runs on every robot and is not counted). Paired tests against the MHT, scene by
scene and seed by seed: McNemar on "correct" and a sign test on the scene F1.

Outputs in --out: test_scenes.csv (one row per scene, method, marker network and seed), summary.csv, paired.csv,
tune.csv (every parameter set on the tune set), chosen.json, rays.csv (per set: boxes in view found by the
markers and by the YOLO rays), mht/<set>.csv (MHT rows, reused with --reuse-mht), fig_scaling.png.

    python -m yolo_gnn.evaluate_sim --tune-set data/scale/tune --sets "data/scale/S*" \\
        --gnn plan=results/plan_v2/M4/gnn/detection_full_full_h2 --gnn scale=results/scale/gnn/detection_full_full_h2 \\
        --cache data/yolo_sim_eval/cache --out results/scale/
"""

import argparse
import csv
import glob
import itertools
import json
import math
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace

import cv2
import numpy as np

from gnn.dataset import build_scenes
from gnn.model import decode
from gnn.train_sim import mcnemar, predict, write_csv
from pipeline.evaluate_runs import MATCH_CUTOFF_M, detection_metrics
from sim.scaling import read_geometry
from yolo_gnn.evaluate import GRID, VARIANTS, encode, load_models, ray_diagnostics, run
from yolo_gnn.hybrid import Params, frame_rays
from yolo_gnn.segment_sim import frame_path

REPORTED = ("marker", "inject", "cannot_link", "full")
LABELS = dict(mht="MHT", marker="marker network", inject="+ YOLO boxes", cannot_link="+ YOLO cannot-link",
              full="YOLO + marker network")


# ====================================================================
# Scenes and rays
# ====================================================================

def full_scenes(set_dir, calib_path, max_scenes=None):
    """One scene per run, with EVERY robot (also the ones that reported nothing) and the covered boxes as truth."""
    name = os.path.basename(os.path.normpath(set_dir))
    geo = read_geometry(set_dir)
    built = {s["run"]: s for s in build_scenes(set_dir, os.path.join(set_dir, "ground_truth.json"), calib_path)}
    out = []
    for run in sorted(geo)[:max_scenes]:
        g = geo[run]
        ids = sorted(f"P{k}" for k in range(len(g["poses"])))
        index = {q: n for n, q in enumerate(ids)}
        poses = {f"P{k}": tuple(float(v) for v in p) for k, p in enumerate(g["poses"])}
        boxes = np.array(g["boxes"], float).reshape(-1, 2)
        cov = [int(j) for j in g["covered"]]
        box_index = {j: n for n, j in enumerate(cov)}
        s = built.get(run)
        if s is not None:
            for q, p in zip(s["robots"], s["poses"]):
                poses[q] = tuple(float(v) for v in p)
            det = dict(det_robot=np.array([index[s["robots"][r]] for r in s["det_robot"]], int),
                       distance=s["distance"], bearing=s["bearing"], view=s["view"], pos=s["pos"].reshape(-1, 2),
                       label=np.array([box_index.get(int(b), -1) for b in s["label"]], int))
        else:
            det = dict(det_robot=np.zeros(0, int), distance=np.zeros(0), bearing=np.zeros(0), view=np.zeros(0),
                       pos=np.zeros((0, 2)), label=np.zeros(0, int))
        out.append(dict(run=run, layout=name, round=0, robots=ids, poses=np.array([poses[q] for q in ids], float),
                        truth=boxes[cov].reshape(-1, 2), side=float(g["side_m"]), n_boxes=len(boxes), **det))
    return out


def sim_rays(scenes, set_dir, cache, readings, p):
    """Per scene and robot index: (lens origin, the frame rays of every reading with a cached label image)."""
    name = os.path.basename(os.path.normpath(set_dir))
    path = os.path.join(cache, name, "camera.json")
    if not os.path.exists(path):
        print(f"warning: no YOLO cache for {name} (run yolo_gnn.segment_sim first): no rays")
        return [{} for _ in scenes]
    with open(path) as f:
        cam = json.load(f)
    K, D = np.array(cam["camera_matrix"], float), np.array(cam["dist_coeffs"], float).ravel()
    pitch = math.radians(cam["pitch_deg"])
    out, missing = [], 0
    for s in scenes:
        per_robot = {}
        for qi, q in enumerate(s["robots"]):
            pose = tuple(s["poses"][qi])
            rd = []
            for k in range(readings):
                fp = frame_path(cache, set_dir, s["run"], q, k)
                lab = cv2.imread(fp, cv2.IMREAD_GRAYSCALE) if os.path.exists(fp) else None
                if lab is None:
                    missing += 1
                    continue
                rd.append(frame_rays(lab, K, D, pose, pitch, p))
            if rd:
                per_robot[qi] = (pose[:2], rd)
        out.append(per_robot)
    if missing:
        print(f"warning: {name}: {missing} frames without a cached label image")
    return out


def base_predictions(scenes, a, models):
    enc = [encode(s, a) if len(s["pos"]) else None for s in scenes]
    idx = [n for n, e in enumerate(enc) if e is not None]
    preds = []
    for m in models:
        pr = dict(zip(idx, predict(m, [enc[n] for n in idx]))) if idx else {}
        preds.append([decode(s["pos"], enc[n]["pairs"], *pr[n]) if n in pr else ([], None)
                      for n, s in enumerate(scenes)])
    return enc, preds


def f1(r):
    den = int(r["n_true"]) + int(r["n_est"])
    return 1.0 if den == 0 else 2.0 * int(r["n_matched"]) / den


# ====================================================================
# MHT
# ====================================================================

def _mht_init(model_path):
    from vf.fusion import sensor_aware_mht
    sensor_aware_mht.set_model(model_path, real_terms=False, clutter_factor=1.0)


def _mht_work(task):
    from pipeline.plan_mht import run_scene
    from vf.fusion.sensor_model import SensorModel
    scene, H, density, cutoff, budget = task
    half = scene["side"] / 2 + 0.5
    sensor = SensorModel(n_prior=1.0, arena=(-half, half, -half, half))
    sensor.set_coverage([tuple(p) for p in scene["poses"]])
    sensor.n_prior = density * sensor.observed_area()
    row, _ = run_scene(scene, H, 0, cutoff, sensor, budget or None)
    row["dn"] = row["n_est"] - row["n_true"]
    row["f1"] = f1(row)
    return row


def run_mht(pool, scenes, H, density, cutoff, budget):
    tasks = [(s, H, density, cutoff, budget) for s in scenes]
    return list(pool.imap(_mht_work, tasks, chunksize=max(1, len(tasks) // 200)))


# ====================================================================
# Summaries and figure
# ====================================================================

def summarize(rows, cutoff):
    """Metrics of one method on one set: mean and standard deviation over the seeds."""
    per = defaultdict(list)
    for r in rows:
        per[r.get("seed", 0)].append(r)
    vals = defaultdict(list)
    for rs in per.values():
        dn = np.array([int(r["dn"]) for r in rs])
        dm = detection_metrics(rs, cutoff)
        vals["correct_pct"].append(100 * float(np.mean([int(r["correct"]) for r in rs])))
        vals["too_few_pct"].append(100 * float(np.mean(dn < 0)))
        vals["too_many_pct"].append(100 * float(np.mean(dn > 0)))
        vals["f1_mean"].append(float(np.mean([f1(r) for r in rs])))
        vals["pd_pct"].append(dm["pd_pct"])
        vals["false_per_scene"].append(dm["false_per_scene"])
        vals["rmse_all_cm"].append(dm["rmse_all_cm"])
        rm = np.array([float(r["rmse"]) for r in rs if r["rmse"] not in ("", None)], float)
        rm = rm[np.isfinite(rm)]
        vals["rmse_median_cm"].append(100 * float(np.median(rm)) if len(rm) else math.nan)
        vals["seconds_per_scene"].append(float(np.mean([float(r.get("seconds", math.nan)) for r in rs])))
        if any("forward_seconds" in r for r in rs):                    # GNN rows only (MHT has no forward)
            vals["forward_seconds_per_scene"].append(float(np.mean([float(r["forward_seconds"]) for r in rs])))
    out = dict(scenes=len(next(iter(per.values()))), seeds=len(per))
    for k, v in vals.items():
        out[k] = round(float(np.nanmean(v)), 4)
        out[k + "_sd"] = round(float(np.nanstd(v)), 4)
    return out


def sweep_x(name):
    """(sweep, x value) of a test set name: SC_n16 -> ("SC", 16)."""
    sid, _, level = name.partition("_")
    digits = "".join(ch for ch in level if ch.isdigit())
    return sid, int(digits) if digits else None


def figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("correct_pct", "correct scenes [%]"), ("f1_mean", "scene F1"), ("pd_pct", "P_D [%]"),
               ("false_per_scene", "false estimates / scene"), ("seconds_per_scene", "fusion time / scene [s]")]
    sweeps = [(s, x) for s, x in (("SC", "robots = boxes"), ("SR", "robots (16 boxes)"), ("SB", "boxes (16 robots)"))
              if any(sweep_x(r["set"])[0] == s for r in summary)]
    if not sweeps:
        return
    fig, axes = plt.subplots(len(sweeps), len(metrics), figsize=(4.0 * len(metrics), 3.2 * len(sweeps)), squeeze=False)
    curves = sorted({(r["method"], r["gnn"]) for r in summary}, key=lambda c: (c[0] != "mht", c))
    for i, (sid, xlabel) in enumerate(sweeps):
        rows = [r for r in summary if sweep_x(r["set"])[0] == sid]
        if sid in ("SR", "SB"):                         # the middle point of these sweeps is SC_n16
            rows += [dict(r, set=f"{sid}_x16") for r in summary if r["set"] == "SC_n16"]
        for j, (key, label) in enumerate(metrics):
            ax = axes[i][j]
            for method, gnn in curves:
                pts = sorted((sweep_x(r["set"])[1], r[key], r.get(key + "_sd", 0.0)) for r in rows
                             if r["method"] == method and r["gnn"] == gnn)
                if not pts:
                    continue
                x, y, e = map(np.array, zip(*pts))
                name = LABELS.get(method, method) + ("" if method == "mht" else f" ({gnn})")
                ax.errorbar(x, y, yerr=e, marker="o", ms=3, capsize=2, label=name,
                            color="k" if method == "mht" else None, lw=1.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            if key == "seconds_per_scene":
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ====================================================================
# Main
# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="test set folders of sim/scaling.py (glob patterns)")
    ap.add_argument("--tune-set", required=True, help="set folder used to choose the free parameters")
    ap.add_argument("--gnn", action="append", required=True, metavar="NAME=DIR",
                    help="a marker network (gnn/train_sim.py variant folder); repeat to compare several")
    ap.add_argument("--cache", default="data/yolo_sim_eval/cache", help="yolo_gnn/segment_sim.py --cache")
    ap.add_argument("--readings", type=int, default=3, help="frames per robot (yolo_gnn/segment_sim.py --readings)")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--model", default="config/sensor_model.json", help="sensor file given to the MHT")
    ap.add_argument("--mht-h", type=int, default=100)
    ap.add_argument("--mht-prior-grid", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8],
                    help="MHT birth densities tried on the tune set [boxes per m^2 of surveilled area]")
    ap.add_argument("--mht-search-budget", type=int, default=0, help="cap of the k-best search (0: exact)")
    ap.add_argument("--reuse-mht", action="store_true", help="read mht/<set>.csv when it exists")
    ap.add_argument("--max-scenes", type=int, default=None, help="only the first N scenes of every set (quick tests)")
    ap.add_argument("--cutoff", type=float, default=MATCH_CUTOFF_M)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/scale/")
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)
    os.makedirs(os.path.join(a.out, "mht"), exist_ok=True)

    tests = sorted({d for p in a.sets for d in glob.glob(p) if os.path.exists(os.path.join(d, "geometry.jsonl"))},
                   key=lambda d: (sweep_x(os.path.basename(os.path.normpath(d)))[0],
                                  sweep_x(os.path.basename(os.path.normpath(d)))[1] or 0))
    if not tests:
        raise SystemExit(f"no set with geometry.jsonl matches {a.sets}")
    gnns = []
    for spec in a.gnn:
        name, _, d = spec.partition("=")
        va, models = load_models(d)
        gnns.append((name, va, models))
    p0 = Params()
    pool = multiprocessing.Pool(a.jobs, initializer=_mht_init, initargs=(a.model,))

    # ---------------------------------------------------------------- tune set
    t0 = time.time()
    tune = full_scenes(a.tune_set, a.sensor_calib, a.max_scenes)
    tune_rays = sim_rays(tune, a.tune_set, a.cache, a.readings, p0)
    print(f"tune set {a.tune_set}: {len(tune)} scenes", flush=True)
    tune_rows, chosen = [], dict(mht={}, yolo={})
    best = None
    for density in a.mht_prior_grid:
        rows = run_mht(pool, tune, a.mht_h, density, a.cutoff, a.mht_search_budget)
        score = float(np.mean([r["f1"] for r in rows]))
        tune_rows.append(dict(method="mht", gnn="", params=json.dumps(dict(prior_density=density)), f1_mean=round(score, 4),
                              correct_pct=round(100 * float(np.mean([r["correct"] for r in rows])), 2)))
        print(f"  MHT birth density {density:g}/m^2: F1 {score:.3f}", flush=True)
        if best is None or score > best[0]:
            best = (score, density)
    chosen["mht"] = dict(prior_density=best[1], H=a.mht_h)
    for gname, va, models in gnns:
        enc, pred = base_predictions(tune, va, models)
        for variant, spec in VARIANTS.items():
            if variant not in REPORTED:
                continue
            combos = [dict(zip(spec["tune"], v)) for v in itertools.product(*[GRID[k] for k in spec["tune"]])]
            scores = []
            for c in combos:
                rows = run(tune, tune_rays, enc, pred, models, va, replace(p0, **spec["fixed"], **c))
                scores.append(float(np.mean([f1(r) for r in rows])))
                tune_rows.append(dict(method=variant, gnn=gname, params=json.dumps(c), f1_mean=round(scores[-1], 4),
                                      correct_pct=round(100 * float(np.mean([r["correct"] for r in rows])), 2)))
            k = max(range(len(combos)), key=lambda i: (scores[i], -i))
            chosen["yolo"].setdefault(gname, {})[variant] = combos[k]
            print(f"  {gname} {variant:<12} F1 {scores[k]:.3f} with {combos[k]}", flush=True)
    write_csv(os.path.join(a.out, "tune.csv"), tune_rows)
    with open(os.path.join(a.out, "chosen.json"), "w") as f:
        json.dump(dict(chosen, fixed=asdict(p0), tune_set=a.tune_set, readings=a.readings), f, indent=2)
    print(f"parameters chosen on the tune set ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- test sets
    all_rows, summary, paired, rays_rows = [], [], [], []
    for set_dir in tests:
        name = os.path.basename(os.path.normpath(set_dir))
        scenes = full_scenes(set_dir, a.sensor_calib, a.max_scenes)
        rays = sim_rays(scenes, set_dir, a.cache, a.readings, p0)
        meta = dict(set=name, n_robots=round(float(np.mean([len(s["robots"]) for s in scenes])), 1),
                    n_boxes=round(float(np.mean([s["n_boxes"] for s in scenes])), 1),
                    n_covered=round(float(np.mean([len(s["truth"]) for s in scenes])), 1))
        diag = ray_diagnostics(scenes, rays, p0)
        seen = sum(r["boxes_in_view"] for r in diag)
        rays_rows.append(dict(set=name, boxes_in_view=seen,
                              found_marker_pct=round(100 * sum(r["found_marker"] for r in diag) / max(seen, 1), 2),
                              found_yolo_pct=round(100 * sum(r["found_yolo"] for r in diag) / max(seen, 1), 2),
                              yolo_rays=sum(r["yolo_rays"] for r in diag),
                              yolo_false_rays=sum(r["yolo_false_rays"] for r in diag)))

        mht_path = os.path.join(a.out, "mht", f"{name}.csv")
        if a.reuse_mht and os.path.exists(mht_path):
            with open(mht_path) as f:
                mht = [dict(r, seed=0) for r in csv.DictReader(f)]
            for r in mht:
                for k in ("n_true", "n_est", "n_matched", "correct", "dn"):
                    r[k] = int(r[k])
                r["f1"] = float(r["f1"])
        else:
            mht = run_mht(pool, scenes, a.mht_h, chosen["mht"]["prior_density"], a.cutoff, a.mht_search_budget)
            write_csv(mht_path, mht)
        mht_by_run = {r["run"]: r for r in mht}
        summary.append(dict(meta, method="mht", gnn="", **summarize(mht, a.cutoff)))
        all_rows += [dict(r, set=name, method="mht", gnn="", seed=0) for r in mht]
        line = f"{name:<7} N {meta['n_robots']:4.1f} B {meta['n_boxes']:4.1f} | MHT {summary[-1]['correct_pct']:5.1f}% F1 {summary[-1]['f1_mean']:.3f}"

        for gname, va, models in gnns:
            t_fwd = time.time()
            enc, pred = base_predictions(scenes, va, models)
            t_fwd = time.time() - t_fwd                    # encoding + forward of every seed's model on every scene
            for variant in REPORTED:
                p = replace(p0, **VARIANTS[variant]["fixed"], **chosen["yolo"][gname][variant])
                t1 = time.time()
                rows = run(scenes, rays, enc, pred, models, va, p)
                per_scene = (time.time() - t1) / max(1, len(rows))
                # seconds: rays, injection, cannot-link and clustering only; forward_seconds: encoding + network
                # forward, shared by the variants (one row per scene and seed, one forward per scene and seed)
                for r in rows:
                    r.update(set=name, method=variant, gnn=gname, seconds=per_scene,
                             forward_seconds=t_fwd / max(1, len(rows)), f1=f1(r))
                all_rows += rows
                summary.append(dict(meta, method=variant, gnn=gname, **summarize(rows, a.cutoff),
                                    added_boxes_mean=round(float(np.mean([r["added_boxes"] for r in rows])), 3)))
                if variant in ("marker", "full"):
                    line += f" | {gname} {variant} {summary[-1]['correct_pct']:5.1f}% F1 {summary[-1]['f1_mean']:.3f}"
                    for sd in sorted({r["seed"] for r in rows}):
                        rs = [r for r in rows if r["seed"] == sd]
                        cb = sum(1 for r in rs if r["correct"] and not mht_by_run[r["run"]]["correct"])
                        cm = sum(1 for r in rs if mht_by_run[r["run"]]["correct"] and not r["correct"])
                        fb = sum(1 for r in rs if r["f1"] > mht_by_run[r["run"]]["f1"] + 1e-9)
                        fm = sum(1 for r in rs if r["f1"] < mht_by_run[r["run"]]["f1"] - 1e-9)
                        paired.append(dict(set=name, method=variant, gnn=gname, seed=sd, metric="correct",
                                           method_better=cb, mht_better=cm, p_value=mcnemar(cb, cm)))
                        paired.append(dict(set=name, method=variant, gnn=gname, seed=sd, metric="f1",
                                           method_better=fb, mht_better=fm, p_value=mcnemar(fb, fm)))
        print(line, flush=True)
        write_csv(os.path.join(a.out, "summary.csv"), summary)        # after every set: partial results survive
        write_csv(os.path.join(a.out, "paired.csv"), paired)
        write_csv(os.path.join(a.out, "rays.csv"), rays_rows)
    pool.close()
    write_csv(os.path.join(a.out, "test_scenes.csv"), all_rows)
    figure(summary, os.path.join(a.out, "fig_scaling.png"))
    print(f"written in {a.out}: summary.csv, paired.csv, test_scenes.csv, tune.csv, chosen.json, rays.csv, "
          f"fig_scaling.png")


if __name__ == "__main__":
    main()
