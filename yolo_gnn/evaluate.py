"""YOLO + marker network on all the real scenes, with honest thresholds.

The marker network (gnn/train_sim.py, trained in simulation only) and the YOLO-seg model (yolo_gnn/train_seg.py,
trained in simulation only) are fixed. The few thresholds of the rays (hybrid.Params: min_frac, tol_sym,
cl_robots, same_robot) are chosen by two-fold cross-validation over the LAYOUTS: the layouts are split in two
halves (alternately within each group of layouts), the thresholds that give the most correct scenes on one half
are applied to the other half, and vice versa. Every scene is therefore scored with thresholds chosen without
it, and no scene of a layout is used to choose the thresholds applied to that layout.

Variants (one test_scenes_<variant>.csv each, one row per scene and seed of the marker network):
  marker       marker network alone (no rays)
  same_robot   + two markers of one robot far apart in bearing are different boxes (no YOLO)
  inject       + boxes triangulated from the YOLO rays
  cannot_link  + pairs separated by the YOLO rays
  full         inject + cannot_link (+ same_robot if the cross-validation picks it): the proposed method

Outputs in --out: test_scenes_<variant>.csv, summary.csv (per variant: all scenes, and each half), grid.csv
(every threshold set on every half), cv_params.json (the thresholds picked for each half), rays.csv (per robot and
stop: true boxes in view, found by the markers, by the YOLO rays, YOLO rays on no box).

    python -m yolo_gnn.evaluate --survey-dir data/survey_all/ --runs "data/survey_all/*" \\
        --gnn-dir results/plan_v2/M4/gnn/detection_full_full_h2 --cache data/yolo_real/cache \\
        --mht results/evaluation_all/scenes.csv --out results/yolo_gnn/
"""

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, replace

import cv2
import numpy as np
import torch

from gnn.dataset import build_scenes, load_calibration
from gnn.model import build_model, decode, encode_scene
from gnn.train_sim import mcnemar, predict, read_baseline, write_csv
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics
from yolo_gnn.camera import PITCH_DEG, calibration, camera_pose, load_run, run_dirs
from yolo_gnn.hybrid import Params, frame_rays, group_rays
from yolo_gnn import hybrid
from yolo_gnn.segment_real import cache_path

GRID = dict(min_frac=[0.8, 0.6], tol_sym=[0.18, 0.12, 0.25], cl_robots=[2, 1], same_robot=[False, True])
VARIANTS = dict(
    marker=dict(fixed=dict(inject=False, cannot_link=False, same_robot=False), tune=[]),
    same_robot=dict(fixed=dict(inject=False, cannot_link=False, same_robot=True), tune=[]),
    inject=dict(fixed=dict(inject=True, cannot_link=False, same_robot=False), tune=["min_frac", "tol_sym"]),
    cannot_link=dict(fixed=dict(inject=False, cannot_link=True, same_robot=False), tune=["min_frac", "cl_robots"]),
    full=dict(fixed=dict(inject=True, cannot_link=True), tune=["min_frac", "tol_sym", "cl_robots", "same_robot"]),
)
FOV_HALF = math.radians(44.0)
MAX_RANGE_M = 3.0


def load_models(gnn_dir):
    with open(os.path.join(gnn_dir, "variant.json")) as f:
        a = argparse.Namespace(**json.load(f))
    models = []
    seed = 0
    while os.path.exists(os.path.join(gnn_dir, f"model_seed{seed}.pt")):
        m = build_model(a.graph, a.hidden, a.hops, a.features, a.skip)
        m.load_state_dict(torch.load(os.path.join(gnn_dir, f"model_seed{seed}.pt"), map_location="cpu"))
        m.eval()
        models.append(m)
        seed += 1
    if not models:
        raise SystemExit(f"no model_seed*.pt in {gnn_dir}")
    return a, models


def encode(s, a):
    return encode_scene(s, features=a.features, edges=a.edges, comm_range=a.comm_range, gate=a.gate,
                        association=a.association, hops=a.hops)


def all_scenes(survey_dir, gt_path, calib_path, runs):
    """The scenes of the marker network (gnn/dataset.py), plus the stops at which no robot detected a marker
    (no detection node, but YOLO may still see the boxes)."""
    scenes = build_scenes(survey_dir, gt_path, calib_path)
    have = {(s["layout"], int(s["round"])) for s in scenes}
    off = load_calibration(calib_path)["camera_offset"]
    for d, layout, boxes, frames, cams in runs:
        for rnd in sorted({r for r, k, o in frames if o == 0}):
            if (layout, rnd) in have:
                continue
            recs = frames.get((rnd, 0, 0)) or next(v for (r, k, o), v in frames.items() if r == rnd and o == 0)
            robots = sorted(recs)
            poses = []
            for q in robots:
                x, y, yaw = recs[q]["camera_pose"]
                poses.append((x + off * math.cos(yaw), y + off * math.sin(yaw), yaw))
            scenes.append(dict(run=os.path.basename(os.path.normpath(d)), layout=layout, round=rnd, robots=robots,
                               poses=np.array(poses, float), det_robot=np.zeros(0, int), distance=np.zeros(0),
                               bearing=np.zeros(0), view=np.zeros(0), pos=np.zeros((0, 2)),
                               label=np.zeros(0, int), truth=np.asarray(boxes, float)))
    scenes.sort(key=lambda s: (s["layout"], s["round"]))
    return scenes


def reading_rays(scenes, runs, cache, cal, p):
    """Per scene and robot index: (lens origin, list of the frame rays of every reading with a cached mask)."""
    by_layout = {layout: (d, frames, cams) for d, layout, boxes, frames, cams in runs}
    out, missing = [], 0
    for s in scenes:
        per_robot = {}
        d, frames, cams = by_layout.get(s["layout"], (None, {}, {}))
        for qi, q in enumerate(s["robots"]):
            pitch_deg, yaw_off = cal.get((s["layout"], q), (PITCH_DEG.get(q, 11.5), 0.0))
            origin, readings = None, []
            for k in range(5):
                rec = frames.get((int(s["round"]), k, 0), {}).get(q)
                if rec is None:
                    continue
                path = cache_path(cache, d, rec)
                labels = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
                if labels is None:
                    missing += 1
                    continue
                x, y, yaw = camera_pose(rec)
                pose = (x, y, yaw + math.radians(yaw_off))
                origin = origin if origin is not None else (x, y)
                K, D = cams[q]
                readings.append(frame_rays(labels, K, D, pose, math.radians(pitch_deg), p))
            if readings:
                per_robot[qi] = (origin, readings)
        out.append(per_robot)
    if missing:
        print(f"warning: {missing} frames have no cached YOLO mask (run yolo_gnn.segment_real first)")
    return out


def scene_rays(raw, p):
    return [r for qi, (origin, readings) in raw.items() for r in group_rays(readings, origin, qi, p)]


def folds(scenes, levels_path):
    """Layout -> 0 or 1: alternately within each group of layouts (block column of tools/scene_levels.py)."""
    block = {}
    if levels_path and os.path.exists(levels_path):
        with open(levels_path) as f:
            for r in csv.DictReader(f):
                block[r["layout"]] = (r.get("block") or "").split(" ")[0] or "L"

    def order(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 0, name)

    groups = defaultdict(list)
    for layout in sorted({s["layout"] for s in scenes}, key=order):
        groups[block.get(layout, layout[0])].append(layout)
    return {layout: n % 2 for g in groups.values() for n, layout in enumerate(g)}


def run(scenes, rays_raw, base_enc, base_pred, models, a, p):
    """One row per scene and seed with the thresholds p."""
    rows = []
    rays = [scene_rays(raw, p) if (p.inject or p.cannot_link) else [] for raw in rays_raw]
    for seed, m in enumerate(models):
        work, enc, added = [], [], []
        for n, s in enumerate(scenes):
            sc, k = s, 0
            if p.inject and rays[n]:
                est = base_pred[seed][n][0]
                sc, k = hybrid.inject(s, rays[n], est, p)
            work.append(sc)
            added.append(k)
            enc.append(base_enc[n] if k == 0 else (encode(sc, a) if len(sc["pos"]) else None))
        idx = [n for n, e in enumerate(enc) if e is not None]
        preds = dict(zip(idx, predict(m, [enc[n] for n in idx]))) if idx else {}
        for n, sc in enumerate(work):
            if n in preds:
                ps, pf = preds[n]
                ps = hybrid.cannot_link(sc, enc[n]["pairs"], ps, rays[n], p)
                est, _ = decode(sc["pos"], enc[n]["pairs"], ps, pf)
            else:
                est = []
            r = scene_metrics(est, list(sc["truth"]), MATCH_CUTOFF_M, 2.0)
            r["correct"] = int(r["dn"] == 0 and r["n_matched"] == r["n_true"])
            r.update(run=sc["run"], layout=sc["layout"], round=int(sc["round"]), seed=seed, added_boxes=added[n],
                     n_robots=len(sc["robots"]))
            rows.append(r)
    return rows


def pct(rows, key="correct", cond=None):
    v = [r[key] if cond is None else cond(r) for r in rows]
    return 100.0 * float(np.mean(v)) if v else math.nan


def seed_stats(rows):
    out = {}
    for name, cond in (("correct", None), ("too_few", lambda r: r["dn"] < 0), ("too_many", lambda r: r["dn"] > 0)):
        per = [pct([r for r in rows if r["seed"] == sd], cond=cond) if cond else
               pct([r for r in rows if r["seed"] == sd]) for sd in sorted({r["seed"] for r in rows})]
        out[f"{name}_pct"] = round(float(np.mean(per)), 2)
        out[f"{name}_pct_sd"] = round(float(np.std(per)), 2)
    rm = np.array([r["rmse"] for r in rows], float)
    rm = rm[np.isfinite(rm)]
    out["rmse_median_cm"] = round(100 * float(np.median(rm)), 2) if len(rm) else math.nan
    out["scenes"] = len({(r["layout"], r["round"]) for r in rows})
    return out


def ray_diagnostics(scenes, rays_raw, p):
    """Per robot and stop: true boxes in view (within the field of view and 3 m), found by a marker of that
    robot (within 30 cm), found by a YOLO ray (passing within the box circumradius at a plausible range),
    and YOLO rays on no box."""
    rows = []
    for s, raw in zip(scenes, rays_raw):
        rays = scene_rays(raw, p)
        for qi, q in enumerate(s["robots"]):
            x, y, yaw = s["poses"][qi]
            mine = [r for r in rays if r["robot"] == qi]
            dets = s["pos"][s["det_robot"] == qi]
            seen = found_m = found_y = 0
            for b in s["truth"]:
                v = b - [x, y]
                dist = float(np.linalg.norm(v))
                if dist > MAX_RANGE_M or abs(hybrid.wrap(math.atan2(v[1], v[0]) - yaw)) > FOV_HALF:
                    continue
                seen += 1
                found_m += int(len(dets) > 0 and np.min(np.linalg.norm(dets - b, axis=1)) <= 0.30)
                found_y += int(any(hybrid.along(r, b)[1] <= 0.136 and hybrid.along(r, b)[0] > 0 for r in mine))
            false = sum(1 for r in mine
                        if all(hybrid.along(r, b)[1] > 0.20 or hybrid.along(r, b)[0] <= 0 for b in s["truth"]))
            rows.append(dict(layout=s["layout"], round=int(s["round"]), robot=q, boxes_in_view=seen,
                             found_marker=found_m, found_yolo=found_y, yolo_rays=len(mine), yolo_false_rays=false))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", default="data/survey_all/")
    ap.add_argument("--runs", nargs="+", default=["data/survey_all/*"])
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--gnn-dir", default="results/plan_v2/M4/gnn/detection_full_full_h2")
    ap.add_argument("--cache", default="data/yolo_real/cache")
    ap.add_argument("--levels", default="results/scene_levels.csv", help="groups of layouts, for the two halves")
    ap.add_argument("--calibrate-scope", choices=["other-layouts", "same-run"], default="other-layouts",
                    help="camera pitch and yaw offset of a layout from the markers of the other layouts or its own")
    ap.add_argument("--mht", default=None, help="scenes.csv of pipeline/evaluate_runs.py: paired test against mht_sensor")
    ap.add_argument("--out", default="results/yolo_gnn/")
    a_cli = ap.parse_args()
    os.makedirs(a_cli.out, exist_ok=True)

    a, models = load_models(a_cli.gnn_dir)
    runs = [(d, *load_run(d)) for d in run_dirs(a_cli.runs)]
    cal = calibration(runs, dict(PITCH_DEG), a_cli.calibrate_scope)
    scenes = all_scenes(a_cli.survey_dir, a_cli.ground_truth, a_cli.sensor_calib, runs)
    fold = folds(scenes, a_cli.levels)
    print(f"{len(scenes)} scenes, {len(fold)} layouts ({sum(fold.values())} in half 1), {len(models)} seeds, "
          f"camera calibration {a_cli.calibrate_scope}", flush=True)

    p0 = Params()
    rays_raw = reading_rays(scenes, runs, a_cli.cache, cal, p0)
    write_csv(os.path.join(a_cli.out, "rays.csv"), ray_diagnostics(scenes, rays_raw, p0))

    base_enc = [encode(s, a) if len(s["pos"]) else None for s in scenes]
    idx = [n for n, e in enumerate(base_enc) if e is not None]
    base_pred = []
    for m in models:
        pr = dict(zip(idx, predict(m, [base_enc[n] for n in idx])))
        base_pred.append([decode(s["pos"], base_enc[n]["pairs"], *pr[n]) if n in pr else ([], None)
                          for n, s in enumerate(scenes)])

    cache_rows, grid_rows, cv_params, summary = {}, [], {}, []
    for variant, spec in VARIANTS.items():
        combos = [dict(zip(spec["tune"], vals)) for vals in itertools.product(*[GRID[k] for k in spec["tune"]])]
        results = []
        for c in combos:
            p = replace(p0, **spec["fixed"], **c)
            key = json.dumps(asdict(p), sort_keys=True)
            if key not in cache_rows:
                cache_rows[key] = run(scenes, rays_raw, base_enc, base_pred, models, a, p)
            rows = cache_rows[key]
            score = [pct([r for r in rows if fold[r["layout"]] == h]) for h in (0, 1)]
            results.append((c, rows, score))
            grid_rows.append(dict(variant=variant, **c, half0_correct_pct=round(score[0], 2),
                                  half1_correct_pct=round(score[1], 2)))
        chosen = []
        for h in (0, 1):                       # thresholds for half h: the best on the other half
            best = max(range(len(results)), key=lambda i: (results[i][2][1 - h], -i))
            c, rows, _ = results[best]
            cv_params.setdefault(variant, {})[f"half{h}"] = c
            chosen += [dict(r, half=h) for r in rows if fold[r["layout"]] == h]
        chosen.sort(key=lambda r: (r["layout"], r["round"], r["seed"]))
        write_csv(os.path.join(a_cli.out, f"test_scenes_{variant}.csv"), chosen)
        for part, rows in (("all", chosen), ("half0", [r for r in chosen if r["half"] == 0]),
                           ("half1", [r for r in chosen if r["half"] == 1])):
            summary.append(dict(variant=variant, scenes_set=part, **seed_stats(rows),
                                added_boxes_mean=round(float(np.mean([r["added_boxes"] for r in rows])), 3)))
        st = seed_stats(chosen)
        print(f"  {variant:<12} correct {st['correct_pct']:5.1f} ± {st['correct_pct_sd']:.1f}%  "
              f"too few {st['too_few_pct']:4.1f}%  too many {st['too_many_pct']:4.1f}%  "
              f"RMSE {st['rmse_median_cm']:.1f} cm   thresholds {cv_params[variant]}", flush=True)

        if a_cli.mht and variant in ("marker", "full"):
            base = read_baseline(a_cli.mht).get("mht_sensor", {})
            for sd in sorted({r["seed"] for r in chosen}):
                b = c_ = 0
                for r in (r for r in chosen if r["seed"] == sd):
                    rb = base.get((r["run"], r["round"])) or base.get(("layout", r["layout"], r["round"]))
                    mht_ok = rb["correct"] if rb else 0
                    b += int(r["correct"] and not mht_ok)
                    c_ += int(mht_ok and not r["correct"])
                print(f"    seed {sd}: {variant} right and MHT wrong {b}, MHT right and {variant} wrong {c_}, "
                      f"McNemar p = {mcnemar(b, c_):.2g}")

    write_csv(os.path.join(a_cli.out, "summary.csv"), summary)
    write_csv(os.path.join(a_cli.out, "grid.csv"), grid_rows)
    with open(os.path.join(a_cli.out, "cv_params.json"), "w") as f:
        json.dump(dict(cv_params, fixed=asdict(p0), calibrate_scope=a_cli.calibrate_scope,
                       halves={k: v for k, v in sorted(fold.items())}), f, indent=2)
    diag = list(csv.DictReader(open(os.path.join(a_cli.out, "rays.csv"))))
    seen = sum(int(r["boxes_in_view"]) for r in diag)
    if seen:
        print(f"boxes in view: {seen}; found by the markers {100 * sum(int(r['found_marker']) for r in diag) / seen:.1f}%, "
              f"by the YOLO rays {100 * sum(int(r['found_yolo']) for r in diag) / seen:.1f}%; "
              f"YOLO rays on no box {sum(int(r['yolo_false_rays']) for r in diag)} of "
              f"{sum(int(r['yolo_rays']) for r in diag)}")
    print(f"written in {a_cli.out}")


if __name__ == "__main__":
    main()
