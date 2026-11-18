"""Model C without the ArUco reports, on the 444 real scenes: what the YOLO rays alone give to the marker network.

Every scene loses its marker reports. The boxes come only from the injection (two unexplained rays of different
robots crossing at the expected range give report nodes) and are clustered by the same marker network, with the
cannot-link. The thresholds of the rays (min_frac, tol_sym, cl_robots) are chosen as in yolo_gnn/evaluate.py: by
two-fold cross-validation over the layouts, on the correct scenes of the other half.

The markers are still used once, before any scene: the camera pitch and yaw offsets come from the marker reports of
the OTHER layouts (yolo_gnn/camera.calibration, as in model C). A system with no markers at all would need them
measured another way.

    python -m yolo_gnn.segment_real --runs "data/survey_all/*" --weights results/yolo_seg/box_seg/weights/best.pt
    python -m tools.c_without_markers --out results/yolo_gnn_no_markers/
"""

import argparse
import itertools
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from gnn.model import decode
from gnn.train_sim import predict
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics
from yolo_gnn import hybrid
from yolo_gnn.camera import PITCH_DEG, calibration, load_run, run_dirs
from yolo_gnn.evaluate import all_scenes, encode, folds, load_models, reading_rays, scene_rays
from yolo_gnn.hybrid import Params

GRID = dict(min_frac=[0.8, 0.6], tol_sym=[0.18, 0.12, 0.25], cl_robots=[2, 1])


def strip(s):
    """The scene with no marker report (robots, poses and truth kept)."""
    return dict(s, det_robot=np.zeros(0, int), distance=np.zeros(0), bearing=np.zeros(0), view=np.zeros(0),
                pos=np.zeros((0, 2)), label=np.zeros(0, int))


def run(scenes, raw, models, a, p):
    rows = []
    rays = [scene_rays(r, p) for r in raw]
    for seed, m in enumerate(models):
        for s, rs in zip(scenes, rays):
            sc, added = hybrid.inject(s, rs, [], p) if rs else (s, 0)
            est = []
            if len(sc["pos"]):
                enc = encode(sc, a)
                ps, pf = predict(m, [enc])[0]
                if p.cannot_link:
                    ps = hybrid.cannot_link(sc, enc["pairs"], ps, rs, p)
                est, _ = decode(sc["pos"], enc["pairs"], ps, pf)
            r = scene_metrics(est, list(sc["truth"]), MATCH_CUTOFF_M, 2.0)
            r.update(correct=int(r["dn"] == 0 and r["n_matched"] == r["n_true"]), layout=s["layout"],
                     round=int(s["round"]), seed=seed, injected_boxes=added, rays=len(rs))
            rows.append(r)
    return pd.DataFrame(rows)


def summary(d):
    f1 = 2 * d.n_matched / (d.n_true + d.n_est).clip(lower=1)
    return dict(correct=100 * d.correct.mean(), too_few=100 * (d.dn < 0).mean(), too_many=100 * (d.dn > 0).mean(),
                f1=f1.mean(), boxes_found=100 * d.n_matched.sum() / d.n_true.sum(),
                false_per_scene=(d.n_est - d.n_matched).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", default="data/survey_all/")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--gnn-dir", default="results/plan_v2/M4/gnn/detection_full_full_h2")
    ap.add_argument("--cache", default="data/yolo_real/cache")
    ap.add_argument("--cv", default="results/yolo_gnn/cv_params.json")
    ap.add_argument("--levels", default="results/scene_levels.csv")
    ap.add_argument("--out", default="results/yolo_gnn_no_markers/")
    o = ap.parse_args()
    os.makedirs(o.out, exist_ok=True)

    a, models = load_models(o.gnn_dir)
    runs = [(d, *load_run(d)) for d in run_dirs([os.path.join(o.survey_dir, "*")])]
    cal = calibration(runs, dict(PITCH_DEG), "other-layouts")
    scenes = [strip(s) for s in all_scenes(o.survey_dir, o.ground_truth, o.sensor_calib, runs)]
    fold = folds(scenes, o.levels)
    cv = json.load(open(o.cv))
    p0 = replace(Params(**{k: v for k, v in cv["fixed"].items() if k in Params.__dataclass_fields__}),
                 inject=True, cannot_link=True, same_robot=False)
    raw = reading_rays(scenes, runs, o.cache, cal, p0)
    print(f"{len(scenes)} scenes, {len(models)} seeds", flush=True)

    results = []
    for vals in itertools.product(*GRID.values()):
        c = dict(zip(GRID, vals))
        d = run(scenes, raw, models, a, replace(p0, **c))
        d["half"] = d.layout.map(fold)
        score = [100 * d[d.half == h].correct.mean() for h in (0, 1)]
        results.append((c, d, score))
        print(f"  {c}: correct half0 {score[0]:.1f}%, half1 {score[1]:.1f}%", flush=True)
    parts, chosen = [], {}
    for h in (0, 1):
        c, d, _ = max(results, key=lambda r: r[2][1 - h])
        chosen[f"half{h}"] = c
        parts.append(d[d.half == h])
    d = pd.concat(parts).sort_values(["layout", "round", "seed"])
    d.to_csv(os.path.join(o.out, "test_scenes_no_markers.csv"), index=False)
    json.dump(chosen, open(os.path.join(o.out, "chosen.json"), "w"), indent=2)

    level = pd.read_csv(o.levels).set_index(["layout", "round"])["level"]
    m = pd.read_csv("results/evaluation_all/scenes.csv")
    m = m[m.method == "mht_sensor"].copy()
    m["correct"] = ((m.dn == 0) & (m.n_matched == m.n_true)).astype(int)
    others = {"MHT": m.assign(seed=0), "model A": pd.read_csv("results/yolo_gnn/test_scenes_marker.csv"),
              "model C": pd.read_csv("results/yolo_gnn/test_scenes_full.csv"), "C without markers": d}
    rows = []
    for name, df in others.items():
        df = df.join(level, on=["layout", "round"])
        for lv in ("all", "easy", "medium", "hard"):
            part = df if lv == "all" else df[df.level == lv]
            per_seed = [summary(part[part.seed == s]) for s in sorted(part.seed.unique())]
            mean = {k: float(np.mean([x[k] for x in per_seed])) for k in per_seed[0]}
            sd = float(np.std([x["correct"] for x in per_seed]))
            rows.append(dict(method=name, level=lv, scenes=part.groupby(["layout", "round"]).ngroups,
                             correct_sd=sd, **mean))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(o.out, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    print("thresholds:", chosen)


if __name__ == "__main__":
    main()
