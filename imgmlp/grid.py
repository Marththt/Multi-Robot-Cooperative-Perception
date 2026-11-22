"""Trained imgmlp variants on the robots x boxes grid (sim/scaling.py, sets G_n<N>_b<B>, N and B in 2, 4, 8, 16,
32, 50): the same scenes and the same scoring as yolo_gnn/evaluate_sim.py (every robot of the scene, the covered
boxes as truth, the simulated detections of the set corrected with --sensor-calib). Nothing is trained here.

Every variant is tested on the same detections, whatever measurements it was trained with, so a noisyrand model
meets here a sensor it never saw. With k > 0 the frames of the grid come from the cache --cache-root/<set>
(python -m imgmlp.frames sim --set data/scale/<set> --out data/imgmlp/grid/<set>). With --unc the uncertainty
declared on the grid is --declare FIELD=VALUE ... (one setting).

    python -m imgmlp.grid --variants results/imgmlp/measure/sensor_fixed_k0_64x36 results/imgmlp/rand/* \\
        --sets "data/scale/G_*" --cache-root data/imgmlp/grid --out results/imgmlp/grid/

Output: summary.csv (one row per set and variant: correct scenes and scene F1, mean and sd over the seeds, as
tools/fig_grid_f1.py reads them; method = variant name), scenes.csv (one row per scene and seed).
"""

import argparse
import csv
import glob
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from imgmlp.frames import load_cache
from imgmlp.measure import attach_frames
from imgmlp.model import ImageDetectionGNN
from imgmlp.train import declared_model, encode_all, evaluate, write_csv


def f1(r):
    den = int(r["n_true"]) + int(r["n_est"])
    return 1.0 if den == 0 else 2.0 * int(r["n_matched"]) / den


def load_variant(folder):
    with open(os.path.join(folder, "variant.json")) as f:
        cfg = json.load(f)
    a = SimpleNamespace(**cfg)
    a.unc = getattr(a, "unc", False)
    models = []
    for path in sorted(glob.glob(os.path.join(folder, "model_seed*.pt"))):
        m = ImageDetectionGNN(a.k, a.img_size, a.geometry, a.hidden, a.hops, graph=a.graph, unc=a.unc)
        m.load_state_dict(torch.load(path))
        models.append(m)
    return a, models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True, help="variant folders of imgmlp/train.py")
    ap.add_argument("--sets", default="data/scale/G_*")
    ap.add_argument("--cache-root", default="data/imgmlp/grid", help="frame caches of the sets (k > 0)")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--declare", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="uncertainty declared on the grid for --unc variants (default: NoiseModel defaults)")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--out", default="results/imgmlp/grid/")
    a = ap.parse_args()
    from yolo_gnn.evaluate_sim import full_scenes

    sets = sorted(d for d in glob.glob(a.sets) if os.path.exists(os.path.join(d, "geometry.jsonl")))
    if not sets:
        raise SystemExit(f"no set in {a.sets} (python -m sim.scaling --sets G --test-runs 100 --out data/scale/)")
    variants = [v for v in a.variants if os.path.exists(os.path.join(v, "variant.json"))]
    skipped = sorted(set(a.variants) - set(variants))
    if skipped:
        print(f"skipped (no variant.json): {' '.join(skipped)}")
    loaded = [(os.path.basename(os.path.normpath(v)), *load_variant(v)) for v in variants]
    declared = declared_model(a.declare)
    os.makedirs(a.out, exist_ok=True)
    summary, scenes_rows = [], []
    for set_dir in sets:
        name = os.path.basename(os.path.normpath(set_dir))
        t0 = time.time()
        scenes = full_scenes(set_dir, a.sensor_calib, a.max_scenes)
        for s in scenes:
            s["declared"] = declared
        cache = {}
        line = f"{name:<12} {len(scenes):4d} scenes"
        for vname, va, models in loaded:
            images = None
            if va.k > 0:
                key = tuple(va.img_size)
                if key not in cache:
                    folder = os.path.join(a.cache_root, name)
                    if not os.path.exists(os.path.join(folder, "frames.npy")):
                        print(f"  {vname}: no frame cache {folder}, skipped")
                        continue
                    images, index, _ = load_cache(folder, va.img_size)
                    missing = attach_frames(scenes, index)
                    if missing:
                        print(f"  warning: {missing} robots of {name} have no frame (black image)")
                    cache[key] = images
                images = cache[key]
            enc = encode_all(scenes, va)
            correct, f1s = [], []
            for seed, m in enumerate(models):
                rows = evaluate(m, scenes, enc, images, va)
                for r in rows:
                    r.update(set=name, method=vname, seed=seed, f1=round(f1(r), 4))
                scenes_rows += rows
                correct.append(100 * float(np.mean([r["correct"] for r in rows])))
                f1s.append(float(np.mean([r["f1"] for r in rows])))
            summary.append(dict(set=name, method=vname, gnn="", scenes=len(scenes),
                                n_robots=round(float(np.mean([len(s["robots"]) for s in scenes])), 1),
                                n_covered=round(float(np.mean([len(s["truth"]) for s in scenes])), 1),
                                correct_pct=round(float(np.mean(correct)), 2),
                                correct_pct_sd=round(float(np.std(correct)), 2),
                                f1_mean=round(float(np.mean(f1s)), 4), f1_mean_sd=round(float(np.std(f1s)), 4)))
            line += f" | {vname} {summary[-1]['correct_pct']:5.1f}% F1 {summary[-1]['f1_mean']:.3f}"
        print(line + f"  ({time.time() - t0:.0f} s)", flush=True)
        write_csv(os.path.join(a.out, "summary.csv"), summary)
    write_csv(os.path.join(a.out, "scenes.csv"), scenes_rows)
    print(f"written in {a.out}")


if __name__ == "__main__":
    main()
