"""Three real scenes with the reports of every robot, the YOLO rays and the boxes estimated by MHT, model A and
model C, drawn side by side (figure fig_real_examples.png of the thesis).

MHT is run as in the live demo (demo.methods.Classical, the settings of pipeline/evaluate_runs --sensor-calib) on
the round alone; model A and model C as in yolo_gnn/evaluate (first seed of the marker network, camera pitch and
yaw from the other layouts, thresholds chosen on the other half of the layouts). Each panel title gives the verdict
recomputed here; the script prints it next to the verdict of the evaluation CSVs, which must agree.

The YOLO masks of the three runs must be in the cache:
    python -m yolo_gnn.segment_real --runs data/survey_all/L6_* data/survey_all/L9_* data/survey_all/R63_* \\
        --weights results/yolo_seg/box_seg/weights/best.pt --cache data/yolo_real/cache
    python -m tools.fig_review_examples --out ../fig_real_examples.png
"""

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demo.methods import Classical, judge
from gnn.model import decode
from gnn.train_sim import predict
from yolo_gnn import hybrid
from yolo_gnn.camera import PITCH_DEG, calibration, load_run, run_dirs
from yolo_gnn.evaluate import all_scenes, encode, load_models, reading_rays, scene_rays
from yolo_gnn.hybrid import Params

SCENES = [("L6", 2, "a. an isolated false report"), ("L9", 0, "b. two boxes almost touching"),
          ("R63", 2, "c. a box with no marker read")]
ROBOT_C = "#2867d6"
COLORS = {"MHT": "#2F5FA8", "model A": "#C06A1E", "model C": "#1E2D4F"}


def round_dir(run, rnd, tmp):
    """The round of a run as round 0 of a new folder (what demo.methods.Classical reads)."""
    out = os.path.join(tmp, f"{os.path.basename(run)}_r{rnd}")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(run):
        if f.endswith(".jsonl"):
            rows = [json.loads(line) for line in open(os.path.join(run, f)) if line.strip()]
            rows = [dict(r, round=0) for r in rows if int(r.get("round", 0)) == rnd]
            with open(os.path.join(out, f), "w") as g:
                g.write("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy(os.path.join(run, "ground_truth.json"), out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", default="data/survey_all/")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--sensor-model", default="config/sensor_model.json")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--gnn-dir", default="results/plan_v2/M4/gnn/detection_full_full_h2")
    ap.add_argument("--cache", default="data/yolo_real/cache")
    ap.add_argument("--cv", default="results/yolo_gnn/cv_params.json")
    ap.add_argument("--out", default="results/figures/fig_real_examples.png")
    a_cli = ap.parse_args()

    a, models = load_models(a_cli.gnn_dir)
    model = models[0]
    runs = [(d, *load_run(d)) for d in run_dirs([os.path.join(a_cli.survey_dir, "*")])]
    cal = calibration(runs, dict(PITCH_DEG), "other-layouts")
    wanted = {lay for lay, _r, _t in SCENES}
    runs_w = [r for r in runs if r[1] in wanted]
    scenes = [s for s in all_scenes(a_cli.survey_dir, a_cli.ground_truth, a_cli.sensor_calib, runs_w)
              if (s["layout"], int(s["round"])) in {(l, r) for l, r, _ in SCENES}]
    cv = json.load(open(a_cli.cv))
    classical = Classical(a_cli.sensor_calib, a_cli.sensor_model, a_cli.regions)
    run_of = {r[1]: r[0] for r in runs_w}
    levels = pd.read_csv("results/scene_levels.csv").set_index(["layout", "round"])["level"]

    ref = {}
    mht_csv = pd.read_csv("results/evaluation_all/scenes.csv")
    mht_csv = mht_csv[mht_csv.method == "mht_sensor"]
    for name, df in (("MHT", mht_csv), ("model A", pd.read_csv("results/yolo_gnn/test_scenes_marker.csv")),
                     ("model C", pd.read_csv("results/yolo_gnn/test_scenes_full.csv"))):
        if "seed" in df:
            df = df[df.seed == 0]
        for _, r in df.iterrows():
            ref[(name, r.layout, int(r["round"]))] = (int(r.n_est), int(r.n_matched), int(r.n_true))

    fig, axes = plt.subplots(len(SCENES), 3, figsize=(12, 4.1 * len(SCENES)))
    tmp = tempfile.mkdtemp()
    try:
        for row, (lay, rnd, title) in enumerate(SCENES):
            s = next(x for x in scenes if x["layout"] == lay and int(x["round"]) == rnd)
            p = replace(Params(**{k: v for k, v in cv["fixed"].items() if k in Params.__dataclass_fields__}),
                        **cv["full"][f"half{cv['halves'][lay]}"])
            raw = reading_rays([s], runs_w, a_cli.cache, cal, p)[0]
            rays = scene_rays(raw, p)
            enc = encode(s, a) if len(s["pos"]) else None
            est_a = []
            if enc is not None:
                ps, pf = predict(model, [enc])[0]
                est_a, _ = decode(s["pos"], enc["pairs"], ps, pf)
            sc, added = hybrid.inject(s, rays, est_a, p) if rays else (s, 0)
            enc_c = encode(sc, a) if len(sc["pos"]) else None
            est_c = []
            if enc_c is not None:
                ps, pf = predict(model, [enc_c])[0]
                ps = hybrid.cannot_link(sc, enc_c["pairs"], ps, rays, p)
                est_c, _ = decode(sc["pos"], enc_c["pairs"], ps, pf)
            out, _dets = classical.run(round_dir(run_of[lay], rnd, tmp))
            est = {"MHT": out["mht_sensor"], "model A": [list(e) for e in est_a], "model C": [list(e) for e in est_c]}
            truth = [list(t) for t in s["truth"]]
            c = np.mean(truth, axis=0)
            for col, name in enumerate(["MHT", "model A", "model C"]):
                ax = axes[row, col]
                j = judge(est[name], truth)
                print(f"{lay} r{rnd} {name}: here {j['n_est']}/{j['n_matched']}/{j['n_true']} {j['verdict']}, "
                      f"evaluation {ref.get((name, lay, rnd))}")
                for qi, q in enumerate(s["robots"]):
                    x, y, yaw = s["poses"][qi]
                    ax.plot(x, y, "s", ms=9, mfc="#D9E3F1", mec=ROBOT_C)
                    ax.plot([x, x + 0.25 * math.cos(yaw)], [y, y + 0.25 * math.sin(yaw)], color=ROBOT_C, lw=1.5)
                    ax.annotate(q, (x, y), xytext=(4, -12), textcoords="offset points", fontsize=7, color=ROBOT_C)
                if name == "model C":
                    for r in rays:
                        o, ang = r["origin"], r["ang"]
                        L = min(float(r["d"]) + 0.5, 3.0)
                        ax.plot([o[0], o[0] + L * math.cos(ang)], [o[1], o[1] + L * math.sin(ang)],
                                color="#E07A1F", lw=0.8, alpha=0.6)
                if len(sc["pos"] if name == "model C" else s["pos"]):
                    pos = sc["pos"] if name == "model C" else s["pos"]
                    ax.plot(pos[:, 0], pos[:, 1], ".", color="#6B7683", ms=5)
                for t, ok in zip(truth, j["truth_ok"]):
                    ax.add_patch(plt.Circle(t, 0.136, fill=False, ec="#2E7A4F" if ok else "#B23A2E", lw=1.6,
                                            ls="-" if ok else "--"))
                for e, ok in zip(est[name], j["est_ok"]):
                    ax.plot(e[0], e[1], "x", color=COLORS[name] if ok else "#B23A2E", ms=9, mew=2.2)
                ax.set_xlim(c[0] - 1.9, c[0] + 1.9)
                ax.set_ylim(c[1] - 1.9, c[1] + 1.9)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                verdict = j["verdict"] if j["verdict"] != "too few" else "too few boxes"
                verdict = "too many boxes" if verdict == "too many" else verdict
                ax.set_title(f"{name}: {j['n_est']} of {j['n_true']}, {verdict}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{title}\n({lay}, round {rnd + 1}, {levels[(lay, rnd)]} scene)", fontsize=10)
        handles = [plt.Line2D([], [], marker="s", ls="", mfc="#D9E3F1", mec=ROBOT_C, label="robot (camera, heading)"),
                   plt.Line2D([], [], marker=".", ls="", color="#6B7683", label="report (marker, or injected)"),
                   plt.Line2D([], [], color="#E07A1F", label="YOLO ray (model C)"),
                   plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#2E7A4F", label="true box, found"),
                   plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#B23A2E", label="true box, missed"),
                   plt.Line2D([], [], marker="x", ls="", color="#1E2D4F", mew=2, label="estimate within 15 cm"),
                   plt.Line2D([], [], marker="x", ls="", color="#B23A2E", mew=2, label="estimate matched to no box")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        os.makedirs(os.path.dirname(os.path.abspath(a_cli.out)), exist_ok=True)
        fig.savefig(a_cli.out, dpi=200)
        print("figure:", a_cli.out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
