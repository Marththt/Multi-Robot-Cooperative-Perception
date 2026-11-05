"""Module M4 of the research plan: is there room for switching from MHT to the GNN region by region?

For every test set, the boxes of MHT (pipeline.plan_mht, estimates_H<H>_o0.jsonl) and of the GNN
(gnn.train_sim --save-estimates, estimates/<set>_seed<k>.jsonl) are split over the regions of the scene
(pipeline/plan_regions.py). Per region each method is right or wrong; the ORACLE SELECTOR takes MHT's boxes
in a region if they are right, otherwise the GNN's if they are right, otherwise MHT's. It uses the ground truth,
so it is not a method: it is the upper bound of any gate that switches regions between MHT and the GNN.

Outputs in --out:
  table_M4.csv   per set: correct scenes of MHT, GNN and oracle (GNN and oracle: mean and sd over the GNN seeds),
                 the oracle's gain over MHT and over the better of the two, the share of scenes in which every
                 box was reported by some robot (the ceiling of every method), and the 2 x 2 count of regions
                 (MHT right / wrong x GNN right / wrong, % of the regions)
  fig_M4_<ID>.png  correct scenes against the level of the scenario, MHT, GNN, oracle and the ceiling, both
                 families
and prints the decision of the plan: go on to the gate (M5) if the oracle is at least --decision points above
MHT in at least one level of a special case.

    python -m pipeline.plan_oracle --sets "G0_*" "E1_*" "E3_*" "S1_*" --mht results/plan/M3/ \\
        --gnn results/plan/M4/gnn/detection_full_full_h2/ --out results/plan/M4/
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from gnn.dataset import build_scenes
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics
from pipeline.plan_regions import LINK_M, region_results
from pipeline.plan_failure_map import LEVEL_VALUE, X_LABEL

EMPTY = dict(boxes=[], members=[])


def load(path):
    with open(path) as f:
        return {e["run"]: e for e in map(json.loads, f)}


def correct_of(boxes, truth, cutoff):
    m = scene_metrics([np.asarray(b, float) for b in boxes], list(truth), cutoff, 2.0)
    return int(m["dn"] == 0 and m["n_matched"] == m["n_true"])


def evaluate_set(scenes, mht, gnn, link, cutoff):
    """Scene-level correct of MHT, GNN and oracle, and the 2 x 2 region counts, for one GNN seed."""
    out = dict(mht=[], gnn=[], oracle=[], all_reported=[])
    cells = np.zeros((2, 2))                       # [MHT right?][GNN right?]
    for s in scenes:
        em, eg = mht.get(s["run"], EMPTY), gnn.get(s["run"], EMPTY)
        _, rm = region_results(s, em, link, cutoff)
        _, rg = region_results(s, eg, link, cutoff)
        chosen = []
        for a, b in zip(rm, rg):
            cells[1 - a["correct"], 1 - b["correct"]] += 1
            if a["correct"] or not b["correct"]:
                chosen += [em["boxes"][e] for e in a["est"]]
            else:
                chosen += [eg["boxes"][e] for e in b["est"]]
        out["mht"].append(correct_of(em["boxes"], s["truth"], cutoff))
        out["gnn"].append(correct_of(eg["boxes"], s["truth"], cutoff))
        out["oracle"].append(correct_of(chosen, s["truth"], cutoff))
        reported = {int(l) for l in s["label"] if l >= 0}
        out["all_reported"].append(int(len(reported) == len(s["truth"])))
    return {k: 100 * float(np.mean(v)) for k, v in out.items()}, 100 * cells / max(cells.sum(), 1)


def figure(sid, rows, out, plt):
    rs = [r for r in rows if r["scenario"] == sid]
    if not rs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, family in zip(ax, ("circle", "random")):
        fr = sorted((r for r in rs if r["family"] == family), key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in fr]
        a.plot(x, [r["mht_correct"] for r in fr], "o-", label="MHT")
        a.errorbar(x, [r["gnn_correct"] for r in fr], [r["gnn_correct_sd"] for r in fr], fmt="s-", label="GNN")
        a.plot(x, [r["oracle_correct"] for r in fr], "d--", color="gray", label="oracle selector")
        a.plot(x, [r["all_boxes_reported"] for r in fr], ":", color="k", label="every box reported (ceiling)")
        a.set(title=f"{sid}, {family} family", xlabel=X_LABEL.get(sid, "level"), ylim=(0, 102))
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("correct scenes [%]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_M4_{sid}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="set names or patterns in --data (E1_*, ...)")
    ap.add_argument("--data", default="data/plan/")
    ap.add_argument("--mht", default="results/plan/M3/", help="pipeline.plan_mht output with one folder per set")
    ap.add_argument("--mht-h", type=int, default=100)
    ap.add_argument("--gnn", required=True, help="gnn.train_sim variant folder (with estimates/)")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--link", type=float, default=LINK_M)
    ap.add_argument("--cutoff", type=float, default=MATCH_CUTOFF_M)
    ap.add_argument("--decision", type=float, default=2.0, help="oracle gain over MHT needed to build the gate")
    ap.add_argument("--out", default="results/plan/M4/")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({os.path.basename(os.path.normpath(d)) for p in a.sets for d in glob.glob(os.path.join(a.data, p))
                    if os.path.isdir(d)})
    if not names:
        raise SystemExit(f"no set of {a.data} matches {a.sets}")
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for name in names:
        sid, level, family = name.split("_")
        mht_path = os.path.join(a.mht, name, f"estimates_H{a.mht_h}_o0.jsonl")
        gnn_paths = sorted(glob.glob(os.path.join(a.gnn, "estimates", f"{name}_seed*.jsonl")))
        if not os.path.exists(mht_path) or not gnn_paths:
            print(f"  {name}: missing {'MHT' if not os.path.exists(mht_path) else 'GNN'} estimates, skipped")
            continue
        d = os.path.join(a.data, name)
        scenes = build_scenes(d, os.path.join(d, "ground_truth.json"), a.sensor_calib)
        with open(os.path.join(d, "ground_truth.json")) as f:     # scenes without any report: empty estimate
            truth = json.load(f)
        seen = {s["run"] for s in scenes}
        for run, boxes in truth.items():
            if run.startswith(name + "_") and run not in seen:
                scenes.append(dict(run=run, pos=np.zeros((0, 2)), label=np.zeros(0, int), det_robot=np.zeros(0, int),
                                   truth=np.array([[b["x"], b["y"]] for b in boxes], float)))
        mht = load(mht_path)
        per_seed = [evaluate_set(scenes, mht, load(p), a.link, a.cutoff) for p in gnn_paths]
        res = [r for r, _ in per_seed]
        cells = np.mean([c for _, c in per_seed], axis=0)
        g = [r["gnn"] for r in res]
        o = [r["oracle"] for r in res]
        m = res[0]["mht"]
        row = dict(set=name, scenario=sid, level=level, value=LEVEL_VALUE.get((sid, level), 0), family=family,
                   scenes=len(scenes), seeds=len(res), mht_correct=round(m, 2),
                   gnn_correct=round(float(np.mean(g)), 2), gnn_correct_sd=round(float(np.std(g)), 2),
                   oracle_correct=round(float(np.mean(o)), 2), oracle_correct_sd=round(float(np.std(o)), 2),
                   oracle_gain_over_mht=round(float(np.mean(o)) - m, 2),
                   oracle_gain_over_best=round(float(np.mean(o)) - max(m, float(np.mean(g))), 2),
                   all_boxes_reported=round(res[0]["all_reported"], 2),
                   regions_both_right=round(cells[0, 0], 2), regions_mht_right_gnn_wrong=round(cells[0, 1], 2),
                   regions_mht_wrong_gnn_right=round(cells[1, 0], 2), regions_both_wrong=round(cells[1, 1], 2))
        rows.append(row)
        print(f"  {name:20s} MHT {m:5.1f}  GNN {row['gnn_correct']:5.1f}  oracle {row['oracle_correct']:5.1f} "
              f"(+{row['oracle_gain_over_mht']:.1f} over MHT)  ceiling {row['all_boxes_reported']:5.1f}  "
              f"regions MHT wrong & GNN right {row['regions_mht_wrong_gnn_right']:.2f}%", flush=True)
    if not rows:
        raise SystemExit("nothing to compare")
    with open(os.path.join(a.out, "table_M4.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for sid in sorted({r["scenario"] for r in rows}):
        if sid != "G0":
            figure(sid, rows, a.out, plt)
    special = [r for r in rows if r["scenario"][0] in "ES"]
    passed = [r["set"] for r in special if r["oracle_gain_over_mht"] >= a.decision]
    print(f"decision (oracle >= MHT + {a.decision:g} points in a special case): "
          + (f"yes, go on to the gate ({', '.join(passed)})" if passed else "no: stop and discuss before M5"))


if __name__ == "__main__":
    main()
