"""The marker network with its two decoding thresholds chosen on the tune set, as the MHT birth density is.

yolo_gnn/evaluate_sim.py tunes the MHT (birth density) on the tune set but decodes the marker network with the
fixed thresholds of gnn/model.decode (0.5 for p_false and 0.5 for the average linkage). Here both thresholds of
the marker network are chosen on the SAME tune set by the SAME criterion (mean scene F1), then the test sets are
scored as in evaluate_sim (same scenes, same truth, same metrics; the MHT rows are read from its mht/ folder):

  t_false   a detection is dropped when p_false >= t_false
  t_link    two clusters merge while their average p_same > t_link

The thresholds are chosen once per network on the whole tune set (N 3-50, B 2-50), not per cell of the grid.
Only the marker variant (no YOLO rays needed). Outputs in --out: tune.csv, chosen.json, summary.csv, paired.csv,
test_scenes.csv; tools/fig_grid_f1.py draws the heat maps from summary.csv.

    python -m tools.grid_tune_thresholds --tune-set data/scale/tune --sets "data/scale/G_*" \\
        --gnn plan=results/plan_v2/M4/gnn/detection_full_full_h2 --gnn scale=results/scale/gnn/detection_full_full_h2 \\
        --mht-dir results/scale_grid/mht --out results/scale_grid_tuned/
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from gnn.train_sim import mcnemar, predict, write_csv
from pipeline.evaluate_runs import MATCH_CUTOFF_M
from yolo_gnn.evaluate import encode, load_models, scene_metrics
from yolo_gnn.evaluate_sim import f1, full_scenes, summarize, sweep_x

T_FALSE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T_LINK = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def decode2(pos, pairs, p_same, p_false, t_false, t_link):
    """gnn/model.decode with separate thresholds for the false detections and for the linkage."""
    pos = np.asarray(pos, float).reshape(-1, 2)
    n = len(pos)
    P = np.eye(n)
    for (i, j), p in zip(pairs, p_same):
        P[i, j] = P[j, i] = p
    keep = [i for i in range(n) if p_false[i] < t_false]
    if not keep:
        return []
    S = P[np.ix_(keep, keep)].copy()
    sizes = np.ones(len(keep))
    alive = np.ones(len(keep), bool)
    members = [[i] for i in keep]
    np.fill_diagonal(S, -np.inf)
    while alive.sum() > 1:
        link = S / np.outer(sizes, sizes)
        link[~alive, :] = -np.inf
        link[:, ~alive] = -np.inf
        a, c = np.unravel_index(np.argmax(link), link.shape)
        if link[a, c] <= t_link:
            break
        a, c = min(a, c), max(a, c)
        S[a, :] += S[c, :]
        S[:, a] += S[:, c]
        S[a, a] = -np.inf
        sizes[a] += sizes[c]
        alive[c] = False
        members[a] += members[c]
    return [pos[members[k]].mean(0) for k in range(len(keep)) if alive[k]]


def forward(scenes, va, models):
    """Per seed, per scene: (pairs, p_same, p_false) or None (no detection)."""
    enc = [encode(s, va) if len(s["pos"]) else None for s in scenes]
    idx = [n for n, e in enumerate(enc) if e is not None]
    out = []
    for m in models:
        pr = dict(zip(idx, predict(m, [enc[n] for n in idx]))) if idx else {}
        out.append([(enc[n]["pairs"], *pr[n]) if n in pr else None for n in range(len(scenes))])
    return out


def rows_for(scenes, fw, t_false, t_link):
    rows = []
    for seed, per in enumerate(fw):
        for s, f in zip(scenes, per):
            est = [] if f is None else decode2(s["pos"], f[0], f[1], f[2], t_false, t_link)
            r = scene_metrics(est, list(s["truth"]), MATCH_CUTOFF_M, 2.0)
            r["correct"] = int(r["dn"] == 0 and r["n_matched"] == r["n_true"])
            r.update(run=s["run"], layout=s["layout"], round=0, seed=seed, n_robots=len(s["robots"]),
                     seconds=float("nan"))
            r["f1"] = f1(r)
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--tune-set", required=True)
    ap.add_argument("--gnn", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--mht-dir", required=True, help="mht/<set>.csv of yolo_gnn/evaluate_sim.py (same scenes)")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--out", default="results/scale_grid_tuned/")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tests = sorted({d for p in a.sets for d in glob.glob(p) if os.path.exists(os.path.join(d, "geometry.jsonl"))},
                   key=lambda d: tuple(v or 0 for v in sweep_x(os.path.basename(os.path.normpath(d)))))
    gnns = [(n, *load_models(d)) for n, _, d in (s.partition("=") for s in a.gnn)]

    tune = full_scenes(a.tune_set, a.sensor_calib, a.max_scenes)
    tune_rows, chosen = [], {}
    for gname, va, models in gnns:
        fw = forward(tune, va, models)
        best = None
        for tf in T_FALSE:
            for tl in T_LINK:
                rs = rows_for(tune, fw, tf, tl)
                sc = float(np.mean([r["f1"] for r in rs]))
                tune_rows.append(dict(gnn=gname, t_false=tf, t_link=tl, f1_mean=round(sc, 4),
                                      correct_pct=round(100 * float(np.mean([r["correct"] for r in rs])), 2)))
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, tf, tl)
        base = next(r for r in tune_rows if r["gnn"] == gname and r["t_false"] == 0.5 and r["t_link"] == 0.5)
        chosen[gname] = dict(t_false=best[1], t_link=best[2], tune_f1=round(best[0], 4),
                             tune_f1_default=base["f1_mean"])
        print(f"tune {gname}: t_false {best[1]} t_link {best[2]}  F1 {best[0]:.4f} (0.5/0.5: {base['f1_mean']:.4f})",
              flush=True)
    write_csv(os.path.join(a.out, "tune.csv"), tune_rows)
    with open(os.path.join(a.out, "chosen.json"), "w") as f:
        json.dump(dict(chosen, tune_set=a.tune_set, t_false_grid=T_FALSE, t_link_grid=T_LINK), f, indent=2)

    summary, paired, all_rows, mismatch = [], [], [], 0
    for set_dir in tests:
        name = os.path.basename(os.path.normpath(set_dir))
        scenes = full_scenes(set_dir, a.sensor_calib, a.max_scenes)
        meta = dict(set=name, n_robots=round(float(np.mean([len(s["robots"]) for s in scenes])), 1),
                    n_boxes=round(float(np.mean([s["n_boxes"] for s in scenes])), 1),
                    n_covered=round(float(np.mean([len(s["truth"]) for s in scenes])), 1))
        with open(os.path.join(a.mht_dir, f"{name}.csv")) as f:
            mht = {r["run"]: r for r in csv.DictReader(f)}
        for s in scenes:                                   # same scenes as the MHT run: same reports and truth
            r = mht[s["run"]]
            if int(r["n_reports"]) != len(s["pos"]) or int(r["n_true"]) != len(s["truth"]):
                mismatch += 1
        mrows = [dict(r, seed=0, **{k: int(r[k]) for k in ("n_true", "n_est", "n_matched", "correct", "dn")},
                      f1=float(r["f1"])) for r in (mht[s["run"]] for s in scenes)]
        summary.append(dict(meta, method="mht", gnn="", **summarize(mrows, MATCH_CUTOFF_M)))
        line = f"{name:<10} MHT F1 {summary[-1]['f1_mean']:.3f}"
        for gname, va, models in gnns:
            fw = forward(scenes, va, models)
            for tag, tf, tl in (("marker", 0.5, 0.5), ("marker_tuned", chosen[gname]["t_false"],
                                                       chosen[gname]["t_link"])):
                rows = rows_for(scenes, fw, tf, tl)
                summary.append(dict(meta, method=tag, gnn=gname, **summarize(rows, MATCH_CUTOFF_M)))
                all_rows += [dict(r, set=name, method=tag, gnn=gname) for r in rows]
                line += f" | {gname} {tag} F1 {summary[-1]['f1_mean']:.3f}"
                for sd in sorted({r["seed"] for r in rows}):
                    rs = [r for r in rows if r["seed"] == sd]
                    for metric in ("correct", "f1"):
                        b = sum(1 for r in rs if float(r[metric]) > float(mht[r["run"]][metric]) + 1e-9)
                        w = sum(1 for r in rs if float(r[metric]) < float(mht[r["run"]][metric]) - 1e-9)
                        paired.append(dict(set=name, method=tag, gnn=gname, seed=sd, metric=metric,
                                           method_better=b, mht_better=w, p_value=mcnemar(b, w)))
        print(line, flush=True)
        write_csv(os.path.join(a.out, "summary.csv"), summary)
        write_csv(os.path.join(a.out, "paired.csv"), paired)
    write_csv(os.path.join(a.out, "test_scenes.csv"), all_rows)
    print(f"scenes whose reports or truth differ from the MHT run: {mismatch}")
    print(f"written in {a.out}")


if __name__ == "__main__":
    main()
