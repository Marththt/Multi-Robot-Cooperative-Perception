"""GNN stage 1: leave-one-layout-out training and evaluation on the real runs.

For every layout L (L4 ... L11) a model is trained on the scenes of all the other layouts and tested on
the 7 scenes of L, with all robots fused. The test scenes are never seen in training, and nothing is
tuned on them: the settings below are fixed before the run.

Training data augmentation (training scenes only):
  rotation        the whole scene is rotated by a random angle about the ring centre
  robot dropout   with probability --robot-dropout one robot is removed from the scene
  det dropout     every detection is removed with probability --det-dropout (at least one is kept)

The metrics are those of evaluate_runs.py (same matching, RMSE, count error, OSPA, cut-off 0.5 m), so
the GNN rows can be compared directly with the classical methods. With --baseline the scenes.csv written
by evaluate_runs.py is read and the classical methods are printed next to the GNN.

    python gnn_train.py --survey-dir survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --sensor-calib config/sensor_calib.json --baseline evaluation_calib/scenes.csv --out evaluation_gnn/

Outputs (in --out): gnn_scenes.csv (one row per test scene), gnn_summary.csv (overall and per layout),
comparison.csv (GNN and classical methods per layout, with --baseline), model_<layout>_<seed>.pt.
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from evaluate_runs import scene_metrics, summarize, write_csv
from gnn_dataset import build_scenes, subset_robots
from gnn_model import RobotGNN, collate, decode, encode_scene, loss_fn

METHOD_ORDER = ["zone_class", "union", "phd", "cphd", "mht", "gnn"]
LABELS = dict(zone_class="Zone+class", union="Union", phd="PHD", cphd="CPHD", mht="MHT", gnn="GNN")


def layout_key(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 0, name)


def augment(scene, rng, robot_dropout, det_dropout):
    s = scene
    n_rob = len(s["robots"])
    if n_rob > 2 and rng.random() < robot_dropout:
        drop = int(rng.integers(n_rob))
        s = subset_robots(s, [i for i in range(n_rob) if i != drop])
    if det_dropout > 0 and len(s["pos"]) > 1:
        keep = rng.random(len(s["pos"])) >= det_dropout
        if not keep.any():
            keep[int(rng.integers(len(keep)))] = True
        if not keep.all():
            s = dict(s)
            for key in ("det_robot", "distance", "bearing", "view", "label"):
                s[key] = s[key][keep]
            s["pos"] = s["pos"][keep]
    return encode_scene(s, rotation=float(rng.uniform(0, 2 * math.pi)))


def class_weights(scenes):
    """pos_weight for the two heads: negatives / positives in the training scenes (capped at 10)."""
    pos_same = neg_same = pos_false = neg_false = 0
    for s in scenes:
        e = encode_scene(s)
        pos_same += sum(e["same"])
        neg_same += len(e["same"]) - sum(e["same"])
        pos_false += e["false"].sum()
        neg_false += len(e["false"]) - e["false"].sum()
    return (min(10.0, neg_same / max(pos_same, 1)), min(10.0, neg_false / max(pos_false, 1)))


def train(scenes, a, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RobotGNN(hidden=a.hidden, hops=a.hops)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    w_same, w_false = class_weights(scenes)
    model.train()
    for epoch in range(a.epochs):
        order = rng.permutation(len(scenes))
        total = 0.0
        for start in range(0, len(order), a.batch):
            batch = collate([augment(scenes[i], rng, a.robot_dropout, a.det_dropout)
                             for i in order[start:start + a.batch]])
            same, false = model(batch)
            loss = loss_fn(same, false, batch, w_same, w_false)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if a.verbose and (epoch + 1) % 50 == 0:
            print(f"      epoch {epoch + 1}: loss {total:.3f}", flush=True)
    return model


def predict(models, scene):
    """Same-box and false-detection probabilities, averaged over the models (seeds)."""
    e = encode_scene(scene)
    b = collate([e])
    p_same, p_false = 0.0, 0.0
    with torch.no_grad():
        for m in models:
            m.eval()
            s, f = m(b)
            p_same = p_same + torch.sigmoid(s).numpy()
            p_false = p_false + torch.sigmoid(f).numpy()
    return e["pairs"], np.atleast_1d(p_same / len(models)), np.atleast_1d(p_false / len(models))


def pair_accuracy(scene, pairs, p_same, p_false):
    lab = scene["label"]
    ok = [(p >= 0.5) == (lab[i] >= 0 and lab[i] == lab[j]) for (i, j), p in zip(pairs, p_same)]
    ok_false = [(p >= 0.5) == (l < 0) for p, l in zip(p_false, lab)]
    return ok, ok_false


def read_baseline(path):
    """All-robot rows of evaluate_runs.py scenes.csv."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["k"]) != int(r["n_robots"]):
                continue
            rows.append(dict(layout=r["layout"], method=r["method"], round=int(r["round"]),
                             dn=int(r["dn"]), rmse=float(r["rmse"]) if r["rmse"] not in ("", "nan") else math.nan))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--sensor-calib", default=None, help="calibrate.py --out (range / bearing fit)")
    ap.add_argument("--baseline", default=None, help="scenes.csv of evaluate_runs.py, for the comparison")
    ap.add_argument("--out", default="evaluation_gnn/")
    ap.add_argument("--layouts", default=None, help="comma-separated test layouts (default: all)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--robot-dropout", type=float, default=0.3)
    ap.add_argument("--det-dropout", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=3, help="models per fold, probabilities averaged")
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--ospa-p", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    scenes = build_scenes(a.survey_dir, a.ground_truth, a.sensor_calib)
    layouts = sorted({s["layout"] for s in scenes}, key=layout_key)
    if a.layouts:
        layouts = [l for l in layouts if l in a.layouts.split(",")]
    print(f"{len(scenes)} scenes, layouts {', '.join(layouts)}; "
          f"{a.epochs} epochs, hidden {a.hidden}, hops {a.hops}, {a.seeds} seed(s) per fold")

    rows, pair_ok, false_ok = [], [], []
    for test in layouts:
        t0 = time.time()
        train_set = [s for s in scenes if s["layout"] != test]
        test_set = [s for s in scenes if s["layout"] == test]
        models = []
        for seed in range(a.seeds):
            model = train(train_set, a, seed)
            torch.save(model.state_dict(), os.path.join(a.out, f"model_{test}_{seed}.pt"))
            models.append(model)
        for s in sorted(test_set, key=lambda q: q["round"]):
            pairs, p_same, p_false = predict(models, s)
            est, _ = decode(s["pos"], pairs, p_same, p_false)
            r = scene_metrics(est, list(s["truth"]), a.cutoff, a.ospa_p)
            r.update(run=s["run"], layout=s["layout"], round=s["round"], k=len(s["robots"]),
                     n_robots=len(s["robots"]), method="gnn", robots=" ".join(s["robots"]))
            rows.append(r)
            ok, okf = pair_accuracy(s, pairs, p_same, p_false)
            pair_ok += ok
            false_ok += okf
        fold = [r for r in rows if r["layout"] == test]
        dns = " ".join("%+d" % r["dn"] for r in fold)
        print(f"  {test}: exact {100 * np.mean([r['dn'] == 0 for r in fold]):.0f}%  dn {dns}  "
              f"RMSE {100 * np.nanmean([r['rmse'] for r in fold]):.1f} cm  ({time.time() - t0:.0f} s)", flush=True)

    write_csv(os.path.join(a.out, "gnn_scenes.csv"), rows)
    summ = summarize(rows, ["gnn"], None) + summarize(rows, ["gnn"], "layout")
    write_csv(os.path.join(a.out, "gnn_summary.csv"), summ)
    print(f"\npair decisions correct: {100 * np.mean(pair_ok):.1f}% of {len(pair_ok)}, "
          f"false-detection decisions correct: {100 * np.mean(false_ok):.1f}% of {len(false_ok)}")

    # comparison table: exact count % / mean RMSE cm, per layout and overall
    table = defaultdict(dict)
    for r in rows:
        table[(r["layout"], "gnn")].setdefault("rows", []).append((r["dn"], r["rmse"]))
    methods = ["gnn"]
    if a.baseline:
        for r in read_baseline(a.baseline):
            if r["layout"] in layouts:
                table[(r["layout"], r["method"])].setdefault("rows", []).append((r["dn"], r["rmse"]))
        methods = [m for m in METHOD_ORDER if any((l, m) in table for l in layouts)]

    def cell(entries):
        if not entries:
            return "-", math.nan, math.nan
        exact = 100 * np.mean([dn == 0 for dn, _ in entries])
        rmse = 100 * np.nanmean([e for _, e in entries])
        return f"{exact:.0f} / {rmse:.1f}", exact, rmse

    print("\nexact count % / mean RMSE cm (all robots fused)")
    print("  " + f"{'layout':<8}" + "".join(f"{LABELS.get(m, m):>14}" for m in methods))
    comp = []
    for l in layouts + ["all"]:
        line = f"  {l:<8}"
        for m in methods:
            entries = ([e for ll in layouts for e in table[(ll, m)].get("rows", [])] if l == "all"
                       else table[(l, m)].get("rows", []))
            text, exact, rmse = cell(entries)
            line += f"{text:>14}"
            comp.append(dict(layout=l, method=m, scenes=len(entries), exact_pct=exact, rmse_mean_cm=rmse))
        print(line)
    write_csv(os.path.join(a.out, "comparison.csv"), comp)
    print(f"\nwritten in {a.out}: gnn_scenes.csv, gnn_summary.csv, comparison.csv, model_*.pt")


if __name__ == "__main__":
    main()