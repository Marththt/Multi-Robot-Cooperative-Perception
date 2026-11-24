"""Train and evaluate one variant of the review (gnn_review/model.py) on the sets of the research plan.

Same data, optimiser, early stop and model selection as the adopted model (gnn/train_sim.py, variant
results/plan_v2/M4/gnn/detection_full_full_h2): train on data/plan_v2/train (50,000 scenes, nominal sensor),
choose the epoch on data/plan_v2/val (2,000 scenes), then evaluate on the validation set and on the test sets
of the plan. Thresholds of the decoding: 0.5 / 0.5 (chosen on the validation set, see the slides).

    python -m gnn_review.train --variant base       # K = 2, as the adopted model (reference on the same machine)
    python -m gnn_review.train --variant k1         # K = 1
    python -m gnn_review.train --variant k3         # K = 3
    python -m gnn_review.train --variant nofalse    # pair head only
    python -m gnn_review.train --variant logdeg     # + log(1 + |N(d)|) in the node input
    python -m gnn_review.train --variant world      # world frame of the plan instead of the robots' centroid

A seed whose model_seed<k>.pt already exists is not trained again (the run can be stopped and restarted);
--eval-only evaluates the saved models. Outputs in --out/<variant>/: model_seed<k>.pt, train_log.csv,
variant.json, summary.csv (one row per test set: mean and sd over the seeds), test_scenes.csv.
"""

import argparse
import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch

from gnn.dataset import build_scenes
from gnn.model import collate
from gnn.train_sim import class_weights, evaluate_scenes, stats, write_csv
from gnn_review.model import ReviewGNN, encode, loss_fn
from pipeline.evaluate_runs import MATCH_CUTOFF_M

VARIANTS = {
    "base":    dict(hops=2, frame="centroid", degree=False, false_head=True),
    "k1":      dict(hops=1, frame="centroid", degree=False, false_head=True),
    "k3":      dict(hops=3, frame="centroid", degree=False, false_head=True),
    "nofalse": dict(hops=2, frame="centroid", degree=False, false_head=False),
    "logdeg":  dict(hops=2, frame="centroid", degree=True,  false_head=True),
    "world":   dict(hops=2, frame="world",    degree=False, false_head=True),
}
METRICS = ("correct_pct", "exact_pct", "pd_pct", "false_per_scene", "rmse_all_cm", "too_few_pct", "too_many_pct")


def load(directory, calib):
    return build_scenes(directory, os.path.join(directory, "ground_truth.json"), calib)


def build(v, hidden):
    return ReviewGNN(hidden=hidden, hops=v["hops"], degree=v["degree"], false_head=v["false_head"])


def train_one(train_enc, val_scenes, val_enc, v, a, seed, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(v, a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    w_same, w_false = class_weights(train_enc)          # once over the whole training set
    best, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_enc))
        total = 0.0
        for start in range(0, len(order), a.batch):
            b = collate([train_enc[i] for i in order[start:start + a.batch]])
            same, false = model(b)
            loss = loss_fn(same, false, b, w_same, w_false, v["false_head"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        val = evaluate_scenes(model, val_scenes, val_enc, a)
        correct = 100 * float(np.mean([r["correct"] for r in val]))
        log.append(dict(seed=seed, epoch=epoch, loss=round(total / max(1, math.ceil(len(order) / a.batch)), 5),
                        val_correct_pct=round(correct, 2), seconds=round(time.time() - t0, 1)))
        print(f"    seed {seed} epoch {epoch:3d}: loss {log[-1]['loss']:.4f}  validation correct {correct:5.1f}%  "
              f"({log[-1]['seconds']:.0f} s)", flush=True)
        if correct > best:
            best, best_state, best_epoch, waited = correct, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= a.patience:
                break
    model.load_state_dict(best_state)
    print(f"    seed {seed}: kept epoch {best_epoch} (validation correct {best:.1f}%)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--train-dir", default="data/plan_v2/train/")
    ap.add_argument("--val-dir", default="data/plan_v2/val/")
    ap.add_argument("--plan-dir", default="data/plan_v2/")
    ap.add_argument("--tests", nargs="*", default=["G0_*", "E1_*", "E3_*", "S1_*"],
                    help="test sets of --plan-dir (names or patterns); none: validation only")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--out", default="results/review/")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gate", type=float, default=1.0, help="candidate-pair distance d_c [m]")
    ap.add_argument("--max-train", type=int, default=None, help="first N training scenes only (quick check)")
    ap.add_argument("--max-val", type=int, default=None, help="first N validation scenes only (quick check)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default: all cores)")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    a.cutoff, a.ospa_p = MATCH_CUTOFF_M, 2.0
    v = VARIANTS[a.variant]
    out = os.path.join(a.out, a.variant)
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(a.threads or max(1, os.cpu_count() or 1))
    n_par = sum(p.numel() for p in build(v, a.hidden).parameters())
    print(f"variant {a.variant} {v}: {n_par:,} parameters -> {out}", flush=True)
    with open(os.path.join(out, "variant.json"), "w") as f:
        json.dump(dict(variant=a.variant, parameters=n_par, **v, **{k: x for k, x in vars(a).items()
                                                                       if k not in ("variant", "eval_only")}), f, indent=2)

    def enc(scenes):
        return [encode(s, v["frame"], v["degree"], a.gate) for s in scenes]

    t0 = time.time()
    val = load(a.val_dir, a.sensor_calib)[:a.max_val]
    val_enc = enc(val)
    models, train_enc = [], None
    log_path = os.path.join(out, "train_log.csv")
    log = []
    if os.path.exists(log_path):
        import csv
        with open(log_path) as f:
            log = list(csv.DictReader(f))
    for seed in range(a.seeds):
        path = os.path.join(out, f"model_seed{seed}.pt")
        if os.path.exists(path):
            m = build(v, a.hidden)
            m.load_state_dict(torch.load(path))
            print(f"  seed {seed}: loaded {path}", flush=True)
        elif a.eval_only:
            raise SystemExit(f"no {path}")
        else:
            if train_enc is None:
                train = load(a.train_dir, a.sensor_calib)
                if a.max_train:
                    train = train[:a.max_train]
                train_enc = enc(train)
                print(f"{len(train)} training scenes, {len(val)} validation scenes "
                      f"({time.time() - t0:.0f} s to load)", flush=True)
            log = [r for r in log if int(r["seed"]) != seed]
            m = train_one(train_enc, val, val_enc, v, a, seed, log)
            torch.save(m.state_dict(), path)
            write_csv(log_path, log)
        models.append(m)

    sets = [("VAL", val, val_enc)]
    names = sorted({os.path.basename(os.path.normpath(d)) for p in (a.tests or [])
                    for d in glob.glob(os.path.join(a.plan_dir, p)) if os.path.isdir(d)})
    for n in names:
        sc = load(os.path.join(a.plan_dir, n), a.sensor_calib)
        sets.append((n, sc, enc(sc)))
    summary, scenes_rows = [], []
    for name, sc, e in sets:
        per_seed = []
        for seed, m in enumerate(models):
            rows = evaluate_scenes(m, sc, e, a)
            for r in rows:
                r.update(test=name, seed=seed)
            scenes_rows += rows
            per_seed.append(stats(rows, a.cutoff))
        row = dict(variant=a.variant, test=name, scenes=len(sc), seeds=len(models))
        for k in METRICS:
            vals = np.array([s[k] for s in per_seed], float)
            row[k] = round(float(np.nanmean(vals)), 3)
            row[k + "_sd"] = round(float(np.nanstd(vals)), 3)
        summary.append(row)
        print(f"  {name:<20} correct {row['correct_pct']:5.1f}±{row['correct_pct_sd']:.1f}%  "
              f"P_D {row['pd_pct']:5.1f}%  false {row['false_per_scene']:.3f}  RMSE* {row['rmse_all_cm']:.1f} cm",
              flush=True)
    write_csv(os.path.join(out, "summary.csv"), summary)
    write_csv(os.path.join(out, "test_scenes.csv"), scenes_rows)
    print(f"written in {out}", flush=True)


if __name__ == "__main__":
    main()
