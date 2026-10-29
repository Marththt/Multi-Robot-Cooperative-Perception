"""GNN trained on the simulator: training on the general environment, model selection on a validation set
of the same distribution, then one evaluation per test set (failure-prone simulated conditions T0-T7 and
the real runs). No test scene is ever used for training or for choosing anything.

    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ \\
        --sensor-calib config/sensor_calib.json --standard-tests \\
        --graph robot --features full --edges ring --hops 2 --seeds 5 --out results/gnn_sim/

The sets come from sim.make_sets; --standard-tests tests on T0-T7 and on the 56 real scenes. Other test sets
can be added with --test (below). --combined-tests adds the real scenes with combined rounds (4 to 28
viewpoints, pipeline.combine_rounds); with --eval-only and no other test they are written to
summary_combined.csv, mcnemar_combined.csv, test_scenes_combined.csv, next to the main results:
    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ \\
        --sensor-calib config/sensor_calib.json --combined-tests --eval-only --out results/gnn_sim/

Variant (the folder name in --out is built from it, e.g. robot_full_ring_h2):
  --graph     robot (one node per robot) | detection (one node per detection)
  --features  full (with the viewing angle) | positions (positions and robot poses only)
  --edges     ring (ring neighbours) | range (robots closer than --comm-range) | full (every pair)
  --hops      rounds of message passing

Training: Adam, --epochs at most; after every epoch the model is run on the validation scenes and the epoch
with the highest correct rate is kept (early stop after --patience epochs without improvement). --seeds
models are trained independently; every result is reported as mean and standard deviation over them.

A --test is NAME DIR GROUND_TRUTH BASELINE: BASELINE is the scenes.csv of pipeline/evaluate_runs.py on the
same runs ('-' for none). The GNN and every classical method are compared scene by scene; McNemar's exact
test is run on "correct" (count right and every true box matched) for every seed. Simulated scenes in which
no robot reported anything are counted for every method as an empty estimate.

Outputs (in --out/<variant>/): model_seed<k>.pt, variant.json, train_log.csv, test_scenes.csv (GNN, one row
per test scene and seed), summary.csv (per test set and method), mcnemar.csv. --eval-only skips the
training and evaluates the saved models.
"""

import argparse
import copy
import csv
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from gnn.dataset import build_scenes
from gnn.model import build_model, collate, decode, encode_scene, loss_fn
from pipeline.evaluate_runs import scene_metrics

METHOD_ORDER = ["zone_class", "union", "phd", "cphd", "mht"]
LABELS = dict(zone_class="Zone+class", union="Union", phd="PHD", cphd="CPHD", mht="MHT", gnn="GNN")


def variant_name(a):
    edges = a.edges if a.edges != "range" else f"range{a.comm_range:g}"
    return f"{a.graph}_{a.features}_{edges}_h{a.hops}"


def encode_all(scenes, a):
    return [encode_scene(s, features=a.features, edges=a.edges, comm_range=a.comm_range, gate=a.gate)
            for s in scenes]


def class_weights(encoded):
    """pos_weight of the two heads: negatives / positives in the training scenes (capped at 10)."""
    pos_same = sum(float(e["same"].sum()) for e in encoded)
    n_same = sum(len(e["same"]) for e in encoded)
    pos_false = sum(float(e["false"].sum()) for e in encoded)
    n_false = sum(len(e["false"]) for e in encoded)
    return (min(10.0, (n_same - pos_same) / max(pos_same, 1)), min(10.0, (n_false - pos_false) / max(pos_false, 1)))


def predict(model, encoded, batch=64):
    """(p_same, p_false) per scene."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch):
            chunk = encoded[start:start + batch]
            same, false = model(collate(chunk))
            same = torch.sigmoid(same).numpy()
            false = torch.sigmoid(false).numpy()
            ps = np.cumsum([0] + [len(e["pairs"]) for e in chunk])
            pd = np.cumsum([0] + [len(e["det"]) for e in chunk])
            for k in range(len(chunk)):
                out.append((same[ps[k]:ps[k + 1]], false[pd[k]:pd[k + 1]]))
    return out


def evaluate_scenes(model, scenes, encoded, a):
    rows = []
    for s, e, (p_same, p_false) in zip(scenes, encoded, predict(model, encoded)):
        est, _ = decode(s["pos"], e["pairs"], p_same, p_false)
        r = scene_metrics(est, list(s["truth"]), a.cutoff, a.ospa_p)
        r["correct"] = int(r["dn"] == 0 and r["n_matched"] == r["n_true"])
        r.update(run=s["run"], layout=s["layout"], round=s["round"], n_robots=len(s["robots"]))
        rows.append(r)
    return rows


def train_one(train_enc, val_scenes, val_enc, a, seed, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(a.graph, a.hidden, a.hops, a.features)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    w_same, w_false = class_weights(train_enc)
    best, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_enc))
        total = 0.0
        for start in range(0, len(order), a.batch):
            b = collate([train_enc[i] for i in order[start:start + a.batch]])
            same, false = model(b)
            loss = loss_fn(same, false, b, w_same, w_false)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        val = evaluate_scenes(model, val_scenes, val_enc, a)
        correct = 100 * float(np.mean([r["correct"] for r in val]))
        exact = 100 * float(np.mean([r["dn"] == 0 for r in val]))
        log.append(dict(seed=seed, epoch=epoch, loss=round(total / max(1, math.ceil(len(order) / a.batch)), 5),
                        val_exact_pct=round(exact, 2), val_correct_pct=round(correct, 2),
                        seconds=round(time.time() - t0, 1)))
        print(f"    seed {seed} epoch {epoch:3d}: loss {log[-1]['loss']:.4f}  validation exact {exact:5.1f}%  "
              f"correct {correct:5.1f}%  ({log[-1]['seconds']:.0f} s)", flush=True)
        if correct > best:
            best, best_state, best_epoch, waited = correct, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= a.patience:
                break
    model.load_state_dict(best_state)
    print(f"    seed {seed}: kept epoch {best_epoch} (validation correct {best:.1f}%)", flush=True)
    return model


# ====================================================================
# Test sets and baselines
# ====================================================================

def expected_runs(directory):
    """Run ids of a simulated set (runs.csv of sim.generate random), None for real runs."""
    path = os.path.join(directory, "runs.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [r["run"] for r in csv.DictReader(f)]


def empty_rows(runs, seen, truth, a):
    """Scenes in which no robot reported anything: an empty estimate (round 0, one round per sim run)."""
    rows = []
    for run in runs or []:
        if run in seen or run not in truth:
            continue
        r = scene_metrics([], [np.array([b["x"], b["y"]]) for b in truth[run]], a.cutoff, a.ospa_p)
        r["correct"] = int(r["dn"] == 0 and r["n_matched"] == r["n_true"])
        r.update(run=run, layout=run, round=0, n_robots=0)
        rows.append(r)
    return rows


def read_baseline(path):
    """{method: {(run, round): row}} of the all-robot rows of pipeline/evaluate_runs.py scenes.csv."""
    out = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["k"]) != int(r["n_robots"]):
                continue
            row = dict(n_true=int(r["n_true"]), n_est=int(r["n_est"]), dn=int(r["dn"]), n_matched=int(r["n_matched"]),
                       rmse=float(r["rmse"]) if r["rmse"] not in ("", "nan") else math.nan, ospa=float(r["ospa"]))
            row["correct"] = int(row["dn"] == 0 and row["n_matched"] == row["n_true"])
            out[r["method"]][(r["run"], int(r["round"]))] = row
            out[r["method"]].setdefault(("layout", r["layout"], int(r["round"])), row)
    return out


def baseline_row(table, row):
    return table.get((row["run"], row["round"])) or table.get(("layout", row["layout"], row["round"]))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    return min(1.0, 2 * sum(math.exp(x) for x in logs))


def stats(rows):
    dn = np.array([r["dn"] for r in rows])
    rmse = np.array([r["rmse"] for r in rows], float)
    rmse = rmse[np.isfinite(rmse)]
    return dict(scenes=len(rows), exact_pct=100 * float(np.mean(dn == 0)),
                correct_pct=100 * float(np.mean([r["correct"] for r in rows])),
                too_few_pct=100 * float(np.mean(dn < 0)), too_many_pct=100 * float(np.mean(dn > 0)),
                rmse_median_cm=100 * float(np.median(rmse)) if len(rmse) else math.nan,
                ospa_mean=float(np.mean([r["ospa"] for r in rows])))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--sensor-calib", required=True, help="pipeline/calibrate.py --out, the same for every set")
    ap.add_argument("--test", nargs=4, action="append", default=[], metavar=("NAME", "DIR", "GROUND_TRUTH", "BASELINE"))
    ap.add_argument("--standard-tests", action="store_true",
                    help="test on T0-T7 (data/sim_T*/, baselines in results/baselines_sim/T*/, as written by "
                         "sim.make_sets) and on the real runs (data/survey_rounds/, results/evaluation_calib/)")
    ap.add_argument("--combined-tests", action="store_true",
                    help="test on the real scenes with combined rounds (data/real_combined/rounds_*/, baselines in "
                         "results/baselines_real_combined/, as written by pipeline.combine_rounds)")
    ap.add_argument("--results-name", default=None,
                    help="suffix of the result files (summary_<name>.csv ...), so an evaluation on other test sets "
                         "does not overwrite the main one (default: 'combined' with only --combined-tests)")
    ap.add_argument("--out", default="results/gnn_sim/")
    ap.add_argument("--graph", choices=["robot", "detection"], default="robot")
    ap.add_argument("--features", choices=["full", "positions"], default="full")
    ap.add_argument("--edges", choices=["ring", "range", "full"], default="ring")
    ap.add_argument("--comm-range", type=float, default=1.5, help="communication range [m] with --edges range")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--gate", type=float, default=1.0, help="pairs of detections farther apart are not scored [m]")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training scenes (quick tests)")
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--ospa-p", type=float, default=2.0)
    ap.add_argument("--eval-only", action="store_true", help="evaluate the models already saved in the output folder")
    a = ap.parse_args()

    if a.standard_tests:
        for k in range(8):
            d = f"data/sim_T{k}/"
            if os.path.isdir(d):
                a.test.append([f"T{k}", d, os.path.join(d, "ground_truth.json"),
                               f"results/baselines_sim/T{k}/scenes.csv"])
        a.test.append(["REAL", "data/survey_rounds/", "config/ground_truth_positions.json",
                       "results/evaluation_calib/scenes.csv"])
    if a.results_name is None:
        a.results_name = "combined" if a.combined_tests and not a.standard_tests and not a.test else ""
    if a.combined_tests:
        for r in range(1, 8):
            d = f"data/real_combined/rounds_{r}/"
            if os.path.isdir(d):
                a.test.append([f"REALx{r}", d, os.path.join(d, "ground_truth.json"),
                               f"results/baselines_real_combined/rounds_{r}/scenes.csv"])
    for t in a.test:
        if t[3] != "-" and not os.path.exists(t[3]):
            print(f"warning: no baseline {t[3]} for {t[0]}; the GNN alone is evaluated")
            t[3] = "-"
    out = os.path.join(a.out, variant_name(a))
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"variant {variant_name(a)} -> {out}")

    models = []
    if a.eval_only:
        for seed in range(a.seeds):
            m = build_model(a.graph, a.hidden, a.hops, a.features)
            m.load_state_dict(torch.load(os.path.join(out, f"model_seed{seed}.pt")))
            models.append(m)
    else:
        t0 = time.time()
        train = build_scenes(a.train_dir, os.path.join(a.train_dir, "ground_truth.json"), a.sensor_calib)
        if a.max_train:
            train = train[:a.max_train]
        val = build_scenes(a.val_dir, os.path.join(a.val_dir, "ground_truth.json"), a.sensor_calib)
        train_enc, val_enc = encode_all(train, a), encode_all(val, a)
        print(f"{len(train)} training scenes, {len(val)} validation scenes ({time.time() - t0:.0f} s to load)",
              flush=True)
        log = []
        for seed in range(a.seeds):
            m = train_one(train_enc, val, val_enc, a, seed, log)
            torch.save(m.state_dict(), os.path.join(out, f"model_seed{seed}.pt"))
            models.append(m)
            write_csv(os.path.join(out, "train_log.csv"), log)
        with open(os.path.join(out, "variant.json"), "w") as f:
            json.dump({k: v for k, v in vars(a).items() if k not in ("test", "eval_only")}, f, indent=2)

    summary, tests_rows, mc = [], [], []
    for name, directory, gt_path, baseline_path in a.test:
        scenes = build_scenes(directory, gt_path, a.sensor_calib)
        enc = encode_all(scenes, a)
        with open(gt_path) as f:
            truth = json.load(f)
        runs = expected_runs(directory)
        seen = {s["run"] for s in scenes}
        empties = empty_rows(runs, seen, truth, a)
        per_seed = []
        for seed, m in enumerate(models):
            rows = evaluate_scenes(m, scenes, enc, a) + [dict(r) for r in empties]
            for r in rows:
                r.update(test=name, seed=seed)
            per_seed.append(rows)
            tests_rows += rows

        seed_stats = [stats(rows) for rows in per_seed]
        g = dict(test=name, method="gnn", scenes=seed_stats[0]["scenes"],
                 viewpoints=round(float(np.mean([r["n_robots"] for r in per_seed[0]])), 1))
        for k in ("exact_pct", "correct_pct", "too_few_pct", "too_many_pct", "rmse_median_cm", "ospa_mean"):
            vals = np.array([s[k] for s in seed_stats], float)
            g[k] = round(float(np.nanmean(vals)), 3)
            g[k + "_sd"] = round(float(np.nanstd(vals)), 3)
        summary.append(g)
        line = f"  {name:<6} ({g['viewpoints']:4.1f} viewpoints) GNN {g['exact_pct']:5.1f}±{g['exact_pct_sd']:.1f}% exact, " \
               f"{g['correct_pct']:5.1f}±{g['correct_pct_sd']:.1f}% correct"

        if baseline_path != "-":
            base = read_baseline(baseline_path)
            for method in [m for m in METHOD_ORDER if m in base]:
                paired = [(r, baseline_row(base[method], r)) for r in per_seed[0]]
                missing = sum(1 for _, b in paired if b is None)
                rows = [b for _, b in paired if b is not None]
                # scenes without any detection are absent from scenes.csv: empty estimate for this method too
                rows += [r for r, b in paired if b is None and r.get("n_robots", 1) == 0]
                st = stats(rows)
                summary.append(dict(test=name, method=method, **{k: round(v, 3) for k, v in st.items()}))
                for seed, rows_g in enumerate(per_seed):
                    b = c = 0
                    for r in rows_g:
                        rb = baseline_row(base[method], r)
                        ok_b = rb["correct"] if rb else (r["correct"] if r.get("n_robots", 1) == 0 else None)
                        if ok_b is None:
                            continue
                        b += int(r["correct"] and not ok_b)
                        c += int(ok_b and not r["correct"])
                    mc.append(dict(test=name, baseline=method, seed=seed, gnn_only=b, baseline_only=c,
                                   p_value=mcnemar(b, c)))
                if missing - sum(1 for r, bb in paired if bb is None and r.get("n_robots", 1) == 0):
                    print(f"    warning: {method}: some {name} scenes have no row in {baseline_path}")
                if method == "mht":
                    ps = [x["p_value"] for x in mc if x["test"] == name and x["baseline"] == "mht"]
                    line += f" | MHT {st['exact_pct']:5.1f}% exact, {st['correct_pct']:5.1f}% correct " \
                            f"(McNemar p max over seeds {max(ps):.2g})"
        print(line, flush=True)

    sfx = f"_{a.results_name}" if a.results_name else ""
    write_csv(os.path.join(out, f"test_scenes{sfx}.csv"), tests_rows)
    write_csv(os.path.join(out, f"summary{sfx}.csv"), summary)
    write_csv(os.path.join(out, f"mcnemar{sfx}.csv"), mc)
    print(f"\nwritten in {out}: summary{sfx}.csv, mcnemar{sfx}.csv, test_scenes{sfx}.csv"
          + ("" if a.eval_only else ", models"))


if __name__ == "__main__":
    main()
