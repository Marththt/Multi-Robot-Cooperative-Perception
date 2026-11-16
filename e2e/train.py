"""Trains and tests the end-to-end network (e2e/model.py) on the simulated sets of e2e/data.py and the real
scenes of e2e/real.py.

The epoch and the detection threshold are chosen on the validation set (share of correct scenes: right number
of boxes and every box within 15 cm, the metric of MHT and the marker network, counted against the boxes some
robot can see); no test scene is used for anything but the test. Outputs in --out:
  model.pt, train_log.csv, variant.json
  test_scenes.csv   one row per test scene: n_true, n_est, dn, n_matched, rmse, correct, run, set
  summary.csv       per test set: correct, too few, too many, misplaced, RMSE, box recall and precision, F1
Real scenes (sets with scenario "real"): every robot took several frames per stop; the box-centre maps of the
frames of one stop are averaged before decoding, one estimate per real scene. With --levels and --mht the real
summary is also given per difficulty level, next to MHT on the same scenes.

Runs on a CUDA GPU when there is one (--device auto), otherwise on the CPU (slow: start with --max-train).

    python -m e2e.train --train data/e2e/train --val data/e2e/val --tests "data/e2e/G0_*" "data/e2e/E3_*" \\
        "data/e2e/S1_*" data/e2e_real/real --mht results/evaluation_all/scenes.csv --out results/e2e/
    python -m e2e.train --eval-only --out results/e2e/ --tests data/e2e_real/real --results-name real
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from e2e import geometry as geo
from e2e.data import E2EDataset, ShardSampler, collate
from e2e.model import E2ENet, losses
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics

THRESHOLDS = np.round(np.arange(0.20, 0.751, 0.05), 2)


def expand(patterns):
    return sorted({d for p in patterns for d in (glob.glob(p) if any(c in p for c in "*?[") else [p])
                   if os.path.isdir(d)})


def loader(ds, batch, shuffle, workers, seed=0):
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=ShardSampler(ds, seed) if shuffle else None,
                                       num_workers=workers, collate_fn=collate, persistent_workers=workers > 0,
                                       pin_memory=torch.cuda.is_available())


def to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_iter(model, dl, device):
    """Per scene: (probability map (G, G), offsets (2, G, G), true boxes, run, set, seen boxes (B,) bool)."""
    model.eval()
    with torch.no_grad():
        for b in dl:
            b = to(b, device)
            heat, off = model(b["images"], b["grids"], b["valid"], b["view"], b["scene"], b["n_scenes"])
            heat = torch.sigmoid(heat[:, 0]).float().cpu().numpy()
            off = off.float().cpu().numpy()
            for k in range(b["n_scenes"]):
                yield heat[k], off[k], b["boxes"][k], b["run"][k], b["set"][k], b["seen"][k]


def score_one(heat, off, boxes, run, set_name, threshold, seen=None, cutoff=MATCH_CUTOFF_M):
    """correct: against every box. correct_seen: against the boxes some robot can see (the same on a scene in
    which every box is seen)."""
    est = geo.decode(heat, off, threshold)
    boxes = np.asarray(boxes)
    m = scene_metrics(list(est), list(boxes), cutoff, 2.0)
    m["correct"] = int(m["dn"] == 0 and m["n_matched"] == m["n_true"])
    seen = np.ones(len(boxes), bool) if seen is None else np.asarray(seen, bool)
    m["solvable"] = int(seen.all())
    if m["solvable"]:
        m["correct_seen"] = m["correct"]
    else:
        ms = scene_metrics(list(est), list(boxes[seen]), cutoff, 2.0)
        m["correct_seen"] = int(ms["dn"] == 0 and ms["n_matched"] == ms["n_true"])
    return dict(m, run=run, set=set_name)


def fuse_frames(preds):
    """Real sets: the maps of the frames of one stop (same run id) averaged, one prediction per stop."""
    groups = defaultdict(list)
    for p in preds:
        groups[(p[4], p[3])].append(p)
    out = []
    for (set_name, run), ps in groups.items():
        out.append((np.mean([p[0] for p in ps], 0), np.mean([p[1] for p in ps], 0), ps[0][2], run, set_name, ps[0][5]))
    return out


def summarize(rows, **key):
    n_true = sum(r["n_true"] for r in rows)
    n_est = sum(r["n_est"] for r in rows)
    n_match = sum(r["n_matched"] for r in rows)
    rec = n_match / n_true if n_true else float("nan")
    prec = n_match / n_est if n_est else float("nan")
    return dict(**key, scenes=len(rows),
                correct_pct=round(100 * np.mean([r["correct"] for r in rows]), 2),
                too_few_pct=round(100 * np.mean([r["dn"] < 0 for r in rows]), 2),
                too_many_pct=round(100 * np.mean([r["dn"] > 0 for r in rows]), 2),
                misplaced_pct=round(100 * np.mean([r["dn"] == 0 and not r["correct"] for r in rows]), 2),
                rmse_cm=round(100 * float(np.nanmean([r["rmse"] for r in rows])), 2) if n_match else float("nan"),
                recall_pct=round(100 * rec, 2), precision_pct=round(100 * prec, 2),
                f1_pct=round(200 * rec * prec / (rec + prec), 2) if rec + prec > 0 else 0.0,
                correct_seen_pct=round(100 * np.mean([r["correct_seen"] for r in rows]), 2))


def read_levels(path):
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[(r["layout"], int(r["round"]))] = r["level"]
    return out


def read_mht(path, method):
    """(layout, round) -> metrics of MHT with all the robots (pipeline/evaluate_runs.py scenes.csv)."""
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["method"] != method or r["k"] != r["n_robots"]:
                    continue
                m = dict(n_true=int(r["n_true"]), n_est=int(r["n_est"]), dn=int(r["dn"]),
                         n_matched=int(r["n_matched"]), rmse=float(r["rmse"]) if r["rmse"] not in ("", "nan") else
                         float("nan"))
                m["correct"] = int(m["dn"] == 0 and m["n_matched"] == m["n_true"])
                m["correct_seen"] = m["correct"]
                out[(r["layout"], int(r["round"]))] = m
    return out


def real_key(run):
    """Run id of e2e/real.py (<layout>_r<round>) -> (layout, round)."""
    layout, rnd = run.rsplit("_r", 1)
    return layout, int(rnd)


def save_variant(out, a, threshold, best, epoch):
    with open(os.path.join(out, "variant.json"), "w") as f:
        json.dump(dict(vars(a), eval_only=False, threshold=threshold, best_val_correct_pct=best, best_epoch=epoch),
                  f, indent=2)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["data/e2e/train"])
    ap.add_argument("--val", nargs="+", default=["data/e2e/val"])
    ap.add_argument("--tests", nargs="*", default=["data/e2e/G0_*", "data/e2e/E3_*", "data/e2e/S1_*"])
    ap.add_argument("--out", default="results/e2e/")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--state", type=int, default=32, help="channels of a robot's node state per floor cell")
    ap.add_argument("--layers", type=int, default=2, help="rounds of message passing among the robots")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--jitter", type=float, default=0.3, help="colour and brightness jitter of training images")
    ap.add_argument("--pitch-noise", type=float, default=1.0,
                    help="sd [deg] of the error added to the pitch given with each training image")
    ap.add_argument("--yaw-noise", type=float, default=1.0, help="sd [deg] of the error added to the yaw")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="scenes per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--offset-weight", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training scenes")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6, help="data loader processes")
    ap.add_argument("--device", default="auto", help="auto (cuda if there is one, else cpu) | cuda | cpu | mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--levels", default="results/scene_levels.csv", help="difficulty level of every real scene")
    ap.add_argument("--mht", default=None, help="pipeline/evaluate_runs.py scenes.csv of the real scenes")
    ap.add_argument("--mht-method", default="mht_sensor")
    ap.add_argument("--save-maps", type=int, default=0, help="save the maps of the first N scenes of every test set")
    ap.add_argument("--results-name", default="", help="test outputs as summary_<name>.csv, test_scenes_<name>.csv")
    a = ap.parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")     # macOS: few open files allowed per process
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    cuda = device.type == "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = a.out
    if a.eval_only and os.path.exists(os.path.join(out, "variant.json")):      # the network of the saved model
        with open(os.path.join(out, "variant.json")) as f:
            saved = json.load(f)
        for k in ("channels", "state", "layers", "heads"):
            setattr(a, k, saved[k])
    os.makedirs(out, exist_ok=True)
    print(f"device {device}, output {out}", flush=True)
    model = E2ENet(a.channels, a.state, len(geo.HEIGHTS_M), a.layers, a.heads).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    if a.eval_only:
        model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device))
        with open(os.path.join(out, "variant.json")) as f:
            threshold = json.load(f)["threshold"]
    else:
        val = E2EDataset(expand(a.val), a.max_val)
        val_dl = loader(val, a.batch, False, a.workers)
        train = E2EDataset(expand(a.train), a.max_train)
        train.pitch_noise_deg, train.yaw_noise_deg = a.pitch_noise, a.yaw_noise
        print(f"{len(train)} training scenes, {len(val)} validation scenes", flush=True)
        train_dl = loader(train, a.batch, True, a.workers, a.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl),
                                                    pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        best, log, waited, threshold = -1.0, [], 0, 0.5
        try:                                                  # Ctrl+C: stop training, test the best epoch
            for epoch in range(1, a.epochs + 1):
                model.train()
                t0, total, nb = time.time(), 0.0, 0
                for b in train_dl:
                    b = to(b, device)
                    img = b["images"]
                    if a.jitter > 0:                                  # colour, brightness and contrast of each image
                        n = img.shape[0]
                        gain = 1 + a.jitter * (torch.rand(n, 3, 1, 1, device=device) - 0.5)
                        bias = 0.5 * a.jitter * (torch.rand(n, 1, 1, 1, device=device) - 0.5)
                        img = (img * gain + bias).clamp(0, 1)
                    with (torch.autocast("cuda", dtype=torch.float16) if cuda else contextlib.nullcontext()):
                        heat, off = model(img, b["grids"], b["valid"], b["view"], b["scene"], b["n_scenes"])
                    focal, l1 = losses(heat.float(), off.float(), b["heat"], b["off"], b["mask"])
                    loss = focal + a.offset_weight * l1
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    total += loss.item()
                    nb += 1
                preds = list(predict_iter(model, val_dl, device))
                rates = {t: 100 * np.mean([score_one(h, o, bx, r, s, t, v)["correct_seen"]
                                           for h, o, bx, r, s, v in preds]) for t in THRESHOLDS}
                t_best = max(rates, key=rates.get)
                log.append(dict(epoch=epoch, loss=round(total / max(nb, 1), 4), val_correct_pct=round(rates[t_best], 2),
                                threshold=float(t_best), seconds=round(time.time() - t0, 1)))
                print(f"  epoch {epoch:3d}: loss {log[-1]['loss']:.4f}  validation correct {rates[t_best]:5.1f}% "
                      f"(threshold {t_best:.2f})  ({log[-1]['seconds']:.0f} s)", flush=True)
                write_csv(os.path.join(out, "train_log.csv"), log)
                if rates[t_best] > best:
                    best, threshold, waited = rates[t_best], float(t_best), 0
                    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
                    save_variant(out, a, threshold, best, epoch)
                else:
                    waited += 1
                    if waited >= a.patience:
                        break
        except KeyboardInterrupt:
            print("  training stopped by the user: the best epoch so far is tested", flush=True)
        model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device))
        top = max(log, key=lambda r: r["val_correct_pct"])
        print(f"kept epoch {top['epoch']} (validation correct {best:.1f}%, threshold {threshold:.2f})", flush=True)

    dirs = expand(a.tests)
    if not dirs:
        return
    ds = E2EDataset(dirs)                     # one data loader over all the test sets
    real_sets = {name for name, m in ds.meta.items() if m.get("scenario") == "real"}
    preds = list(predict_iter(model, loader(ds, a.batch, False, a.workers), device))
    preds = [p for p in preds if p[4] not in real_sets] + fuse_frames([p for p in preds if p[4] in real_sets])
    rows = [score_one(h, o, bx, r, s, threshold, v) for h, o, bx, r, s, v in preds]
    levels, mht = read_levels(a.levels), read_mht(a.mht, a.mht_method)
    for r in rows:
        if r["set"] in real_sets:
            key = real_key(r["run"])
            r["layout"], r["round"], r["level"] = key[0], key[1], levels.get(key, "")
            if key in mht:
                r["mht_correct"] = mht[key]["correct"]
    summary = []
    for set_name in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == set_name]
        summary.append(summarize(rs, set=set_name, group="all", method="e2e"))
        if set_name in real_sets:
            for lv in sorted({r["level"] for r in rs if r["level"]}):
                summary.append(summarize([r for r in rs if r["level"] == lv], set=set_name, group=lv, method="e2e"))
            if mht:                           # MHT and the network on the same real scenes
                both = [r for r in rs if (r["layout"], r["round"]) in mht]
                for grp, sel in [("all", both)] + [(lv, [r for r in both if r["level"] == lv])
                                                    for lv in sorted({r["level"] for r in both if r["level"]})]:
                    if sel:
                        summary.append(summarize(sel, set=set_name, group=f"{grp} (with MHT)", method="e2e"))
                        summary.append(summarize([mht[(r["layout"], r["round"])] for r in sel], set=set_name,
                                                 group=f"{grp} (with MHT)", method=a.mht_method))
    for s in summary:
        print(f"  {s['set']:20s} {s['group']:22s} {s['method']:11s} {s['scenes']:5d} scenes  correct "
              f"{s['correct_pct']:5.1f}%  too few {s['too_few_pct']:5.1f}%  too many {s['too_many_pct']:5.1f}%  "
              f"F1 {s['f1_pct']:5.1f}%  RMSE {s['rmse_cm']:.1f} cm", flush=True)
    sfx = f"_{a.results_name}" if a.results_name else ""
    write_csv(os.path.join(out, f"test_scenes{sfx}.csv"), rows)
    write_csv(os.path.join(out, f"summary{sfx}.csv"), summary)
    if a.save_maps:
        keep = defaultdict(list)
        for h, o, bx, r, s, v in preds:
            if len(keep[s]) < a.save_maps:
                keep[s].append((s, h.astype(np.float16), o.astype(np.float16), np.asarray(bx), r,
                                geo.decode(h, o, threshold)))
        flat = [m for s in sorted(keep) for m in keep[s]]
        np.savez_compressed(os.path.join(out, f"maps{sfx}.npz"), set=np.array([f[0] for f in flat]),
                            heat=np.stack([f[1] for f in flat]), off=np.stack([f[2] for f in flat]),
                            truth=np.array([f[3] for f in flat], dtype=object), run=np.array([f[4] for f in flat]),
                            est=np.array([f[5] for f in flat], dtype=object), threshold=threshold)
    print(f"written in {out}")


if __name__ == "__main__":
    main()
