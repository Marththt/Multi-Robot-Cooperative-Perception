"""Trains and tests the image-based network (bevnet/model.py) on the sets of bevnet/data.py.

The epoch and the detection threshold are chosen on the validation set (the share of correct scenes: right
number of boxes and every box within 15 cm, the metric of MHT and the GNN, counted against the boxes some robot
can see: sets rendered as masks record which ones; see bevnet/data.py); no test scene is used for anything
but the test. Outputs in --out/<variant>/: model.pt, train_log.csv, variant.json, test_scenes.csv (one row per
test scene: n_true, n_est, dn, n_matched, rmse, correct, run, set) and summary.csv (per test set: correct, too
few, too many, misplaced boxes, RMSE). The scenes have the run ids of sim/plan.py, so bevnet/compare.py can put
them next to MHT and the GNN on the same scenes.

Runs on a CUDA GPU when there is one (--device auto), otherwise on the CPU (slow: start with --max-train).

    python -m bevnet.train --train data/bev/train --val data/bev/val --tests "data/bev/G0_*" "data/bev/E3_*" "data/bev/S1_*"
    python -m bevnet.train ... --eval-only           # test the saved model again
    python -m bevnet.train --eval-only --out results/bev_rand/ --tests "data/bev_unseen/*" --results-name unseen
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from bevnet import geometry as geo
from bevnet.data import BevDataset, ShardSampler, collate
from bevnet.model import BevNet, losses
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics

THRESHOLDS = np.round(np.arange(0.20, 0.751, 0.05), 2)


def expand(patterns):
    return sorted({d for p in patterns for d in (glob.glob(p) if any(c in p for c in "*?[") else [p])
                   if os.path.isdir(d)})


def loader(ds, batch, shuffle, workers, seed=0, persistent=True):
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=ShardSampler(ds, seed) if shuffle else None,
                                       num_workers=workers,
                                       collate_fn=collate, persistent_workers=persistent and workers > 0,
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


def predict(model, dl, device):
    return list(predict_iter(model, dl, device))


def score_one(heat, off, boxes, run, set_name, threshold, cutoff=MATCH_CUTOFF_M, min_dist_m=0.0, seen=None):
    """correct: against every box. solvable: every box is seen by some robot. correct_seen: against the seen
    boxes only (what the images allow; the same as correct on a solvable scene)."""
    est = geo.decode(heat, off, threshold, min_dist_m)
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


def score(preds, threshold, cutoff=MATCH_CUTOFF_M):
    return [score_one(h, o, b, r, s, threshold, cutoff, seen=v) for h, o, b, r, s, v in preds]


def save_variant(out, a, threshold, best, epoch):
    with open(os.path.join(out, "variant.json"), "w") as f:
        json.dump(dict(vars(a), eval_only=False, threshold=threshold, best_val_correct_pct=best, best_epoch=epoch),
                  f, indent=2)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["data/bev/train"])
    ap.add_argument("--val", nargs="+", default=["data/bev/val"])
    ap.add_argument("--tests", nargs="*", default=["data/bev/G0_*", "data/bev/E3_*", "data/bev/S1_*"])
    ap.add_argument("--out", default="results/bev/")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--encoder", choices=["small", "resnet18"], default="small",
                    help="resnet18: ImageNet-pretrained ResNet-18 (needs torchvision)")
    ap.add_argument("--encoder-lr", type=float, default=0.2, help="learning rate of a pretrained encoder / --lr")
    ap.add_argument("--jitter", type=float, default=0.0, help="colour and brightness jitter of training images")
    ap.add_argument("--pitch-noise", type=float, default=0.0,
                    help="sd [deg] of the error added to the pitch given with each training image")
    ap.add_argument("--yaw-noise", type=float, default=0.0, help="sd [deg] of the error added to the yaw")
    ap.add_argument("--height-noise", type=float, default=0.0, help="sd [m] of the error added to the camera height")
    ap.add_argument("--mask-noise", type=float, default=0.0,
                    help="training on masks: imperfect segmentation (1 = edges, holes, blobs, missed boxes)")
    ap.add_argument("--min-visible-px", type=int, default=12,
                    help="a box counts as seen with this many visible pixels in one image (sets rendered as masks)")
    ap.add_argument("--name", default=None, help="variant folder name (default bev_c<channels>_m<message>)")
    ap.add_argument("--message", type=int, default=16, help="channels per floor cell a robot sends to the others")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="scenes per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--offset-weight", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training scenes")
    ap.add_argument("--max-val", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4, help="data loader processes")
    ap.add_argument("--device", default="auto", help="auto (cuda if there is one, else cpu) | cuda | cpu | mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--save-maps", type=int, default=0,
                    help="save the box-centre maps of the first N scenes of every test set (maps_<name>.npz)")
    ap.add_argument("--diagnose", action="store_true",
                    help="also print correct %% per test set for other thresholds and peak distances (small sets)")
    ap.add_argument("--results-name", default="", help="test outputs as summary_<name>.csv, test_scenes_<name>.csv")
    a = ap.parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")     # macOS: few open files allowed per process
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    cuda = device.type == "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = os.path.join(a.out, a.name or f"bev_c{a.channels}_m{a.message}")
    if a.eval_only and os.path.exists(os.path.join(out, "variant.json")):      # the network of the saved model
        with open(os.path.join(out, "variant.json")) as f:
            saved = json.load(f)
        a.channels, a.message, a.encoder = saved["channels"], saved["message"], saved.get("encoder", "small")
    os.makedirs(out, exist_ok=True)
    print(f"device {device}, variant {out}", flush=True)
    model = BevNet(a.channels, a.message, len(geo.HEIGHTS_M), a.encoder, pretrained=not a.eval_only).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    val = BevDataset(expand(a.val), a.max_val)
    val.min_visible_px = a.min_visible_px
    val_dl = loader(val, a.batch, False, a.workers)

    if a.eval_only:
        model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device))
        if os.path.exists(os.path.join(out, "variant.json")):
            with open(os.path.join(out, "variant.json")) as f:
                threshold = json.load(f)["threshold"]
        else:                                                 # training stopped before it wrote variant.json
            with open(os.path.join(out, "train_log.csv"), newline="") as f:
                rows = list(csv.DictReader(f))
            top = max(rows, key=lambda r: float(r["val_correct_pct"]))
            threshold = float(top["threshold"])
            save_variant(out, a, threshold, float(top["val_correct_pct"]), int(top["epoch"]))
            print(f"  variant.json rebuilt from train_log.csv: epoch {top['epoch']}, threshold {threshold:.2f}")
    else:
        train = BevDataset(expand(a.train), a.max_train)
        train.pitch_noise_deg, train.yaw_noise_deg, train.height_noise_m = a.pitch_noise, a.yaw_noise, a.height_noise
        train.mask_noise = a.mask_noise
        train.min_visible_px = a.min_visible_px
        print(f"{len(train)} training scenes, {len(val)} validation scenes", flush=True)
        train_dl = loader(train, a.batch, True, a.workers, a.seed)
        pre = [p for n, p in model.named_parameters() if n.startswith("encoder.") and a.encoder == "resnet18"
               and not n.startswith(("encoder.p", "encoder.out"))]
        ids = {id(p) for p in pre}
        groups = [dict(params=[p for p in model.parameters() if id(p) not in ids], lr=a.lr)]
        if pre:
            groups.append(dict(params=pre, lr=a.lr * a.encoder_lr))
        opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                    total_steps=a.epochs * len(train_dl),
                                                    pct_start=0.1)
        scaler = (torch.amp.GradScaler("cuda", enabled=cuda) if hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler(enabled=cuda))
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
                preds = predict(model, val_dl, device)
                rates = {t: 100 * np.mean([r["correct_seen"] for r in score(preds, t)]) for t in THRESHOLDS}
                t_best = max(rates, key=rates.get)
                log.append(dict(epoch=epoch, loss=round(total / max(nb, 1), 4), val_correct_pct=round(rates[t_best], 2),
                                threshold=float(t_best), seconds=round(time.time() - t0, 1)))
                print(f"  epoch {epoch:3d}: loss {log[-1]['loss']:.4f}  validation correct (seen boxes) {rates[t_best]:5.1f}% "
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
        save_variant(out, a, threshold, best, max(log, key=lambda r: r["val_correct_pct"])["epoch"])
        print(f"kept epoch {max(log, key=lambda r: r['val_correct_pct'])['epoch']} "
              f"(validation correct {best:.1f}%, threshold {threshold:.2f})", flush=True)

    # one data loader over all the test sets: a new loader per set leaks open files on macOS
    dirs = expand(a.tests)
    rows, summary = [], []
    if dirs:
        ds = BevDataset(dirs)
        ds.min_visible_px = a.min_visible_px
        by_set, maps, kept = {}, {}, {}
        for heat, off, boxes, run, set_name, seen in predict_iter(model, loader(ds, a.batch, False, a.workers), device):
            by_set.setdefault(set_name, []).append(score_one(heat, off, boxes, run, set_name, threshold, seen=seen))
            if a.save_maps and len(maps.setdefault(set_name, [])) < a.save_maps:
                maps[set_name].append((heat.astype(np.float16), off.astype(np.float16), np.asarray(boxes), run,
                                       geo.decode(heat, off, threshold)))
            if a.diagnose:
                kept.setdefault(set_name, []).append((heat, off, boxes))
        sfx = f"_{a.results_name}" if a.results_name else ""
        for set_name in sorted(by_set):
            r = by_set[set_name]
            rows += r
            summary.append(dict(set=set_name, scenes=len(r),
                                correct_pct=round(100 * np.mean([x["correct"] for x in r]), 2),
                                too_few_pct=round(100 * np.mean([x["dn"] < 0 for x in r]), 2),
                                too_many_pct=round(100 * np.mean([x["dn"] > 0 for x in r]), 2),
                                misplaced_pct=round(100 * np.mean([x["dn"] == 0 and not x["correct"] for x in r]), 2),
                                rmse_cm=round(100 * float(np.nanmean([x["rmse"] for x in r])), 2),
                                solvable_pct=round(100 * np.mean([x["solvable"] for x in r]), 2),
                                correct_solvable_pct=round(100 * np.mean([x["correct"] for x in r if x["solvable"]]), 2)
                                if any(x["solvable"] for x in r) else float("nan")))
            print(f"  {set_name:20s} correct {summary[-1]['correct_pct']:5.1f}%  too few "
                  f"{summary[-1]['too_few_pct']:5.1f}%  too many {summary[-1]['too_many_pct']:5.1f}%  "
                  f"RMSE {summary[-1]['rmse_cm']:.1f} cm  solvable {summary[-1]['solvable_pct']:5.1f}%  "
                  f"correct on solvable {summary[-1]['correct_solvable_pct']:5.1f}%", flush=True)
        write_csv(os.path.join(out, f"test_scenes{sfx}.csv"), rows)
        write_csv(os.path.join(out, f"summary{sfx}.csv"), summary)
        if maps:
            flat = [(s, *m) for s in sorted(maps) for m in maps[s]]
            np.savez_compressed(os.path.join(out, f"maps{sfx}.npz"), set=np.array([f[0] for f in flat]),
                                heat=np.stack([f[1] for f in flat]), off=np.stack([f[2] for f in flat]),
                                truth=np.array([f[3] for f in flat], dtype=object),
                                run=np.array([f[4] for f in flat]), est=np.array([f[5] for f in flat], dtype=object),
                                threshold=threshold)
            print(f"  maps of {len(flat)} scenes: {os.path.join(out, f'maps{sfx}.npz')}")
        if kept:
            print("  correct % for other thresholds (columns) and minimum peak distances (rows):")
            ths = np.round(np.arange(0.2, 0.951, 0.05), 2)
            for set_name in sorted(kept):
                print(f"  {set_name}: " + " ".join(f"{t:5.2f}" for t in ths))
                for md in (0.0, 0.10, 0.15, 0.20, 0.25):
                    c = [100 * np.mean([score_one(h, o, b, "", "", t, min_dist_m=md)["correct"]
                                        for h, o, b in kept[set_name]]) for t in ths]
                    print(f"    {md:4.2f} m " + " ".join(f"{v:5.1f}" for v in c), flush=True)
    if rows:
        print(f"written in {out}")


if __name__ == "__main__":
    main()
