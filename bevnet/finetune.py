"""Fine-tuning of a simulation-trained image network on the real frames, leave-one-layout-out.

For each of the 8 real layouts: start from the simulation model (--init), train on the real scenes of the other
7 layouts (all readings, --epochs, small learning rate, colour jitter), optionally mixed with --sim-scenes
simulated scenes so it does not forget the simulation, choose the detection threshold on those training scenes,
and test on the scenes of the held-out layout. The same held-out scenes are also scored with the simulation
model as it is (zero-shot). No scene of a layout is ever used to train the model that is tested on it.

Outputs in --out: folds.csv (per layout: scenes, zero-shot and fine-tuned correct %, threshold),
test_scenes.csv (per held-out scene and reading), summary.csv.

    python -m bevnet.finetune --init results/bev_obst/bev_c64_m16 --real "data/bev_real/real_k*" --out results/bev_real_ft/
    python -m bevnet.finetune ... --sim data/bev_obst/train --sim-scenes 500
    python -m bevnet.finetune ... --layouts L10,L11          # only these folds; the others in folds.csv are kept
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

from bevnet import geometry as geo
from bevnet.data import BevDataset, collate
from bevnet.model import BevNet, losses
from bevnet.train import THRESHOLDS, predict, score, to, write_csv


def make_loader(ds, batch, shuffle, workers):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                       collate_fn=collate, persistent_workers=False)


def train_fold(model, train_ds, a, device):
    dl = make_loader(train_ds, a.batch, True, a.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl))
    for epoch in range(a.epochs):
        model.train()
        for b in dl:
            b = to(b, device)
            img = b["images"]
            n = img.shape[0]
            gain = 1 + a.jitter * (torch.rand(n, 3, 1, 1, device=device) - 0.5)       # colour and brightness
            bias = 0.5 * a.jitter * (torch.rand(n, 1, 1, 1, device=device) - 0.5)
            img = (img * gain + bias).clamp(0, 1)
            heat, off = model(img, b["grids"], b["valid"], b["view"], b["scene"], b["n_scenes"])
            focal, l1 = losses(heat.float(), off.float(), b["heat"], b["off"], b["mask"])
            loss = focal + l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()


def best_threshold(model, ds, a, device):
    preds = predict(model, make_loader(ds, a.batch, False, a.workers), device)
    rates = {t: np.mean([r["correct"] for r in score(preds, t)]) for t in THRESHOLDS}
    return float(max(rates, key=rates.get))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="variant folder of bevnet.train (model.pt, variant.json)")
    ap.add_argument("--real", nargs="+", default=["data/bev_real/real_k*"])
    ap.add_argument("--sim", nargs="*", default=[], help="simulated set folders mixed into every fold's training")
    ap.add_argument("--sim-scenes", type=int, default=0)
    ap.add_argument("--out", default="results/bev_real_ft/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=0, help="loader processes (0: macOS runs out of open files)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layouts", default=None, help="only these held-out layouts, e.g. L4,L9")
    a = ap.parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    with open(os.path.join(a.init, "variant.json")) as f:
        v = json.load(f)
    state = torch.load(os.path.join(a.init, "model.pt"), map_location="cpu")
    sim_threshold = v["threshold"]
    dirs = sorted({d for p in a.real for d in glob.glob(p) if os.path.isdir(d)})
    real = BevDataset(dirs)
    layout_of = np.array([real.scene(i)["run"].split("_")[0] for i in range(len(real))])
    sim = None
    if a.sim and a.sim_scenes:
        sim = BevDataset(sorted({d for p in a.sim for d in glob.glob(p)}), a.sim_scenes)
    layouts = sorted(set(layout_of), key=lambda s: int(s[1:]) if s[1:].isdigit() else s)
    if a.layouts:
        layouts = [l for l in layouts if l in a.layouts.split(",")]
    os.makedirs(a.out, exist_ok=True)
    folds, rows = [], []                       # layouts done in an earlier call (--layouts) are kept
    if os.path.exists(os.path.join(a.out, "folds.csv")):
        with open(os.path.join(a.out, "folds.csv"), newline="") as f:
            folds = [r for r in csv.DictReader(f) if r["layout"] not in layouts]
        with open(os.path.join(a.out, "test_scenes.csv"), newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["layout"] not in layouts]
    print(f"device {device}; {len(real)} real scenes ({len(dirs)} readings), layouts {', '.join(layouts)}"
          + (f"; + {len(sim)} simulated scenes per fold" if sim else ""), flush=True)
    for lay in layouts:
        t0 = time.time()
        torch.manual_seed(a.seed)
        test_idx = np.nonzero(layout_of == lay)[0].tolist()
        train_idx = np.nonzero(layout_of != lay)[0].tolist()
        test_ds = torch.utils.data.Subset(real, test_idx)
        train_real = torch.utils.data.Subset(real, train_idx)
        train_ds = torch.utils.data.ConcatDataset([train_real, sim]) if sim else train_real
        model = BevNet(v["channels"], v["message"], len(geo.HEIGHTS_M), v.get("encoder", "small"), pretrained=False).to(device)
        model.load_state_dict(state)
        zero = score(predict(model, make_loader(test_ds, a.batch, False, a.workers), device), sim_threshold)
        train_fold(model, train_ds, a, device)
        th = best_threshold(model, train_real, a, device)
        tuned = score(predict(model, make_loader(test_ds, a.batch, False, a.workers), device), th)
        for z, r in zip(zero, tuned):
            rows.append(dict(r, layout=lay, zero_shot_correct=z["correct"], zero_shot_n_est=z["n_est"]))
        folds.append(dict(layout=lay, scenes=len(test_idx), threshold=th,
                          zero_shot_correct_pct=round(100 * np.mean([z["correct"] for z in zero]), 2),
                          finetuned_correct_pct=round(100 * np.mean([r["correct"] for r in tuned]), 2),
                          finetuned_too_few_pct=round(100 * np.mean([r["dn"] < 0 for r in tuned]), 2),
                          finetuned_too_many_pct=round(100 * np.mean([r["dn"] > 0 for r in tuned]), 2),
                          finetuned_rmse_cm=round(100 * float(np.nanmean([r["rmse"] for r in tuned])), 2)))
        f = folds[-1]
        print(f"  {lay:4s} zero-shot {f['zero_shot_correct_pct']:5.1f}%   fine-tuned {f['finetuned_correct_pct']:5.1f}%  "
              f"(too few {f['finetuned_too_few_pct']:4.1f}%, too many {f['finetuned_too_many_pct']:4.1f}%, "
              f"RMSE {f['finetuned_rmse_cm']:.1f} cm, threshold {th:.2f})  ({time.time() - t0:.0f} s)", flush=True)
        write_csv(os.path.join(a.out, "folds.csv"), folds)
        write_csv(os.path.join(a.out, "test_scenes.csv"), rows)
    num = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan
    summary = dict(scenes=len(rows), layouts=len({r["layout"] for r in rows}),
                   zero_shot_correct_pct=round(100 * np.mean([num(r, "zero_shot_correct") for r in rows]), 2),
                   finetuned_correct_pct=round(100 * np.mean([num(r, "correct") for r in rows]), 2),
                   finetuned_too_few_pct=round(100 * np.mean([num(r, "dn") < 0 for r in rows]), 2),
                   finetuned_too_many_pct=round(100 * np.mean([num(r, "dn") > 0 for r in rows]), 2),
                   finetuned_rmse_cm=round(100 * float(np.nanmean([num(r, "rmse") for r in rows])), 2),
                   init=a.init, sim_scenes=len(sim) if sim else 0, epochs=a.epochs, lr=a.lr)
    write_csv(os.path.join(a.out, "summary.csv"), [summary])
    print(f"all layouts: zero-shot {summary['zero_shot_correct_pct']:.1f}%, fine-tuned {summary['finetuned_correct_pct']:.1f}% "
          f"(RMSE {summary['finetuned_rmse_cm']:.1f} cm); written in {a.out}")


if __name__ == "__main__":
    main()
