"""Training of ImageMLP + GNN in simulation (end to end), model selection on a simulated validation set, evaluation
on simulated test sets and on the real scenes. No test scene is used for training or for choosing anything.

    python -m imgmlp.train --train data/plan_v2/train data/imgmlp/plan_v2_train \\
        --val data/plan_v2/val data/imgmlp/plan_v2_val \\
        --real data/survey_all/ data/imgmlp/real --sensor-calib config/sensor_calib.json \\
        --measure noisy --geometry fixed --k 8 --img-size 64 36 --seeds 3 --out results/imgmlp/

A set is SET_DIR CACHE: the set folder (geometry.jsonl, and P<k>.jsonl for --measure sensor) and its frame cache
(imgmlp/frames.py). --test NAME SET_DIR CACHE adds a simulated test set (same measurements as training).

Measurements (--measure, imgmlp/measure.py): ideal | noisy (NoiseModel, --noise-* to change it) | noisyrand (every
scene draws its own error from NoiseRanges, --rand-* LO HI to change them: no sensor model) | sensor (the detections
of the set, corrected with --sensor-calib). The real scenes use the ArUco detections corrected with --sensor-calib
(test "real"); --real-raw adds them without any calibration (test "real_raw": rho and theta as the detector gives
them, i.e. no sensor model at all).

--unc gives every report node the uncertainty its sensor declares (imgmlp/model.py). In simulation it is the error
the scene was drawn with; on the real scenes it is given by hand with --declare NAME FIELD=VALUE ... (fields of
NoiseModel, the others keep their default), one real test per NAME (real_NAME, real_raw_NAME).

Graph (--graph, imgmlp/model.py): A (z appended to every report node) | B (one robot node per robot carrying z,
linked to its reports and to the robots closer than --robot-gate; robots that measured nothing keep their node).

Variant folder: <measure>_<geometry>[_B]_k<k>_<W>x<H>[_unc][_<tag>]. Outputs: model_seed<s>.pt, variant.json, train_log.csv,
summary.csv (per test set: mean and sd over the seeds), <test>_scenes.csv (one row per scene and seed: layout,
round, seed, correct, dn, ... as tools/real_tables.py reads them), and with k > 0 features_<test>_seed<s>.npy (z of
every frame of the cache, rows as frames.csv) for imgmlp/probe.py.
"""

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, fields

import numpy as np
import torch

from gnn.model import decode
from imgmlp.frames import load_cache
from imgmlp.measure import NoiseModel, NoiseRanges, add_silent_robots, attach_frames, measured_scene, sensor_scenes
from imgmlp.model import ImageDetectionGNN, collate, encode_scene, loss_fn, to_world
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics


def variant_name(a):
    graph = "_B" if a.graph == "B" else ""
    name = f"{a.measure}_{a.geometry}{graph}_k{a.k}_{a.img_size[0]}x{a.img_size[1]}"
    return name + ("_unc" if a.unc else "") + (f"_{a.tag}" if a.tag else "")


def noise_model(a):
    if a.measure == "ideal":
        return NoiseModel.ideal()
    if a.measure == "noisyrand":
        return NoiseRanges(**{f.name: tuple(getattr(a, "rand_" + f.name)) for f in fields(NoiseRanges)})
    return NoiseModel(**{f.name: getattr(a, "noise_" + f.name) for f in fields(NoiseModel)})


def sim_set(set_dir, cache, a, seed, max_scenes=None):
    """(scenes, images) of a simulated set: the scenes whose frames are in the cache."""
    from sim.scaling import read_geometry
    images, index, rows = load_cache(cache, a.img_size)
    runs = {r for r, _, _ in index}
    if a.measure == "sensor":
        scenes = [s for s in sensor_scenes(set_dir, os.path.join(set_dir, "ground_truth.json"), a.sensor_calib)
                  if s["run"] in runs]
    else:
        rng = np.random.default_rng(seed)
        noise = noise_model(a)
        scenes = [measured_scene(g, noise, rng) for g in read_geometry(set_dir).values() if g["run"] in runs]
    scenes = scenes[:max_scenes]
    if a.graph == "B":
        added = add_silent_robots(scenes, rows)
        if added:
            print(f"  {added} robots without reports added as robot nodes ({cache})")
    missing = attach_frames(scenes, index)
    if missing:
        print(f"  warning: {missing} robots of {set_dir} have no frame in {cache} (black image)")
    return scenes, images


def declared_model(assignments):
    """NoiseModel from ["field=value", ...] (the others keep their default)."""
    names = {f.name for f in fields(NoiseModel)}
    kw = {}
    for item in assignments:
        key, value = item.split("=")
        if key not in names:
            raise SystemExit(f"--declare: unknown field {key} (fields: {sorted(names)})")
        kw[key] = float(value)
    return NoiseModel(**kw)


def real_set(survey_dir, cache, a, calib, declared=None):
    images, index, rows = load_cache(cache, a.img_size)
    scenes = sensor_scenes(survey_dir, a.ground_truth, calib)
    for s in scenes:
        s["declared"] = declared
    if a.graph == "B":
        added = add_silent_robots(scenes, rows)
        if added:
            print(f"  {added} robots without reports added as robot nodes ({cache})")
    missing = attach_frames(scenes, index)
    if missing:
        print(f"  warning: {missing} real robots have no frame in {cache} (black image)")
    return scenes, images


def encode_all(scenes, a, rng=None):
    rot = (lambda: float(rng.uniform(0, 2 * math.pi))) if rng is not None else (lambda: 0.0)
    return [encode_scene(s, a.geometry, a.gate, rot(), a.robot_gate, a.unc) for s in scenes]


def class_weights(enc):
    ps, ns = sum(e["same"].sum() for e in enc), sum(len(e["same"]) for e in enc)
    pf, nf = sum(e["false"].sum() for e in enc), sum(len(e["false"]) for e in enc)
    return min(10.0, (ns - ps) / max(ps, 1)), min(10.0, (nf - pf) / max(pf, 1))


def predict(model, enc, images, batch=64):
    """(p_same, p_false, positions in world [m]) per scene."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(enc), batch):
            chunk = enc[start:start + batch]
            same, false, pos = model(collate(chunk, images if model.k > 0 else None))
            same, false, pos = torch.sigmoid(same).numpy(), torch.sigmoid(false).numpy(), pos.numpy()
            ps = np.cumsum([0] + [len(e["pairs"]) for e in chunk])
            pd = np.cumsum([0] + [len(e["node"]) for e in chunk])
            for k, e in enumerate(chunk):
                out.append((same[ps[k]:ps[k + 1]], false[pd[k]:pd[k + 1]], to_world(pos[pd[k]:pd[k + 1]], e)))
    return out


def evaluate(model, scenes, enc, images, a):
    rows = []
    for s, e, (p_same, p_false, pos) in zip(scenes, enc, predict(model, enc, images)):
        est = decode(pos, e["pairs"], p_same, p_false)[0] if len(pos) else []
        r = scene_metrics(est, list(s["truth"]), a.cutoff, 2.0)
        r["correct"] = int(r["dn"] == 0 and r["n_matched"] == r["n_true"])
        r.update(run=s["run"], layout=s["layout"], round=int(s["round"]), n_robots=len(s["robots"]),
                 n_meas=len(s["det_robot"]))
        rows.append(r)
    return rows


def train_one(train, val, a, seed, log):
    (tr_enc, tr_img), (va_scenes, va_enc, va_img) = train, val
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ImageDetectionGNN(a.k, a.img_size, a.geometry, a.hidden, a.hops, graph=a.graph, unc=a.unc)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    w_same, w_false = class_weights(tr_enc)
    best, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, a.epochs + 1):
        t0, total, nb = time.time(), 0.0, 0
        model.train()
        order = rng.permutation(len(tr_enc))
        for start in range(0, len(order), a.batch):
            b = collate([tr_enc[i] for i in order[start:start + a.batch]], tr_img if a.k > 0 else None)
            loss = loss_fn(model(b), b, w_same, w_false, a.w_pos)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1
        rows = evaluate(model, va_scenes, va_enc, va_img, a)
        correct = 100 * float(np.mean([r["correct"] for r in rows]))
        log.append(dict(seed=seed, epoch=epoch, loss=round(total / max(nb, 1), 5), val_correct_pct=round(correct, 2),
                        val_exact_pct=round(100 * float(np.mean([r["dn"] == 0 for r in rows])), 2),
                        seconds=round(time.time() - t0, 1)))
        print(f"    seed {seed} epoch {epoch:3d}: loss {log[-1]['loss']:.4f}  validation correct {correct:5.1f}%  "
              f"exact {log[-1]['val_exact_pct']:5.1f}%  ({log[-1]['seconds']:.0f} s)", flush=True)
        if correct > best:
            best, best_state, best_epoch, waited = correct, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= a.patience:
                break
    model.load_state_dict(best_state)
    print(f"    seed {seed}: kept epoch {best_epoch} (validation correct {best:.1f}%)", flush=True)
    return model


def dump_features(model, images, path, batch=512):
    model.eval()
    with torch.no_grad():
        z = [model.image(torch.as_tensor(images[i:i + batch])).numpy() for i in range(0, len(images), batch)]
    np.save(path, np.concatenate(z).astype(np.float32))


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
    ap.add_argument("--train", nargs=2, metavar=("SET_DIR", "CACHE"))
    ap.add_argument("--val", nargs=2, metavar=("SET_DIR", "CACHE"))
    ap.add_argument("--test", nargs=3, action="append", default=[], metavar=("NAME", "SET_DIR", "CACHE"))
    ap.add_argument("--real", nargs=2, metavar=("SURVEY_DIR", "CACHE"), help="the real scenes and their frame cache")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--measure", choices=["ideal", "noisy", "noisyrand", "sensor"], default="noisy")
    for f in fields(NoiseModel):
        ap.add_argument("--noise-" + f.name.replace("_", "-"), dest="noise_" + f.name, type=float, default=f.default)
    for f in fields(NoiseRanges):
        ap.add_argument("--rand-" + f.name.replace("_", "-"), dest="rand_" + f.name, type=float, nargs=2,
                        default=list(f.default), metavar=("LO", "HI"))
    ap.add_argument("--unc", action="store_true", help="declared uncertainty of the sensor on every report node")
    ap.add_argument("--declare", nargs="+", action="append", default=[], metavar=("NAME", "FIELD=VALUE"),
                    help="with --unc: uncertainty declared on the real scenes (repeatable)")
    ap.add_argument("--real-raw", action="store_true", help="also test on the real scenes without calibration")
    ap.add_argument("--tag", default="", help="suffix of the variant folder")
    ap.add_argument("--geometry", choices=["fixed", "strict"], default="fixed")
    ap.add_argument("--graph", choices=["A", "B"], default="A",
                    help="A: z on every report node; B: robot nodes carrying z (heterogeneous graph)")
    ap.add_argument("--robot-gate", type=float, default=4.0,
                    help="graph B, fixed geometry: robots farther apart exchange no messages [m]")
    ap.add_argument("--k", type=int, default=8, help="image features (0: no image)")
    ap.add_argument("--img-size", type=int, nargs=2, default=[64, 36], metavar=("W", "H"))
    ap.add_argument("--gate", type=float, default=1.0, help="fixed geometry: pairs farther apart are not linked [m]")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--w-pos", type=float, default=1.0, help="weight of the box-position loss")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--cutoff", type=float, default=MATCH_CUTOFF_M)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="results/imgmlp/")
    a = ap.parse_args()
    if a.unc and a.measure in ("sensor", "ideal"):
        raise SystemExit("--unc needs --measure noisy or noisyrand (the declared error of the simulated scenes)")
    out = os.path.join(a.out, variant_name(a))
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"variant {variant_name(a)} -> {out}", flush=True)

    models = []
    if a.eval_only:
        for seed in range(a.seeds):
            m = ImageDetectionGNN(a.k, a.img_size, a.geometry, a.hidden, a.hops, graph=a.graph, unc=a.unc)
            m.load_state_dict(torch.load(os.path.join(out, f"model_seed{seed}.pt")))
            models.append(m)
    else:
        t0 = time.time()
        tr_scenes, tr_img = sim_set(*a.train, a, seed=101, max_scenes=a.max_train)
        va_scenes, va_img = sim_set(*a.val, a, seed=102, max_scenes=a.max_val)
        tr_enc = encode_all(tr_scenes, a, np.random.default_rng(103))
        va_enc = encode_all(va_scenes, a)
        print(f"{len(tr_scenes)} training scenes, {len(va_scenes)} validation scenes, images {tr_img.shape[1:]} "
              f"({time.time() - t0:.0f} s to load)", flush=True)
        log = []
        for seed in range(a.seeds):
            m = train_one((tr_enc, tr_img), (va_scenes, va_enc, va_img), a, seed, log)
            torch.save(m.state_dict(), os.path.join(out, f"model_seed{seed}.pt"))
            models.append(m)
            write_csv(os.path.join(out, "train_log.csv"), log)
        cfg = {k: v for k, v in vars(a).items() if k not in ("test", "eval_only")}
        cfg["noise"] = asdict(noise_model(a)) if a.measure != "sensor" else None
        with open(os.path.join(out, "variant.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    tests = [(name, sim_set(d, c, a, seed=200 + i)) for i, (name, d, c) in enumerate(a.test)]
    if a.val:
        tests.insert(0, ("val", sim_set(*a.val, a, seed=102, max_scenes=a.max_val)))
    if a.real:
        declares = [(d[0], declared_model(d[1:])) for d in a.declare] if a.unc else [("", None)]
        if a.unc and not declares:
            declares = [("", NoiseModel())]
        for tag, dec in declares:
            suffix = f"_{tag}" if tag else ""
            tests.append(("real" + suffix, real_set(*a.real, a, a.sensor_calib, dec)))
            if a.real_raw:
                tests.append(("real_raw" + suffix, real_set(*a.real, a, None, dec)))
    summary = []
    for name, (scenes, images) in tests:
        enc = encode_all(scenes, a)
        rows_all, per_seed = [], []
        for seed, m in enumerate(models):
            rows = evaluate(m, scenes, enc, images, a)
            for r in rows:
                r["seed"] = seed
            rows_all += rows
            per_seed.append(100 * float(np.mean([r["correct"] for r in rows])))
            if a.k > 0:
                dump_features(m, images, os.path.join(out, f"features_{name}_seed{seed}.npy"))
        write_csv(os.path.join(out, f"{name}_scenes.csv"), rows_all)
        dn = np.array([r["dn"] for r in rows_all])
        summary.append(dict(test=name, scenes=len(scenes), correct_pct=round(float(np.mean(per_seed)), 2),
                            correct_sd=round(float(np.std(per_seed)), 2),
                            exact_pct=round(100 * float(np.mean(dn == 0)), 2),
                            too_few_pct=round(100 * float(np.mean(dn < 0)), 2),
                            too_many_pct=round(100 * float(np.mean(dn > 0)), 2)))
        s = summary[-1]
        print(f"  {name:<8} {s['scenes']:5d} scenes: correct {s['correct_pct']:5.1f} ± {s['correct_sd']:.1f}%  "
              f"(exact {s['exact_pct']:.1f}%, too few {s['too_few_pct']:.1f}%, too many {s['too_many_pct']:.1f}%)",
              flush=True)
    path = os.path.join(out, "summary.csv")
    if a.eval_only and os.path.exists(path):                    # keep the tests not evaluated again
        with open(path) as f:
            done = {r["test"] for r in summary}
            summary = [r for r in csv.DictReader(f) if r["test"] not in done] + summary
    write_csv(path, summary)
    print(f"written in {out}")


if __name__ == "__main__":
    main()
