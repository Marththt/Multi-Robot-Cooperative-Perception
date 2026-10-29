"""Generates simulated runs in the format of the real runs (one folder per run, P<k>.jsonl + ground_truth.json,
plus a combined ground_truth.json in --out for pipeline/evaluate_runs.py / gnn/train.py).

Two modes:

  replay   recreates real layouts: OptiTrack box positions, box orientations estimated from the real
           detections, and the real camera pose of every robot in every round. Each repetition is one
           simulated run of the same layout with new noise and new robot biases.

    python -m sim.generate replay --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --sensor-calib config/sensor_calib.json --model config/sensor_model.json \\
        --layouts L4,L6,L8,L9,L11 --reps 200 --out data/sim_replay/

  random   new layouts from a scenario spec (JSON with the fields of sim.scene.ScenarioSpec); every run
           has its own seed, recorded in runs.csv

    python -m sim.generate random --model config/sensor_model.json --spec sim/specs/train_general.json \\
        --runs 1000 --name TRAIN --seed 1 --out data/sim_train/

  --runs-per-folder N writes N runs in each folder instead of one folder per run (every record keeps its
  run_id), for large sets such as the 50,000-scene training set.

--set key=value overrides a field of the sensor model for this generation only (e.g. --set false_rate=0.4),
which is how the failure-prone test conditions raise the rate of false or two-face detections.
"""

import argparse
import copy
import csv
import json
import os
from collections import defaultdict
from dataclasses import fields

import numpy as np

from sim.model import SensorModel
from sim.scene import ScenarioSpec, random_scene, scene_from_real
from sim.sensor import capture, robot_bias
from sim.writer import write_run


def override(model, settings):
    for item in settings or []:
        key, value = item.split("=", 1)
        if key not in model.__dataclass_fields__:
            raise SystemExit(f"unknown sensor model field: {key}")
        setattr(model, key, json.loads(value))
    return model


def simulate_scene(scene, model, rng):
    biases = {}
    detections = {}
    for ci, cap in enumerate(scene.captures):
        if cap.robot not in biases:
            biases[cap.robot] = robot_bias(model, rng)
        detections[ci] = capture(scene, cap, model, rng, biases[cap.robot])
    return detections


def save_truth(out, truth):
    path = os.path.join(out, "ground_truth.json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    old.update(truth)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


def replay(a, model):
    from gnn.dataset import build_scenes
    scenes = build_scenes(a.survey_dir, a.ground_truth, a.sensor_calib)
    wanted = [l.strip() for l in a.layouts.split(",") if l.strip()]
    by_run = defaultdict(list)
    for s in scenes:
        if s["layout"] in wanted:
            by_run[s["run"]].append(s)
    rng = np.random.default_rng(a.seed)
    truth = {}
    for run, rs in sorted(by_run.items()):
        base = scene_from_real(rs, rng)
        for rep in range(a.reps):
            base.run_id = f"{base.layout_id}_sim{rep:04d}"
            truth.update(write_run(a.out, base, simulate_scene(base, model, rng), model))
        print(f"  {base.layout_id}: {a.reps} simulated runs of {len(base.captures)} captures", flush=True)
    save_truth(a.out, truth)


NOISE_FIELDS = ("along_sd", "across_sd", "bias_sd_along_rel", "bias_sd_across_rel", "view_noise_sd_rad")


def jittered(model, spec, rng):
    """A copy of the sensor model with the noise and the rare-event rates drawn for one run."""
    m = copy.deepcopy(model)
    f_noise = rng.uniform(1 - spec.noise_jitter, 1 + spec.noise_jitter)
    for name in NOISE_FIELDS:
        v = getattr(m, name)
        setattr(m, name, [x * f_noise for x in v] if isinstance(v, list) else v * f_noise)
    f_false = rng.uniform(*spec.false_factor)
    f_two = rng.uniform(*spec.two_face_factor)
    m.false_rate *= f_false
    m.two_face_q *= f_two
    factors = dict(noise_factor=round(float(f_noise), 3), false_factor=round(float(f_false), 3),
                   two_face_factor=round(float(f_two), 3))
    if spec.sensing_range_m:                 # drawn last, so the other draws stay as before
        m.r_max_m = float(rng.uniform(*spec.sensing_range_m))
        m.false_within_range = True
        factors["sensing_range_m"] = round(m.r_max_m, 3)
    return m, factors


def random_runs(a, model):
    with open(a.spec) as f:
        raw = json.load(f)
    names = {fl.name for fl in fields(ScenarioSpec)}
    spec = ScenarioSpec(**{k: v for k, v in raw.items() if k in names})
    truth = {}
    rows = []
    per_folder = max(1, getattr(a, "runs_per_folder", 1))
    start = getattr(a, "start", 0)          # first run index (sim.make_sets splits large sets into parts)
    for i in range(start, start + a.runs):
        seed = a.seed * 1_000_003 + i
        rng = np.random.default_rng(seed)
        scene = random_scene(spec, rng, layout_id=a.name, run_id=f"{a.name}_{i:05d}")
        run_model, factors = jittered(model, spec, rng)
        folder = f"{a.name}_part{i // per_folder:04d}" if per_folder > 1 else None
        truth.update(write_run(a.out, scene, simulate_scene(scene, run_model, rng), run_model, folder=folder))
        rows.append(dict(run=scene.run_id, seed=seed, n_boxes=len(scene.boxes), **scene.meta, **factors))
    save_truth(a.out, truth)
    with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {a.runs} runs of {a.name} written in {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("replay", "random"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="sensor model JSON (sim/calibrate.py)")
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    r = sub.choices["replay"]
    r.add_argument("--survey-dir", required=True)
    r.add_argument("--ground-truth", required=True)
    r.add_argument("--sensor-calib", required=True)
    r.add_argument("--layouts", required=True)
    r.add_argument("--reps", type=int, default=200)
    g = sub.choices["random"]
    g.add_argument("--spec", required=True)
    g.add_argument("--runs", type=int, default=100)
    g.add_argument("--name", default="SIM", help="layout name of these runs (used to group the results)")
    g.add_argument("--runs-per-folder", type=int, default=1,
                   help="runs written in one folder (e.g. 500 for a training set: far fewer files)")
    a = ap.parse_args()

    model = override(SensorModel.load(a.model), a.set)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sensor_model_used.json"), "w") as f:
        json.dump({k: getattr(model, k) for k in model.__dataclass_fields__}, f, indent=2)
    replay(a, model) if a.mode == "replay" else random_runs(a, model)


if __name__ == "__main__":
    main()
