"""Builds every simulated set of the GNN study with fixed seeds, and runs the five classical methods on the
test sets:

  data/sim_train/   general training environment (sim/specs/train_general.json)        --train-runs scenes
  data/sim_val/     same distribution, new seeds, for model selection only               --val-runs scenes
  data/sim_T0/ ... data/sim_T7/   failure-prone test conditions (sim/specs/T*.json)    --test-runs scenes each
  results/baselines_sim/T0/scenes.csv ...   pipeline/evaluate_runs.py on every test set

Every set has its own seeds (training 1, validation 2, T<k> 10 + k, times --seed), so no scene appears in two
sets. Large sets are generated in parts in parallel (--jobs) and merged: each set folder gets one
ground_truth.json and one runs.csv.

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json \\
        --regions config/optitrack_regions.json

--sets chooses what to build (default: train,val,T0,T1,T2,T3,T4,T5,T6,T7); --no-baselines skips the
classical methods. A set that is built again replaces the old one.

Two-circle scenario (--sets circle): the boxes inside an inner circle, the robots evenly on an outer circle
(sim/specs/circle_train.json, circle_test.json):
  sim_Ctrain, sim_Cval        3-10 robots, outer radius 1.2-2 m, randomised sensor
  sim_C0                      nominal test: 6 robots at 1.5 m, inner circle 0.7 m
  sim_Cn3 ... sim_Cn16        the nominal test with 3, 4, 8, 12 or 16 robots
"""

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from argparse import Namespace

from sim.generate import random_runs
from sim.model import SensorModel

TESTS = {"T0": "T0_nominal", "T1": "T1_close", "T2": "T2_dense", "T3": "T3_obstacles", "T4": "T4_false",
         "T5": "T5_two_faces", "T6": "T6_unseen_count", "T7": "T7_many_robots"}
PART = 2500                                   # runs per parallel part

# two-circle scenario (targets in an inner circle, robots evenly on an outer circle):
# name -> (spec file, overrides of the spec, seed, role)
CIRCLE = {
    "Ctrain": ("circle_train", {}, 3, "train"),
    "Cval": ("circle_train", {}, 4, "val"),
    "C0": ("circle_test", {}, 30, "test"),
    "Cn3": ("circle_test", {"n_robots": [3]}, 31, "test"),
    "Cn4": ("circle_test", {"n_robots": [4]}, 32, "test"),
    "Cn8": ("circle_test", {"n_robots": [8]}, 33, "test"),
    "Cn12": ("circle_test", {"n_robots": [12]}, 34, "test"),
    "Cn16": ("circle_test", {"n_robots": [16]}, 35, "test"),
}


def set_plan(name, a):
    """(spec path, number of runs, seed, layout name) of one set."""
    if name == "train":
        return os.path.join(a.spec_dir, "train_general.json"), a.train_runs, 1, "TRAIN"
    if name == "val":
        return os.path.join(a.spec_dir, "train_general.json"), a.val_runs, 2, "VAL"
    if name in TESTS:
        return os.path.join(a.spec_dir, TESTS[name] + ".json"), a.test_runs, 10 + int(name[1:]), name
    if name in CIRCLE:
        spec, _, seed, role = CIRCLE[name]
        runs = dict(train=a.train_runs, val=a.val_runs).get(role, a.test_runs)
        return os.path.join(a.spec_dir, spec + ".json"), runs, seed, name.upper() if role != "test" else name
    raise SystemExit(f"unknown set {name}")


def expand_sets(text):
    """'circle' stands for every set of the two-circle scenario."""
    out = []
    for s in [x.strip() for x in text.split(",") if x.strip()]:
        out += list(CIRCLE) if s == "circle" else [s]
    return out


def generate_part(task):
    spec, runs, start, seed, layout, part_dir, model_path = task
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        random_runs(Namespace(spec=spec, runs=runs, start=start, seed=seed, name=layout, out=part_dir,
                              runs_per_folder=500), SensorModel.load(model_path))
    return part_dir


def merge_parts(set_dir, part_dirs):
    truth, rows = {}, []
    for d in sorted(part_dirs):
        with open(os.path.join(d, "ground_truth.json")) as f:
            truth.update(json.load(f))
        with open(os.path.join(d, "runs.csv")) as f:
            rows += list(csv.DictReader(f))
        os.remove(os.path.join(d, "ground_truth.json"))
        os.remove(os.path.join(d, "runs.csv"))
    with open(os.path.join(set_dir, "ground_truth.json"), "w") as f:
        json.dump(truth, f)
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(set_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["run"]))


def baseline(task):
    name, set_dir, out, sensor_calib, regions = task
    cmd = [sys.executable, "-m", "pipeline.evaluate_runs", "--survey-dir", set_dir,
           "--ground-truth", os.path.join(set_dir, "ground_truth.json"), "--sensor-calib", sensor_calib,
           "--full-only", "--quiet", "--out", out]
    if regions:
        cmd += ["--regions", regions]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="sensor model JSON (sim/calibrate.py)")
    ap.add_argument("--sensor-calib", required=True)
    ap.add_argument("--regions", default=None, help="arena regions, for zone+class")
    ap.add_argument("--spec-dir", default="sim/specs/")
    ap.add_argument("--sets", default="train,val," + ",".join(TESTS))
    ap.add_argument("--train-runs", type=int, default=50000)
    ap.add_argument("--val-runs", type=int, default=2000)
    ap.add_argument("--test-runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--baseline-dir", default="results/baselines_sim/")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes in parallel (default: CPU cores - 1; at most 60 on Windows)")
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)             # Windows cannot wait on more than 63 process handles

    sets = expand_sets(a.sets)
    tasks, parts = [], {}
    for name in sets:
        spec, runs, seed, layout = set_plan(name, a)
        set_dir = os.path.join(a.data_dir, f"sim_{name}")
        if os.path.isdir(set_dir):
            shutil.rmtree(set_dir)
        os.makedirs(set_dir)
        shutil.copy(a.model, os.path.join(set_dir, "sensor_model_used.json"))
        with open(spec) as f:
            spec_data = json.load(f)
        if name in CIRCLE:
            spec_data.update(CIRCLE[name][1])
        spec = os.path.join(set_dir, "spec.json")            # the spec actually used, kept with the set
        with open(spec, "w") as f:
            json.dump(spec_data, f, indent=2)
        for k, start in enumerate(range(0, runs, PART)):
            part_dir = os.path.join(set_dir, f"part{k:02d}")
            tasks.append((spec, min(PART, runs - start), start, a.seed * 100 + seed, layout, part_dir, a.model))
            parts.setdefault(name, []).append(part_dir)
    print(f"generating {', '.join(sets)}: {len(tasks)} parts on {min(a.jobs, len(tasks))} process(es)", flush=True)
    with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
        for done in pool.imap_unordered(generate_part, tasks):
            print(f"  {done}", flush=True)
    for name in sets:
        merge_parts(os.path.join(a.data_dir, f"sim_{name}"), parts[name])
        n = sum(1 for _ in open(os.path.join(a.data_dir, f"sim_{name}", "runs.csv"))) - 1
        print(f"  sim_{name}: {n} runs", flush=True)

    tests = [s for s in sets if s in TESTS or (s in CIRCLE and CIRCLE[s][3] == "test")]
    if tests and not a.no_baselines:
        print(f"classical methods on {', '.join(tests)}", flush=True)
        jobs = [(t, os.path.join(a.data_dir, f"sim_{t}"), os.path.join(a.baseline_dir, t), a.sensor_calib, a.regions)
                for t in tests]
        with multiprocessing.Pool(min(a.jobs, len(jobs))) as pool:
            for done in pool.imap_unordered(baseline, jobs):
                print(f"  {done}: {os.path.join(a.baseline_dir, done, 'scenes.csv')}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
