"""Real scenes with more viewpoints: the rounds of one real run are combined into one scene.

The boxes do not move while the robots rotate from round to round, so the captures of r rounds of one run
are captures of the same boxes from 4 x r different positions. Every combination of r of the 7 rounds of a
run becomes one scene, in which each (robot, round) capture is a viewpoint of its own (robot id "P0@r3").
Real noise, real false detections, real occlusions; only the number of viewpoints changes:

  rounds_1   4 viewpoints   the 56 original scenes (7 per run), as a check
  rounds_2   8 viewpoints   21 combinations per run
  ...
  rounds_7   28 viewpoints  1 combination per run (8 scenes)

Each folder data/real_combined/rounds_<r>/ holds the scenes in the format of the real runs, with its
ground_truth.json, so pipeline/evaluate_runs.py and gnn/train_sim.py read them unchanged. The five
classical methods are then run on every folder (results/baselines_real_combined/rounds_<r>/scenes.csv).

    python -m pipeline.combine_rounds --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json

and the trained GNN models are evaluated on them with
    python -m gnn.train_sim ... --combined-tests --eval-only

The scenes of one run share the same boxes and overlap in rounds, so they are not independent: the rates
describe the trend with the number of viewpoints, and a paired test on them overstates the significance.
"""

import argparse
import glob
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from itertools import combinations


def read_runs(survey_dir):
    """{run id: {round: [records]}} of the round-mode runs (same files as pipeline/build_scans.load_points)."""
    runs = defaultdict(lambda: defaultdict(list))
    paths = sorted(glob.glob(os.path.join(survey_dir, "**", "*.jsonl"), recursive=True))
    paths = [q for q in paths
             if not {"raw", "sync"} & set(os.path.normpath(os.path.relpath(q, survey_dir)).split(os.sep)[:-1])]
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "layout_id" not in r or r.get("round") is None or r.get("yaw_offset_deg", 0) != 0:
                    continue
                runs[r.get("run_id") or r["layout_id"]][int(r["round"])].append(r)
    return runs


def build(a):
    with open(a.ground_truth) as f:
        truth = json.load(f)
    runs = read_runs(a.survey_dir)
    rng = random.Random(a.seed)
    if os.path.isdir(a.out_data):
        shutil.rmtree(a.out_data)
    counts = {}
    for r in range(1, a.max_rounds + 1):
        out_dir = os.path.join(a.out_data, f"rounds_{r}")
        os.makedirs(out_dir)
        gt, n_scenes = {}, 0
        with open(os.path.join(out_dir, "combined.jsonl"), "w") as f:
            for run, by_round in sorted(runs.items()):
                layout = next(iter(by_round.values()))[0]["layout_id"]
                boxes = truth.get(run, truth.get(layout))
                if boxes is None:
                    continue
                combos = list(combinations(sorted(by_round), r))
                if len(combos) > a.max_combos:
                    combos = sorted(rng.sample(combos, a.max_combos))
                for combo in combos:
                    scene_id = f"{run}_c{''.join(str(k) for k in combo)}"
                    gt[scene_id] = boxes
                    for k in combo:
                        for rec in by_round[k]:
                            out = dict(rec)
                            out.update(run_id=scene_id, robot_id=f"{rec['robot_id']}@r{k}", round=0,
                                       trial_id=f"{rec.get('trial_id', '')}@r{k}", source_round=k)
                            f.write(json.dumps(out) + "\n")
                    n_scenes += 1
        with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
            json.dump(gt, f)
        counts[r] = n_scenes
        print(f"  rounds_{r}: {n_scenes} scenes", flush=True)
    return counts


def baseline(task):
    r, data_dir, out_dir, sensor_calib, regions = task
    cmd = [sys.executable, "-m", "pipeline.evaluate_runs", "--survey-dir", data_dir,
           "--ground-truth", os.path.join(data_dir, "ground_truth.json"), "--sensor-calib", sensor_calib,
           "--full-only", "--quiet", "--out", out_dir]
    if regions:
        cmd += ["--regions", regions]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", default="data/survey_rounds/")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--regions", default=None, help="arena regions, for zone+class")
    ap.add_argument("--max-rounds", type=int, default=7)
    ap.add_argument("--max-combos", type=int, default=35, help="combinations per run and r at most (random subset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-data", default="data/real_combined/")
    ap.add_argument("--out-baselines", default="results/baselines_real_combined/")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)             # Windows cannot wait on more than 63 process handles

    counts = build(a)
    if a.no_baselines:
        return
    tasks = [(r, os.path.join(a.out_data, f"rounds_{r}"), os.path.join(a.out_baselines, f"rounds_{r}"),
              a.sensor_calib, a.regions) for r in counts]
    print("classical methods on every folder (the ones with many viewpoints take longest)", flush=True)
    with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
        for r in pool.imap_unordered(baseline, tasks):
            print(f"  rounds_{r}: {os.path.join(a.out_baselines, f'rounds_{r}', 'scenes.csv')}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
