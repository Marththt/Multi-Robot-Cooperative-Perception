"""Merge the static trials of several robots (written by static_trial.py, one
.jsonl file per robot) and run one or more fusion baselines over them.

No alignment by zone/time window here, unlike fondi_log_reali.py (meant
for continuous patrolling, where robots pass through the same zone at
different times): a static trial ALREADY is "all robots on the same scene",
by construction -- static_trial.py captures each robot separately but with
the SAME --trial-id.

More than one object in the same scene (e.g. 2 boxes seen together)
------------------------------------------------------------------------
static_trial.py may write SEVERAL rows per robot in the same trial, one per
distinct physical instance that robot sees (see capture_trial() there).
Here, BEFORE fusing, the rows of ALL robots for a trial are grouped AGAIN
by position (split_instances(), same principle as fondi_log_reali.py
--separa-istanze) so that "the box seen by P0 at (0.1, 0.2)" and "the box
seen by P1 at (0.12, 0.18)" end up in the SAME group (they are the same
physical box, small estimation difference) instead of two different
groups. Each group is then an independent fusion -- printed as
"trial_id instance#N" when a trial has more than one, otherwise just
"trial_id" (identical behaviour to before for single-object trials).

Usage
-----
    python merge_static_trials.py --trial-dir trials/
    python merge_static_trials.py --trial-dir trials/ --baseline majority
    python merge_static_trials.py --trial-dir trials/ --baseline majority \\
        --ground-truth ground_truth_trials.json

ground_truth_trials.json (optional): {"trial001": "box", "trial002": null, ...}
-- the true class for that trial, used ONLY for single-instance trials
(with several instances it is unknown which one a single ground truth per
trial_id would refer to, so accuracy is not counted there, only the
decisions are printed).
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from vf.fusion.multi_robot_simulation import Observation
from vf.perception.object_position import cluster_by_position
from vf.fusion.aggregation import (
    baseline_single_robot, baseline_0_union, baseline_1_majority,
    baseline_2_robust_statistics, baseline_3a_trimmed_mean,
    baseline_chair_varshney)
BASELINE = {
    "single_robot": lambda obs: baseline_single_robot(obs),
    "union": baseline_0_union,
    "majority": baseline_1_majority,
    "robust_statistics": baseline_2_robust_statistics,
    "trimmed_mean": baseline_3a_trimmed_mean,
    "chair_varshney": baseline_chair_varshney,
}


def load_trials(folder):
    """robot_id -> [rows], from <folder>/<robot_id>.jsonl."""
    per_robot = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        robot_id = os.path.splitext(os.path.basename(path))[0]
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        per_robot[robot_id] = rows
    return per_robot


def group_by_trial(per_robot):
    """trial_id -> [(robot_id, row), ...], a FLAT list (not a dict per
    robot_id) -- a robot may have several rows for the same trial_id when
    the scene has several objects (see split_instances() below, which
    separates them again by physical instance)."""
    trials = defaultdict(list)
    for robot_id, rows in per_robot.items():
        for row in rows:
            trials[row["trial_id"]].append((robot_id, row))
    return trials


def split_instances(robot_rows, threshold_m=0.5):
    """[(robot_id, row), ...] of ONE trial -> list of {robot_id: row},
    one dict per distinct physical instance (same principle as
    fondi_log_reali.py --separa-istanze, but here across rows of different
    robots instead of across readings over time of the same robot).

    Rows WITHOUT a position (class_name=None "nothing seen", or detected but
    without a position estimate) have no point to cluster: they are copied
    into EVERY group, so it stays visible that that robot saw nothing even
    when looking at a single instance. If there are fewer than 2 positioned
    rows in total there is nothing to split: a single group with all rows
    is returned (identical behaviour to before this change)."""
    positioned = [(rid, r) for rid, r in robot_rows if r.get("position")]
    unpositioned = [(rid, r) for rid, r in robot_rows if not r.get("position")]

    if len(positioned) < 2:
        group = {}
        for rid, r in robot_rows:
            group[rid] = r
        return [group]

    positions = [tuple(r["position"]) for _, r in positioned]
    cluster_indices = cluster_by_position(positions, threshold_m=threshold_m)
    n_clusters = max(cluster_indices) + 1

    groups = [dict() for _ in range(n_clusters)]
    for (rid, r), idx in zip(positioned, cluster_indices):
        current = groups[idx].get(rid)
        if current is None or r.get("confidence", 0) > current.get("confidence", 0):
            groups[idx][rid] = r

    for rid, r in unpositioned:
        for group in groups:
            if rid not in group:
                group[rid] = r

    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--baseline", choices=list(BASELINE.keys()) + ["all"],
                   default="all")
    p.add_argument("--instance-threshold-m", type=float, default=0.5,
                   help="within how many metres two readings from different "
                        "robots are considered the same physical object "
                        "(same default as static_trial.py)")
    p.add_argument("--ground-truth", default=None,
                   help="optional JSON trial_id -> true class (or null), "
                        "to count accuracy instead of only printing the "
                        "decisions. Counted only for trials with ONE "
                        "instance (with several objects it is unknown "
                        "which one it would refer to)")
    args = p.parse_args()

    per_robot = load_trials(args.trial_dir)
    if not per_robot:
        print(f"no .jsonl file found in {args.trial_dir}")
        return
    trials = group_by_trial(per_robot)
    if not trials:
        print("no trial found (empty .jsonl files?)")
        return

    ground_truth = None
    if args.ground_truth:
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)

    baselines_to_use = (list(BASELINE.items()) if args.baseline == "all"
                        else [(args.baseline, BASELINE[args.baseline])])

    correct = defaultdict(int)
    total = 0
    skipped_multi_instance_trials = 0
    for trial_id in sorted(trials.keys()):
        groups = split_instances(trials[trial_id],
                                 threshold_m=args.instance_threshold_m)
        multi = len(groups) > 1
        for i, rows in enumerate(groups, start=1):
            observations = {
                robot_id: Observation(
                    robot_id=robot_id, class_name=r["class_name"],
                    confidence=r["confidence"],
                    fault_injected=r.get("fault_injected", False),
                    dropped=r.get("dropped", False))
                for robot_id, r in rows.items()
            }
            votes = ", ".join(f"{rid}={o.class_name}({o.confidence:.2f})"
                              for rid, o in observations.items())
            label = f"{trial_id} instance#{i}" if multi else trial_id
            parts = [f"[{label}] {votes}"]
            count_here = (ground_truth is not None and trial_id in ground_truth
                          and not multi)
            for name, fn in baselines_to_use:
                decision = fn(observations)
                marker = ""
                if count_here:
                    is_correct = decision == ground_truth[trial_id]
                    correct[name] += int(is_correct)
                    marker = " OK" if is_correct else " X"
                parts.append(f"{name}={decision}{marker}")
            print("  ".join(parts))
            if count_here:
                total += 1
        if multi and ground_truth is not None and trial_id in ground_truth:
            skipped_multi_instance_trials += 1

    if ground_truth is not None and total:
        print(f"\naccuracy over {total} (single-instance) trials with known "
              f"ground truth:")
        for name, _ in baselines_to_use:
            print(f"  {name}: {correct[name]}/{total} "
                  f"({100 * correct[name] / total:.0f}%)")
    elif ground_truth is not None:
        print("\nno trial-id from the log found in --ground-truth "
              "(check that the ids match)")
    if skipped_multi_instance_trials:
        print(f"\n({skipped_multi_instance_trials} trials with more than "
              f"one instance excluded from the accuracy count -- "
              f"decisions still printed above, one line per instance)")


if __name__ == "__main__":
    main()