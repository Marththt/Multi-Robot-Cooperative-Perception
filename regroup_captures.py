"""Rebuilds the captures of a survey from the RAW readings (every frame),
with a different grouping threshold, without running the robots again.

Every capture is --n-readings frames; the detections of all its frames are
grouped by position into physical objects (one row per object, the reading
whose distance is closest to the median of its group), exactly as
static_trial.capture_trial does during the survey. During the survey the
threshold is --instance-threshold-m (default 0.5 m): two boxes closer than
that, seen by the same robot, become ONE object. This script redoes the
grouping with another threshold and writes a new survey folder that
analyze_rounds.py / build_scans.py / evaluate_runs.py read as usual.

    python regroup_captures.py --survey-dir survey_rounds/ --out survey_rounds_t015/ --threshold 0.15

Input: <run>/raw/<robot>.jsonl (one row per frame, written by survey.py)
and <run>/<robot>.jsonl (for the capture metadata). Output: the same run
folders with new <robot>.jsonl files (ground_truth.json is copied).
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from vf.perception.object_position import cluster_by_position

META_KEYS = ("trial_id", "robot_id", "layout_id", "stop", "radius_m", "angle_deg", "yaw_offset_deg",
             "fault_injected", "dropped", "zone", "round", "run_id")


def regroup_capture(frames, threshold_m):
    """frames: raw rows of ONE capture. Returns one dict per physical object
    (detections grouped across the frames by position)."""
    dets = []
    for f in frames:
        for d in f["detections"]:
            if d.get("class_name") and d.get("position") is not None:
                dets.append(dict(d, robot_pose=f.get("camera_pose"), t=f["t"]))
    if not dets:
        return []
    labels = cluster_by_position([tuple(d["position"]) for d in dets], threshold_m=threshold_m)
    groups = defaultdict(list)
    for d, k in zip(dets, labels):
        groups[k].append(d)
    out = []
    for g in groups.values():
        with_d = [d for d in g if d.get("distance") is not None]
        if len(with_d) >= 2:
            med = float(np.median([d["distance"] for d in with_d]))
            rep = min(with_d, key=lambda d: abs(d["distance"] - med))
        else:
            rep = g[0]
        out.append(rep)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="detections of one capture closer than this are the same object [m]")
    a = ap.parse_args()

    raw_files = sorted(glob.glob(os.path.join(a.survey_dir, "**", "raw", "*.jsonl"), recursive=True))
    if not raw_files:
        raise SystemExit(f"no raw/*.jsonl under {a.survey_dir}: the runs were recorded before the raw readings "
                         f"were saved")
    for raw in raw_files:
        run_dir = os.path.dirname(os.path.dirname(raw))
        robot = os.path.splitext(os.path.basename(raw))[0]
        rel = os.path.relpath(run_dir, a.survey_dir)
        out_dir = os.path.join(a.out, rel)
        os.makedirs(out_dir, exist_ok=True)
        gt = os.path.join(run_dir, "ground_truth.json")
        if os.path.exists(gt):
            shutil.copy(gt, out_dir)
        meta = {}
        main_file = os.path.join(run_dir, f"{robot}.jsonl")
        if os.path.exists(main_file):
            with open(main_file) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        meta.setdefault((r["trial_id"], r.get("round")), {k: r.get(k) for k in META_KEYS})
        caps = defaultdict(list)
        with open(raw) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    caps[(r["trial_id"], r.get("round"))].append(r)
        n_in = n_out = 0
        with open(os.path.join(out_dir, f"{robot}.jsonl"), "w") as f:
            for key, frames in caps.items():
                m = meta.get(key)
                if m is None:
                    continue
                objs = regroup_capture(frames, a.threshold)
                pose = next((fr["camera_pose"] for fr in frames if fr.get("camera_pose")), None)
                rows = [dict(class_name=o["class_name"], confidence=o.get("confidence", 1.0),
                             position=o["position"], distance=o.get("distance"), bearing_rad=o.get("bearing_rad"),
                             viewing_angle_rad=o.get("viewing_angle_rad"), robot_pose=o.get("robot_pose") or pose,
                             t=o["t"], **m) for o in objs]
                if not rows:
                    rows = [dict(class_name=None, confidence=0.0, distance=None, bearing_rad=None,
                                 viewing_angle_rad=None, robot_pose=pose, t=frames[0]["t"], **m)]
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                n_in += 1
                n_out += sum(1 for r in rows if r["class_name"])
        print(f"{rel}/{robot}: {n_in} capture(s), {n_out} object(s) with threshold {a.threshold:g} m")


if __name__ == "__main__":
    main()