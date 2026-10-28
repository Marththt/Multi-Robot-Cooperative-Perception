"""Scenes for the GNN, built from the same round-mode runs used by pipeline/evaluate_runs.py.

A SCENE is one round of one run: the feature vectors of all the robots standing at their positions of
that round (same definition as pipeline/evaluate_runs.py). For every scene this module returns

  robots     robot ids, in a fixed order
  poses      camera pose (x, y, yaw) of every robot                                  (R, 3)
  det_robot  index of the robot that produced each detection                          (N,)
  distance   measured distance, corrected with the range fit                          (N,)
  bearing    measured bearing, corrected with the bearing fit                         (N,)
  view       viewing angle of the marker                                              (N,)
  pos        world position of the detection, from pose, distance and bearing         (N, 2)
  label      index of the true box the detection belongs to, -1 for a false detection (N,)
  truth      true box positions (OptiTrack)                                           (M, 2)

The distance / bearing correction and the camera offset come from the calibration file written by
pipeline/calibrate.py (--sensor-calib), exactly as pipeline/evaluate_runs.py applies them, so the GNN and the classical
methods see the same detections at the same positions.

A detection is labelled with the nearest true box when it lies within LABEL_RADIUS_M of it, otherwise it
is a false detection (-1). The labels are used for training only; the evaluation compares the estimated
boxes with the true boxes as pipeline/evaluate_runs.py does.

    python -m gnn.dataset --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --sensor-calib config/sensor_calib.json
prints a summary of the scenes (detections, false detections, boxes seen twice by one robot).
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

LABEL_RADIUS_M = 0.4


def load_calibration(path):
    """Range fit, bearing fit and camera offset from pipeline/calibrate.py --out (identity without a file)."""
    if not path:
        return dict(range_fit=None, bearing_fit=None, camera_offset=0.0)
    with open(path) as f:
        calib = json.load(f)
    return dict(range_fit=calib.get("range_fit"), bearing_fit=calib.get("bearing_fit"),
                camera_offset=float(calib.get("camera_offset") or 0.0))


def load_truth(path):
    """{run or layout id: (M, 2) array of true box centres}."""
    with open(path) as f:
        data = json.load(f)
    return {k: np.array([[b["x"], b["y"]] for b in v], float) for k, v in data.items()}


def read_records(survey_dir):
    """All feature vectors of the round-mode runs (same files as build_scans.load_points)."""
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
                if not r.get("class_name") or r.get("distance") is None or r.get("bearing_rad") is None:
                    continue
                if r.get("robot_pose") is None:
                    continue
                yield r


def build_scenes(survey_dir, truth_path, calib_path=None):
    calib = load_calibration(calib_path)
    truth = load_truth(truth_path)
    A, B = calib["range_fit"] or (1.0, 0.0)
    K, B0 = calib["bearing_fit"] or (1.0, 0.0)
    off = calib["camera_offset"]

    grouped = defaultdict(list)
    for r in read_records(survey_dir):
        run = r.get("run_id") or r["layout_id"]
        grouped[(run, r["layout_id"], int(r["round"]))].append(r)

    scenes = []
    for (run, layout, rnd), recs in sorted(grouped.items()):
        true_xy = truth.get(run, truth.get(layout))
        if true_xy is None:
            continue
        robots = sorted({r["robot_id"] for r in recs})
        poses = {}
        det_robot, distance, bearing, view, pos = [], [], [], [], []
        for r in sorted(recs, key=lambda q: (q["robot_id"], q.get("t", 0.0))):
            x, y, yaw = r["robot_pose"]
            cam = (x + off * math.cos(yaw), y + off * math.sin(yaw), yaw)
            poses.setdefault(r["robot_id"], cam)
            d = (float(r["distance"]) - B) / A
            b = (float(r["bearing_rad"]) - B0) / K
            det_robot.append(robots.index(r["robot_id"]))
            distance.append(d)
            bearing.append(b)
            view.append(float(r.get("viewing_angle_rad") or 0.0))
            # bearing positive to the right: world direction = yaw - bearing
            pos.append((cam[0] + d * math.cos(cam[2] - b), cam[1] + d * math.sin(cam[2] - b)))
        pos = np.array(pos, float)
        gaps = np.linalg.norm(pos[:, None, :] - true_xy[None, :, :], axis=2)
        nearest = gaps.argmin(axis=1)
        label = np.where(gaps[np.arange(len(pos)), nearest] <= LABEL_RADIUS_M, nearest, -1)
        scenes.append(dict(
            run=run, layout=layout, round=rnd, robots=robots,
            poses=np.array([poses[q] for q in robots], float),
            det_robot=np.array(det_robot, int), distance=np.array(distance), bearing=np.array(bearing),
            view=np.array(view), pos=pos, label=label.astype(int), truth=true_xy))
    return scenes


def subset_robots(scene, keep):
    """The same scene seen by the robots `keep` only (indices into scene['robots'])."""
    keep = sorted(keep)
    mask = np.isin(scene["det_robot"], keep)
    remap = {old: new for new, old in enumerate(keep)}
    out = dict(scene)
    out["robots"] = [scene["robots"][i] for i in keep]
    out["poses"] = scene["poses"][keep]
    out["det_robot"] = np.array([remap[i] for i in scene["det_robot"][mask]], int)
    for key in ("distance", "bearing", "view", "label"):
        out[key] = scene[key][mask]
    out["pos"] = scene["pos"][mask]
    return out


def describe(scenes):
    n_det = sum(len(s["pos"]) for s in scenes)
    n_false = sum(int((s["label"] < 0).sum()) for s in scenes)
    twice = 0
    for s in scenes:
        seen = defaultdict(int)
        for r, lab in zip(s["det_robot"], s["label"]):
            if lab >= 0:
                seen[(r, lab)] += 1
        twice += sum(1 for v in seen.values() if v > 1)
    print(f"{len(scenes)} scenes, {n_det} detections, {n_false} false detections "
          f"(farther than {LABEL_RADIUS_M} m from every box), {twice} boxes reported twice by one robot")
    per_layout = defaultdict(int)
    for s in scenes:
        per_layout[s["layout"]] += 1
    def order(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 0, name)

    print("scenes per layout: " + ", ".join(f"{k} {per_layout[k]}" for k in sorted(per_layout, key=order)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--sensor-calib", default=None)
    a = ap.parse_args()
    describe(build_scenes(a.survey_dir, a.ground_truth, a.sensor_calib))


if __name__ == "__main__":
    main()