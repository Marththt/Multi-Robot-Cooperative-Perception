"""Real runs and camera geometry: the frames of every robot at every stop, the lens pose from OptiTrack, the
camera matrix and distortion of every robot, and the pitch / yaw offset of every camera fitted on the marker
detections.

A run folder (one layout) holds raw/<robot>.jsonl (one record per frame: round, reading, camera_pose, image,
detections), raw/meta_<layout>_<robot>.json (camera matrix, distortion) and ground_truth.json.
"""

import glob
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np

from sim.geometry import BOX_SIDE_M

PITCH_DEG = dict(P0=11.5, P1=11.5, P3=11.5, P5=22.0)      # measured on the real frames (imgsim/render.py)
CAMERA_HEIGHT_M = 0.25                  # lens above the floor (measured 24-26 cm)
CAMERA_FORWARD_FIX_M = 0.03             # lens 15 cm ahead of the rigid body; the acquisition moved the pose by 12 cm
MARKER_CENTRE_M = 0.15                  # marker centre above the floor (faces 30 cm high)
BOX_HEIGHT_M = 0.30
APOTHEM_M = BOX_SIDE_M / (2 * math.sqrt(3))


def camera_rotation(yaw, pitch, roll):
    """World -> camera rotation (rows: camera x right, y down, z forward), as imgsim/render.py."""
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    d = np.cross(f, r)
    r2 = math.cos(roll) * r + math.sin(roll) * d
    d2 = -math.sin(roll) * r + math.cos(roll) * d
    return np.stack([r2, d2, f])


def camera_pose(rec):
    """(x, y, yaw) of the lens: the recorded camera pose moved forward by CAMERA_FORWARD_FIX_M."""
    x, y, yaw = rec["camera_pose"]
    return (x + CAMERA_FORWARD_FIX_M * math.cos(yaw), y + CAMERA_FORWARD_FIX_M * math.sin(yaw), yaw)


def run_dirs(patterns):
    """Run folders (with a raw/ subfolder) matching the glob patterns, sorted."""
    out = []
    for p in patterns:
        out += [d for d in glob.glob(p) if os.path.isdir(os.path.join(d, "raw"))]
    return sorted(set(out))


def load_run(run_dir):
    """One run folder: (layout id, true boxes (B, 2), frames, cameras), with
    frames[(round, reading, yaw offset)][robot] = the raw record of that frame (pose, image path, detections)
    and cameras[robot] = (camera matrix, distortion)."""
    with open(os.path.join(run_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    frames = defaultdict(dict)
    cams = {}
    layout = None
    for path in sorted(glob.glob(os.path.join(run_dir, "raw", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("round") is None:
                    continue
                layout = r["layout_id"]
                off = int(round(float(r["trial_id"].rsplit("/", 1)[-1])))   # --yaw-offsets-deg captures
                frames[(int(r["round"]), int(r["reading"]), off)][r["robot_id"]] = r
    for path in glob.glob(os.path.join(run_dir, "raw", "meta_*.json")):
        with open(path) as f:
            m = json.load(f)
        cams[m["robot_id"]] = (np.array(m["camera_matrix"], float), np.array(m["dist_coeffs"], float).ravel())
    key = layout if layout in gt else os.path.basename(os.path.normpath(run_dir))
    boxes = np.array([[b["x"], b["y"]] for b in gt[key]], np.float32)
    return layout, boxes, frames, cams


def angle_offsets(runs, pitch):
    """Per (layout, robot): offsets [deg] between where the marker centres are seen (undistorted) and where
    the true box puts them (the face towards the camera, at the marker centre height), vertical and horizontal."""
    out = defaultdict(lambda: ([], []))
    for d, layout, boxes, frames, cams in runs:
        for recs in frames.values():
            for q, r in recs.items():
                K, D = cams[q]
                x, y, yaw = camera_pose(r)
                Rm = camera_rotation(yaw, math.radians(pitch.get(q, 11.5)), 0.0)
                for det in r.get("detections", []):
                    p = np.asarray(det["position"], float)
                    k = int(np.argmin(np.hypot(*(boxes - p).T)))
                    if np.hypot(*(boxes[k] - p)) > 0.3:
                        continue
                    u = (boxes[k] - [x, y]) / np.linalg.norm(boxes[k] - [x, y])
                    face = boxes[k] - APOTHEM_M * u
                    pc = Rm @ (np.array([face[0], face[1], MARKER_CENTRE_M]) - [x, y, CAMERA_HEIGHT_M])
                    bx, by, bw, bh = det["bbox"]
                    obs = cv2.undistortPoints(np.array([[[bx + bw / 2, by + bh / 2]]], np.float64), K, D).reshape(2)
                    out[(layout, q)][0].append(math.degrees(math.atan(obs[1]) - math.atan(pc[1] / pc[2])))
                    out[(layout, q)][1].append(math.degrees(math.atan(obs[0]) - math.atan(pc[0] / pc[2])))
    return out


def calibration(runs, pitch, scope="other-layouts"):
    """(layout, robot) -> (pitch [deg], yaw offset [deg]) of each camera, from the marker detections.
    scope "other-layouts": medians over all the other layouts (the camera mounting is the same throughout),
    so a layout never calibrates its own cameras. scope "same-run": medians over the layout itself."""
    off = angle_offsets(runs, pitch)
    order = [layout for _, layout, *_ in runs]
    cal = {}
    for (layout, q) in off:
        use = {layout} if scope == "same-run" else set(order) - {layout}
        dv = [v for (l2, q2), (vs, _) in off.items() if q2 == q and l2 in use for v in vs]
        du = [v for (l2, q2), (_, us) in off.items() if q2 == q and l2 in use for v in us]
        if dv:
            cal[(layout, q)] = (pitch.get(q, 11.5) - float(np.median(dv)), float(np.median(du)))
    return cal
