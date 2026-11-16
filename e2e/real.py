"""The real scenes in the format of e2e/data.py, to test the end-to-end network trained in simulation only.

For every stop (layout, round) and every reading (each robot took several frames per stop): the frame of each
robot resized to the network's image size (--scale of 1280 x 720), its lens pose from OptiTrack, its own camera
matrix and distortion scaled to that size, the camera pitch and yaw offset fitted on the marker detections of the
OTHER layouts (yolo_gnn/camera.calibration, as for the YOLO rays of the proposed method: the detections only set
the camera angles, they are never an input of the network), and the true boxes. Poses and boxes are shifted so
that the centre of the robots is the centre of the floor grid. The frames get the same gamma lift as the frames
given to YOLO (yolo_gnn/segment.py: the lab frames are darker than the simulated ones).

One set, data/e2e_real/real/ (shard_*.pkl, meta.json); run id <layout>_r<round> for every reading of a stop,
so e2e/train.py averages the maps of the readings of one stop. --check draws the true box centres into a few
frames (fig_e2e_real_check.jpg), to see that poses, pitch and lens put them on the boxes.

    python -m e2e.real --runs "data/survey_all/*" --out data/e2e_real/ --check
"""

import argparse
import json
import math
import os
import pickle

import cv2
import numpy as np

from e2e import geometry as geo
from e2e.data import SHARD
from yolo_gnn.camera import CAMERA_HEIGHT_M, PITCH_DEG, calibration, camera_pose, load_run, run_dirs


def build_scene(run_dir, layout, boxes, recs, cams, cal, scale, quality, lut):
    robots = sorted(recs)
    poses = np.array([camera_pose(recs[q]) for q in robots], np.float64)
    pitch_deg = []
    for n, q in enumerate(robots):
        p, yaw_off = cal.get((layout, q), (PITCH_DEG.get(q, 11.5), 0.0))
        pitch_deg.append(p)
        poses[n, 2] += math.radians(yaw_off)
    centre = poses[:, :2].mean(0)
    poses[:, :2] -= centre
    w, h = int(round(1280 * scale)), int(round(720 * scale))
    jpgs, Ks, Ds = [], [], []
    for q in robots:
        img = cv2.imread(os.path.join(run_dir, recs[q]["image"].replace("\\", "/")))
        if img is None:
            return None
        H, W = img.shape[:2]
        if lut is not None:
            img = cv2.LUT(img, lut)
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpgs.append(buf.tobytes())
        K, D = cams[q]
        K = K.copy()
        K[0] *= w / W
        K[1] *= h / H
        Ks.append(K.tolist())
        Ds.append(np.asarray(D).tolist())
    rnd = int(recs[robots[0]]["round"])
    return dict(run=f"{layout}_r{rnd}", family="real", boxes=(np.asarray(boxes) - centre).astype(np.float32),
                poses=poses.astype(np.float32), pitch=np.radians(pitch_deg).astype(np.float32),
                roll=np.zeros(len(robots), np.float32), height=CAMERA_HEIGHT_M, jpgs=jpgs, robots=robots, K=Ks,
                D=Ds, centre=centre.tolist(), n_robots=len(robots), n_boxes=len(boxes))


def check_figure(scenes, path, n=4):
    """True box centres (green, at the heights of e2e/geometry.py) drawn into the frames of a few scenes."""
    rows = []
    for s in scenes[:n]:
        tiles = []
        for k, jpg in enumerate(s["jpgs"]):
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            size = (img.shape[1], img.shape[0])
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            for b in s["boxes"]:
                for hgt in geo.HEIGHTS_M:
                    uv, ok = geo.project(np.array([[b[0], b[1], hgt]]), s["poses"][k], float(s["pitch"][k]), 0.0,
                                         s["height"], np.array(s["K"][k]), np.array(s["D"][k]), size)
                    if ok[0]:
                        cv2.circle(img, (int(2 * uv[0, 0]), int(2 * uv[0, 1])), 4, (0, 220, 0), -1)
            cv2.putText(img, f"{s['run']} {s['robots'][k]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["data/survey_all/*"])
    ap.add_argument("--out", default="data/e2e_real/")
    ap.add_argument("--scale", type=float, default=0.2, help="as e2e.data --scale")
    ap.add_argument("--readings", default="0,1,2,3,4", help="comma list of the readings of every stop")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--gamma", type=float, default=1.8, help="gamma lift of the real frames (as for YOLO)")
    ap.add_argument("--calibrate-scope", choices=["other-layouts", "same-run"], default="other-layouts")
    ap.add_argument("--check", action="store_true", help="write fig_e2e_real_check.jpg")
    a = ap.parse_args()
    readings = {int(v) for v in a.readings.split(",")}
    runs = [(d, *load_run(d)) for d in run_dirs(a.runs)]
    cal = calibration(runs, dict(PITCH_DEG), a.calibrate_scope)
    lut = (np.clip(255 * (np.arange(256) / 255.0) ** (1.0 / a.gamma), 0, 255).astype(np.uint8)
           if a.gamma != 1.0 else None)
    scenes = []
    for d, layout, boxes, frames, cams in runs:
        for (rnd, reading, off), recs in sorted(frames.items()):
            if off != 0 or reading not in readings:
                continue
            s = build_scene(d, layout, boxes, recs, cams, cal, a.scale, a.jpeg_quality, lut)
            if s is not None:
                s["reading"] = reading
                scenes.append(s)
    out = os.path.join(a.out, "real")
    os.makedirs(out, exist_ok=True)
    for old in [f for f in os.listdir(out) if f.startswith("shard_")]:
        os.replace(os.path.join(out, old), os.path.join(out, old + ".old"))
    sizes = []
    for j, start in enumerate(range(0, len(scenes), SHARD)):
        with open(os.path.join(out, f"shard_{j:04d}.pkl"), "wb") as f:
            pickle.dump(scenes[start:start + SHARD], f, protocol=pickle.HIGHEST_PROTOCOL)
        sizes.append(len(scenes[start:start + SHARD]))
    s0 = scenes[0]
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(set="real", scenario="real", level="", scenes=len(scenes), shard_sizes=sizes, K=s0["K"][0],
                       D=s0["D"][0], size=[int(round(1280 * a.scale)), int(round(720 * a.scale))], scale=a.scale,
                       gamma=a.gamma, calibrate_scope=a.calibrate_scope, runs=[os.path.basename(r[0]) for r in runs]),
                  f, indent=2)
    stops = len({s["run"] for s in scenes})
    print(f"  {len(runs)} layouts, {stops} stops, {len(scenes)} frame sets -> {out}", flush=True)
    if a.check:
        check_figure(scenes[::max(1, len(scenes) // 4)], os.path.join(a.out, "fig_e2e_real_check.jpg"))
        print(f"  check figure: {os.path.join(a.out, 'fig_e2e_real_check.jpg')}")


if __name__ == "__main__":
    main()
