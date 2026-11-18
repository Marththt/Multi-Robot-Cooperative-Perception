"""Image caches: one grey image per robot and stop, only resized, no detector or segmentation applied.

  sim    the scenes of a simulated set (geometry.jsonl of sim/plan.py or sim/scaling.py) rendered by
         imgsim/render.py: the frame of every robot at its pose (render at --render-scale, then area resize)
  real   the real frames of the survey runs (data/survey_all/<run>/raw/<robot>.jsonl): for every round the first
         reading of every robot, area resized

Output, one folder per set: frames.npy (F, H, W) uint8 grey and frames.csv (one row per frame: run, round, robot,
row, camera pose x, y, yaw, plus what the linear probe of imgmlp/probe.py needs: boxes in the field of view, hidden boxes, mean distance
of the boxes in view, mean grey level, and for the simulated frames ambient light, exposure-free floor / wall grey
and camera pitch). The images are kept at --size (default 128 x 72); the model reduces them further when it loads
them (imgmlp/train.py --img-size).

    python -m imgmlp.frames sim --set data/plan_v2/train --out data/imgmlp/plan_v2_train --max-scenes 20000
    python -m imgmlp.frames sim --set data/plan_v2/val --out data/imgmlp/plan_v2_val
    python -m imgmlp.frames real --runs "data/survey_all/*" --ground-truth config/ground_truth_positions.json \\
        --out data/imgmlp/real
"""

import argparse
import csv
import glob
import json
import math
import multiprocessing
import os
import sys

import cv2
import numpy as np

from imgmlp.measure import box_visibility

CALIB = dict(P0="config/checkerboard_calib_P0.json", P1="config/checkerboard_calib_P1.json")
FIELDS = ["row", "run", "round", "robot", "x", "y", "yaw", "in_view", "hidden", "mean_dist_m", "grey", "ambient",
          "floor", "wall", "pitch_deg"]


def pose_fields(pose):
    return dict(x=round(float(pose[0]), 5), y=round(float(pose[1]), 5), yaw=round(float(pose[2]), 5))

_R = None


def to_grey(img, size):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, tuple(size), interpolation=cv2.INTER_AREA)


def view_stats(k, poses, boxes, thetas, obstacles=()):
    vis = box_visibility(k, poses, boxes, thetas, obstacles)
    seen = [d for _, d, _, h in vis if not h]
    return dict(in_view=len(vis), hidden=sum(h for *_, h in vis),
                mean_dist_m=round(float(np.mean(seen)), 4) if seen else "")


# ====================================================================
# Simulated frames
# ====================================================================

def _init(calib, scale, look, seed):
    global _R
    from imgsim.render import Appearance, Renderer
    _R = Renderer(calib, Appearance(visual_style=look, obstacle_style=look), seed=seed, scale=scale)


def render_rows(task):
    """Frames of a chunk of geometry rows: [(meta, image)]."""
    rows, size, seed = task
    from sim.scaling import scene_from_geometry
    out = []
    for i, g in rows:
        scene = scene_from_geometry(g)
        rng = np.random.default_rng([seed, i])
        style = _R.scene_style(rng)
        poses = [tuple(p) for p in g["poses"]]
        boxes = np.array(g["boxes"], float).reshape(-1, 2)
        obstacles = [np.array(o, float) for o in g.get("obstacles", [])]
        for k, cap in enumerate(scene.captures):
            clean, _, info = _R.render_clean(scene, k, rng, dict(style))
            img = to_grey(_R.camera_effects(clean, rng), size)
            meta = dict(run=g["run"], round=0, robot=cap.robot, **pose_fields(poses[k]), grey=round(float(img.mean()), 2),
                        ambient=round(float(style["ambient"]), 3), floor=round(float(style["floor"]), 1),
                        wall=round(float(style["wall"]), 1), pitch_deg=info["pitch_deg"],
                        **view_stats(k, poses, boxes, g.get("yaws"), obstacles))
            out.append((meta, img))
    return out


def sim_main(a):
    from sim.scaling import read_geometry
    geo = list(read_geometry(a.set).values())[:a.max_scenes]
    n_frames = sum(len(g["poses"]) for g in geo)
    os.makedirs(a.out, exist_ok=True)
    W, H = a.size
    frames = np.lib.format.open_memmap(os.path.join(a.out, "frames.npy"), mode="w+", dtype=np.uint8,
                                       shape=(n_frames, H, W))
    chunks = [(list(enumerate(geo))[s:s + a.chunk], a.size, a.seed) for s in range(0, len(geo), a.chunk)]
    print(f"{a.set}: {len(geo)} scenes, {n_frames} frames -> {a.out}", flush=True)
    row = 0
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f, \
            multiprocessing.Pool(a.jobs, initializer=_init, initargs=(CALIB[a.camera], a.render_scale, a.look, a.seed)) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for c, out in enumerate(pool.imap(render_rows, chunks)):
            for meta, img in out:
                frames[row] = img
                w.writerow(dict(row=row, **meta))
                row += 1
            if (c + 1) % 20 == 0 or c + 1 == len(chunks):
                print(f"  {min((c + 1) * a.chunk, len(geo))}/{len(geo)} scenes", flush=True)
    frames.flush()
    with open(os.path.join(a.out, "source.json"), "w") as f:
        json.dump({k: v for k, v in vars(a).items() if k != "func"}, f, indent=2)
    write_sheet(frames, os.path.join(a.out, "sheet.png"))


# ====================================================================
# Real frames
# ====================================================================

def real_main(a):
    from yolo_gnn.camera import load_run, run_dirs
    with open(a.ground_truth) as f:
        gt = json.load(f)
    items = []
    for d in run_dirs(a.runs):
        layout, boxes, frames, _ = load_run(d)
        run = os.path.basename(os.path.normpath(d))
        truth = gt.get(run, gt.get(layout))
        boxes = np.array([[b["x"], b["y"]] for b in truth], float) if truth else np.asarray(boxes, float)
        for rnd in sorted({r for r, _, o in frames if o == 0}):
            recs = frames.get((rnd, 0, 0)) or next(v for (r, _, o), v in sorted(frames.items()) if r == rnd and o == 0)
            robots = sorted(recs)
            poses = [tuple(recs[q]["camera_pose"]) for q in robots]
            for k, q in enumerate(robots):
                items.append((d, recs[q].get("run_id") or run, rnd, q, recs[q]["image"], poses, k, boxes))
    os.makedirs(a.out, exist_ok=True)
    W, H = a.size
    frames = np.lib.format.open_memmap(os.path.join(a.out, "frames.npy"), mode="w+", dtype=np.uint8,
                                       shape=(len(items), H, W))
    print(f"{len(items)} real frames -> {a.out}", flush=True)
    missing = 0
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row, (d, run, rnd, q, path, poses, k, boxes) in enumerate(items):
            img = cv2.imread(os.path.join(d, path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                missing += 1
                img = np.zeros((H, W), np.uint8)
            img = to_grey(img, a.size)
            frames[row] = img
            # no yaws of the real boxes: the footprints are drawn with yaw 0 (only used for hidden / in view)
            w.writerow(dict(row=row, run=run, round=rnd, robot=q, **pose_fields(poses[k]), grey=round(float(img.mean()), 2),
                            **view_stats(k, poses, boxes, None)))
    frames.flush()
    if missing:
        print(f"warning: {missing} frames not found (written as black)")
    with open(os.path.join(a.out, "source.json"), "w") as f:
        json.dump({k: v for k, v in vars(a).items() if k != "func"}, f, indent=2)
    write_sheet(frames, os.path.join(a.out, "sheet.png"))


# ====================================================================
# Reading a cache
# ====================================================================

def write_sheet(frames, path, n=24, cols=6):
    """A sheet of the first frames, enlarged x2 (to check the cache by eye)."""
    imgs = [cv2.resize(frames[i], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for i in range(min(n, len(frames)))]
    if not imgs:
        return
    while len(imgs) % cols:
        imgs.append(np.zeros_like(imgs[0]))
    cv2.imwrite(path, np.vstack([np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]))


def load_cache(folder, img_size=None):
    """(images (F, H, W) uint8 at img_size = (W, H), {(run, round, robot): row}, rows of frames.csv)."""
    frames = np.load(os.path.join(folder, "frames.npy"), mmap_mode="r")
    if img_size is not None and tuple(img_size) != (frames.shape[2], frames.shape[1]):
        frames = np.stack([cv2.resize(f, tuple(img_size), interpolation=cv2.INTER_AREA) for f in frames])
    else:
        frames = np.ascontiguousarray(frames)
    with open(os.path.join(folder, "frames.csv")) as f:
        rows = list(csv.DictReader(f))
    index = {(r["run"], int(r["round"]), r["robot"]): int(r["row"]) for r in rows}
    return frames, index, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    s = sub.add_parser("sim", help="render the frames of a simulated set")
    s.add_argument("--set", required=True, help="set folder with geometry.jsonl")
    s.add_argument("--max-scenes", type=int, default=None)
    s.add_argument("--render-scale", type=float, default=0.2, help="render size / 1280 x 720 (0.2: 256 x 144)")
    s.add_argument("--look", choices=["plain", "random"], default="plain",
                   help="plain: the look matched to the real frames; random: randomized room, robots, obstacles")
    s.add_argument("--camera", choices=sorted(CALIB), default="P0")
    s.add_argument("--seed", type=int, default=5)
    s.add_argument("--chunk", type=int, default=25, help="scenes per task")
    s.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.set_defaults(func=sim_main)
    r = sub.add_parser("real", help="cache the real frames")
    r.add_argument("--runs", nargs="+", default=["data/survey_all/*"])
    r.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    r.set_defaults(func=real_main)
    for p in (s, r):
        p.add_argument("--out", required=True)
        p.add_argument("--size", type=int, nargs=2, default=[128, 72], metavar=("W", "H"))
    a = ap.parse_args()
    if sys.platform == "win32" and getattr(a, "jobs", 1) > 60:
        a.jobs = 60
    a.func(a)


if __name__ == "__main__":
    main()
