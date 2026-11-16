"""Simulated data of the end-to-end network: the scenes of sim/plan.py rendered at low resolution.

Scene i of data/e2e/<set>/ is scene i of data/plan_v2/<set>/ (same seeds), so the end-to-end network, MHT and the
marker network are compared on the same scenes. Every robot's camera image is rendered (imgsim/render.py) with
the appearance used to train YOLO (yolo_gnn/train_seg.py: random obstacles, distractor objects, random looks,
pitch 8-32 deg) at --scale of 1280 x 720 (0.2 -> 256 x 144) and stored as JPEG with the camera pose, pitch and
roll. The network gets only the images and the camera poses: no marker, no mask, no detection.

The exact visible region of every box (yolo_gnn/train_seg.render_instance_ids) is counted in pixels per image, so
a box that no robot can see is left out of the targets (the network is not asked for what no image shows).

Shards of SHARD scenes: data/e2e/<set>/shard_<k>.pkl, plus meta.json (camera matrix and distortion at the
rendered size, image size).

    python -m e2e.data --sets train,val --train-runs 30000 --out data/e2e/
    python -m e2e.data --sets G0,E3,S1 --out data/e2e/
"""

import argparse
import copy
import glob
import json
import math
import multiprocessing
import os
import pickle
import shutil
import sys
from dataclasses import asdict, replace

import cv2
import numpy as np

from e2e import geometry as geo

SHARD = 250
CALIB = dict(P0="config/checkerboard_calib_P0.json", P1="config/checkerboard_calib_P1.json")
MIN_VISIBLE_PX = 12                  # of a 256 x 144 image: a sliver of a far box, a corner of a near one


def appearance():
    """The appearance of the YOLO training images (yolo_gnn/train_seg.py)."""
    from imgsim.render import Appearance
    return Appearance(pitch_deg=[8.0, 32.0], roll_deg=[-1.5, 1.5], floor_grey=[30.0, 90.0],
                      wall_grey=[120.0, 220.0], light_ambient=[0.35, 0.85], exposure=[0.30, 1.05],
                      vignetting=0.35, obstacle_style="random", visual_style="random", light_obstacle_share=0.30)


def render_shard(task):
    """The scenes of one shard (one worker process): every scene of sim.plan with its seed, one image per robot."""
    (name, spec_dict, runs, start, seed, out_path, model_path, calib, scale, quality, obstacles, distractors) = task
    from imgsim.render import Renderer
    from sim.model import SensorModel
    from sim.plan import PlanSpec, plan_scene, run_model
    from yolo_gnn.train_seg import add_distractors, render_instance_ids
    spec = PlanSpec(**spec_dict)
    m = run_model(SensorModel.load(model_path), spec)
    renderer = Renderer(calib, appearance(), seed=seed, scale=scale)
    scenes = []
    for i in range(start, start + runs):
        scene_seed = seed * 1_000_003 + i                       # the seed of sim/plan.py for this scene
        rng = np.random.default_rng(scene_seed)
        family = spec.families[i % len(spec.families)]
        spec_i = spec
        if obstacles:                                           # training sets: occluding obstacles as in E3
            h = int(np.random.default_rng(scene_seed + 104_729).integers(0, obstacles + 1))
            spec_i = replace(spec, hidden_robots=h)
        scene, meta = plan_scene(spec_i, family, rng, name, f"{name}_{i:05d}", m)
        if distractors:
            add_distractors(scene, distractors, np.random.default_rng(scene_seed + 15_485_863))
        rrng = np.random.default_rng(scene_seed + 7_919)        # as yolo_gnn/train_seg.py
        style = renderer.scene_style(rrng)
        jpgs, poses, pitch, roll, px = [], [], [], [], []
        n_box = len(scene.boxes)
        for k, cap in enumerate(scene.captures):
            same = copy.deepcopy(rrng)                          # the same pitch and roll for the visible regions
            img, _markers, info = renderer.render(scene, k, rrng, style)
            ids = render_instance_ids(renderer, scene, k, same, style)
            ids = cv2.resize(ids.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            px.append(np.bincount(np.round(ids).astype(int).clip(0, n_box).ravel(), minlength=n_box + 1)[1:])
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            jpgs.append(buf.tobytes())
            poses.append(cap.pose)
            pitch.append(math.radians(info["pitch_deg"]))
            roll.append(math.radians(info["roll_deg"]))
        scenes.append(dict(run=scene.run_id, family=family, boxes=np.asarray(scene.boxes, np.float32),
                           poses=np.asarray(poses, np.float32), pitch=np.asarray(pitch, np.float32),
                           roll=np.asarray(roll, np.float32), height=float(renderer.app.camera_height_m),
                           jpgs=jpgs, box_pixels=np.asarray(px, np.int32).reshape(len(jpgs), n_box),
                           n_robots=meta["n_robots"], n_boxes=meta["n_boxes"]))
    with open(out_path, "wb") as f:
        pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)
    return name, out_path


def build(argv=None):
    from imgsim.render import Camera
    from sim.plan import select, set_table
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True, help="as sim.plan --sets: train, val, G0, E3, S1, set names ...")
    ap.add_argument("--train-runs", type=int, default=30000)
    ap.add_argument("--val-runs", type=int, default=1000)
    ap.add_argument("--test-runs", type=int, default=1000, help="scenes per test set (the first N of sim.plan)")
    ap.add_argument("--out", default="data/e2e/")
    ap.add_argument("--scale", type=float, default=0.2, help="image size / 1280 x 720")
    ap.add_argument("--camera", choices=sorted(CALIB), default="P0")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--obstacles", type=int, default=2,
                    help="training / validation sets: per scene, h in 0..N occluding obstacles per box (as YOLO)")
    ap.add_argument("--distractors", type=int, default=4, help="0..N free-standing objects per scene (any set)")
    ap.add_argument("--model", default="config/sensor_model.json", help="only for the scene layout")
    ap.add_argument("--seed", type=int, default=1, help="as sim.plan --seed (same seeds = same scenes)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)
    big = 10 ** 9
    table = set_table(argparse.Namespace(train_runs=big, gate_runs=big, val_runs=big, test_runs=big, seed=a.seed))
    chosen = select(table, a.sets)
    if not chosen:
        raise SystemExit(f"no set matches {a.sets} (see python -m sim.plan --list)")
    cam = Camera(CALIB[a.camera], scale=a.scale)
    tasks = []
    for name, spec, _, seed, sid, level, family in chosen:
        runs = dict(train=a.train_runs, gate=a.val_runs, val=a.val_runs).get(name, a.test_runs)
        training = sid in ("train", "gate", "val")
        d = os.path.join(a.out, name)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(dict(set=name, scenario=sid, level=level, scenes=runs, seed=a.seed * 1000 + seed,
                           camera=a.camera, K=cam.K.tolist(), D=cam.D.tolist(), size=[cam.W, cam.H], scale=a.scale,
                           obstacles=a.obstacles if training else 0, distractors=a.distractors,
                           appearance=asdict(appearance())), f, indent=2)
        for k, start in enumerate(range(0, runs, SHARD)):
            tasks.append((name, asdict(spec), min(SHARD, runs - start), start, a.seed * 1000 + seed,
                          os.path.join(d, f"shard_{k:04d}.pkl"), a.model, CALIB[a.camera], a.scale, a.jpeg_quality,
                          a.obstacles if training else 0, a.distractors))
    print(f"{len(chosen)} set(s), {sum(t[2] for t in tasks)} scenes in {len(tasks)} shards on "
          f"{min(a.jobs, len(tasks))} process(es)", flush=True)
    with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
        for k, (name, path) in enumerate(pool.imap_unordered(render_shard, tasks)):
            if (k + 1) % 10 == 0 or k + 1 == len(tasks):
                print(f"  {k + 1}/{len(tasks)} shards ({name})", flush=True)
    print("done")


# ====================================================================
# Loading
# ====================================================================

class E2EDataset:
    """Scenes of one or more set folders (simulated, e2e.data; real, e2e.real). Item: dict with
    images (N, 3, H, W) float32 in [0, 1], grids (N, len(HEIGHTS_M), G, G, 2), valid (N, len(HEIGHTS_M), G, G),
    view (N, 3, G, G), poses (N, 3), heat (G, G), off (2, G, G), mask (G, G), boxes (B, 2), seen (B,), run, set.

    seen: the boxes with at least min_visible_px visible pixels in one image (simulated sets); the targets hold
    only those. Real scenes count every box as seen.

    The shards are read when needed and only the last CACHE stay in memory (each data loader process keeps its
    own); ShardSampler keeps the reads few while shuffling."""

    CACHE = 8

    def __init__(self, set_dirs, max_scenes=None):
        self.index, self.meta, self.shards = [], {}, []
        for d in set_dirs:
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            self.meta[meta["set"]] = meta
            for k, path in enumerate(sorted(glob.glob(os.path.join(d, "shard_*.pkl")))):
                n = meta["shard_sizes"][k] if "shard_sizes" in meta else min(SHARD, meta["scenes"] - k * SHARD)
                self.index += [(len(self.shards), j) for j in range(n)]
                self.shards.append((path, meta["set"]))
        if max_scenes:
            self.index = self.index[:max_scenes]
        self.cells = geo.cell_centres()
        self._cache = {}
        self.pitch_noise_deg = 0.0        # training: the pitch and yaw given to the network are off by N(0, sd)
        self.yaw_noise_deg = 0.0          # from those of the image, as the real calibration is
        self.min_visible_px = MIN_VISIBLE_PX

    def __len__(self):
        return len(self.index)

    def scene(self, i):
        shard, j = self.index[i]
        if shard not in self._cache:
            if len(self._cache) >= self.CACHE:
                self._cache.pop(next(iter(self._cache)))
            path, set_name = self.shards[shard]
            with open(path, "rb") as f:
                scenes = pickle.load(f)
            for s in scenes:
                s["set"] = set_name
            self._cache[shard] = scenes
        return self._cache[shard][j]

    def __getstate__(self):                     # data loader processes start with an empty cache
        return dict(self.__dict__, _cache={})

    def __getitem__(self, i):
        s = self.scene(i)
        meta = self.meta[s["set"]]
        K, D, size = np.array(meta["K"]), np.array(meta["D"]), tuple(meta["size"])
        imgs, grids, valid, view, poses = [], [], [], [], []
        rng = np.random.default_rng() if (self.pitch_noise_deg or self.yaw_noise_deg) else None
        for k, jpg in enumerate(s["jpgs"]):
            Kk = np.array(s["K"][k]) if "K" in s else K              # real frames: each robot its own camera
            Dk = np.array(s["D"][k]) if "D" in s else D
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0)
            pose, pitch = np.asarray(s["poses"][k], float), float(s["pitch"][k])
            if rng is not None:
                pitch += math.radians(rng.normal(0, self.pitch_noise_deg))
                pose = pose + [0.0, 0.0, math.radians(rng.normal(0, self.yaw_noise_deg))]
            g, v, w = geo.sampling_grid(pose, pitch, float(s["roll"][k]), float(s["height"]), Kk, Dk, size,
                                        self.cells)
            grids.append(g)
            valid.append(v)
            view.append(w)
            poses.append(pose)
        seen = seen_boxes(s, self.min_visible_px)
        heat, off, mask = geo.heatmap_targets(s["boxes"][seen])
        return dict(images=np.stack(imgs), grids=np.stack(grids), valid=np.stack(valid), view=np.stack(view),
                    poses=np.asarray(poses, np.float32), heat=heat, off=off, mask=mask, boxes=s["boxes"],
                    seen=seen, run=s["run"], set=s["set"])


def seen_boxes(scene, min_px):
    """(B,) bool: the boxes with at least min_px visible pixels in at least one image of the scene."""
    n = len(scene["boxes"])
    px = scene.get("box_pixels")
    if px is None or np.size(px) == 0:
        return np.ones(n, bool)
    return np.asarray(px).reshape(-1, n).max(0) >= min_px


class ShardSampler:
    """Training order: the shards in random order, taken MIX at a time, the scenes of those MIX shards shuffled."""

    MIX = 4

    def __init__(self, ds, seed=0):
        self.ds, self.epoch, self.seed = ds, 0, seed

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        by_shard = {}
        for i, (shard, _) in enumerate(self.ds.index):
            by_shard.setdefault(shard, []).append(i)
        order = rng.permutation(sorted(by_shard))
        for k in range(0, len(order), self.MIX):
            group = np.concatenate([by_shard[s] for s in order[k:k + self.MIX]])
            yield from rng.permutation(group).tolist()


def collate(items):
    """A batch of scenes: the images of all robots stacked, with the index of their scene."""
    import torch
    out = dict(scene=torch.tensor(np.concatenate([np.full(len(it["images"]), k) for k, it in enumerate(items)]),
                                  dtype=torch.long), n_scenes=len(items),
               boxes=[it["boxes"] for it in items], seen=[it["seen"] for it in items],
               run=[it["run"] for it in items], set=[it["set"] for it in items])
    for key in ("images", "grids", "valid", "view", "poses"):
        out[key] = torch.from_numpy(np.concatenate([it[key] for it in items]))
    for key in ("heat", "off", "mask"):
        out[key] = torch.from_numpy(np.stack([it[key] for it in items]))
    return out


if __name__ == "__main__":
    build()
