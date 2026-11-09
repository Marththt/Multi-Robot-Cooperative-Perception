"""Training and test data of the image-based network: the scenes of sim/plan.py rendered at low resolution.

Every scene of a set is drawn by sim.plan.plan_scene with the seed of sim/plan.py, so scene i of
data/bev/<set>/ is scene i of data/plan/<set>/ (and of data/imgsim/<set>/): the image-based network, MHT and
the GNN can be compared scene by scene. For each robot the camera image is rendered (imgsim/render.py) at
--scale of the real 1280 x 720 (0.2 -> 256 x 144) and stored as JPEG, with the camera pose, pitch and roll.
Nothing else is kept from the renderer: the network gets the images and the poses.

Shards of SHARD scenes: data/bev/<set>/shard_<k>.pkl (a list of scenes), plus meta.json (camera matrix and
distortion at the rendered size, image size, appearance). Loading (BevDataset) decodes the images and
computes, from the poses, where each floor cell falls in each image (bevnet/geometry.py).

    python -m bevnet.data --sets train,val,G0,E3,S1                # 50,000 + 2,000 + 16 x 1,000 scenes
    python -m bevnet.data --sets train --runs 2000 --out data/bev_small/
    python -m bevnet.data --sets train,val,G0,E3,S1 --no-markers --out data/bev_nomarkers/   # boxes without markers
    python -m bevnet.data --sets train,val --obstacles 2 --out data/bev_obst/   # training scenes with the E3 cylinders
    python -m bevnet.data --sets train,val --obstacles 2 --distractors 4 --obstacle-style random \
        --obstacle-shapes cylinder,box,prism --out data/bev_rand/      # obstacles of random shape, colour, pattern
"""

import argparse
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

from bevnet import geometry as geo

SHARD = 250
CALIB = dict(P0="config/checkerboard_calib_P0.json", P1="config/checkerboard_calib_P1.json")


def render_shard(task):
    (name, spec_dict, runs, start, seed, out_path, model_path, calib, scale, app_path, quality, no_markers, obstacles,
     obstacle_style, obstacle_shapes, distractors, visual, masks) = task
    from imgsim.render import Appearance, Renderer
    from sim.model import SensorModel
    from sim.plan import PlanSpec, plan_scene, run_model
    spec = PlanSpec(**spec_dict)
    m = run_model(SensorModel.load(model_path), spec)
    app = Appearance.load(app_path)
    if no_markers:
        app.draw_markers = False
    app.obstacle_style = obstacle_style
    app.visual_style = visual
    app.mask_only = bool(masks)
    if obstacle_shapes:
        app.obstacle_shapes = list(obstacle_shapes)
    renderer = Renderer(calib, app, seed=seed, scale=scale)
    scenes = []
    for i in range(start, start + runs):
        scene_seed = seed * 1_000_003 + i                       # the seed of sim/plan.py for this scene
        rng = np.random.default_rng(scene_seed)
        family = spec.families[i % len(spec.families)]
        spec_i = spec
        if obstacles:                                           # occluding cylinders as in E3, h drawn per scene
            h = int(np.random.default_rng(scene_seed + 104_729).integers(0, obstacles + 1))
            spec_i = replace(spec, hidden_robots=h)
        scene, meta = plan_scene(spec_i, family, rng, name, f"{name}_{i:05d}", m)
        if distractors:
            add_distractors(scene, distractors, np.random.default_rng(scene_seed + 15_485_863))
        rrng = np.random.default_rng(scene_seed + 7_919)        # as imgsim/generate.py
        style = renderer.scene_style(rrng)
        jpgs, poses, pitch, roll, seen_px = [], [], [], [], []
        for k, cap in enumerate(scene.captures):
            img, _markers, info = renderer.render(scene, k, rrng, style)
            if masks:                                           # binary: lossless and small
                ok, buf = cv2.imencode(".png", img[..., 0])
            else:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            jpgs.append(buf.tobytes())
            poses.append(cap.pose)
            pitch.append(np.radians(info["pitch_deg"]))
            roll.append(np.radians(info["roll_deg"]))
            if "box_pixels" in info:
                seen_px.append(info["box_pixels"])
        extra = dict(box_pixels=np.asarray(seen_px, np.int32)) if seen_px else {}
        scenes.append(dict(**extra, run=scene.run_id, family=family, boxes=np.asarray(scene.boxes, np.float32),
                           poses=np.asarray(poses, np.float32), pitch=np.asarray(pitch, np.float32),
                           roll=np.asarray(roll, np.float32), height=float(renderer.app.camera_height_m),
                           camera=os.path.basename(calib).split("_")[-1].split(".")[0], jpgs=jpgs,
                           n_robots=meta["n_robots"], n_boxes=meta["n_boxes"]))
    with open(out_path, "wb") as f:
        pickle.dump(scenes, f, protocol=pickle.HIGHEST_PROTOCOL)
    return name, out_path


def add_distractors(scene, n_max, rng, tries=200):
    """0..n_max free-standing objects (footprint radius 6-20 cm) anywhere within 2 m of the centre, clear of
    the boxes, the robots and the other obstacles: things in the room that are not boxes."""
    from sim.plan import cylinder
    taken = [(np.asarray(b, float), 0.35) for b in scene.boxes] + \
            [(np.asarray(c.pose[:2], float), 0.30) for c in scene.captures] + \
            [(np.asarray(p, float).mean(0), float(np.max(np.hypot(*(np.asarray(p) - np.asarray(p).mean(0)).T))))
             for p in scene.obstacles]
    for _ in range(int(rng.integers(0, n_max + 1))):
        for _ in range(tries):
            r = rng.uniform(0.06, 0.20)
            a, d = rng.uniform(0, 2 * np.pi), 2.0 * np.sqrt(rng.uniform())
            c = np.array([d * np.cos(a), d * np.sin(a)])
            if all(np.hypot(*(c - q)) > r + rq + 0.05 for q, rq in taken):
                scene.obstacles.append(cylinder(c, r))
                taken.append((c, r))
                break


def build(argv=None):
    from imgsim.render import Appearance, Camera
    from sim.plan import select, set_table
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True, help="as sim.plan --sets: train, val, gate, G0, E3, S1, set names ...")
    ap.add_argument("--runs", type=int, default=None, help="scenes per set (default: as in sim/plan.py)")
    ap.add_argument("--out", default="data/bev/")
    ap.add_argument("--scale", type=float, default=0.2, help="image size / 1280 x 720")
    ap.add_argument("--camera", choices=sorted(CALIB), default="P0")
    ap.add_argument("--appearance", default=None, help="JSON with imgsim.render.Appearance fields to change")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--no-markers", action="store_true", help="plain boxes: no paper square, no ArUco marker")
    ap.add_argument("--obstacle-style", choices=["plain", "random"], default="plain",
                    help="random: every obstacle with its own shape, height, colour and pattern")
    ap.add_argument("--obstacle-shapes", default=None,
                    help="with --obstacle-style random: comma list of cylinder,box,prism,hexagon,blob (default all)")
    ap.add_argument("--distractors", type=int, default=0, help="0..N free-standing objects per scene (any set)")
    ap.add_argument("--masks", action="store_true",
                    help="boxes only: white box silhouettes on black, as a segmentation of the real frames gives")
    ap.add_argument("--visual", choices=["plain", "random"], default="plain",
                    help="random: robots with wheels and arm, windows, light spots, desks and screens, wider greys")
    ap.add_argument("--obstacles", type=int, default=0,
                    help="training / validation sets only: per scene, h in 0..OBSTACLES occluding cylinders per box (E3)")
    ap.add_argument("--model", default="config/sensor_model.json", help="only for the scene layout")
    ap.add_argument("--seed", type=int, default=1, help="as sim.plan --seed (same seeds = same scenes)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)
    big = 10 ** 9
    table = set_table(argparse.Namespace(train_runs=big, gate_runs=big, val_runs=big, test_runs=big, seed=a.seed))
    plan_runs = dict(train=50000, gate=10000, val=2000)
    chosen = select(table, a.sets)
    if not chosen:
        raise SystemExit(f"no set matches {a.sets} (see python -m sim.plan --list)")
    cam = Camera(CALIB[a.camera], scale=a.scale)
    from imgsim.render import OBSTACLE_SHAPES
    shapes = a.obstacle_shapes.split(",") if a.obstacle_shapes else None
    if shapes and set(shapes) - set(OBSTACLE_SHAPES):
        raise SystemExit(f"unknown shapes {set(shapes) - set(OBSTACLE_SHAPES)} (known: {OBSTACLE_SHAPES})")
    tasks = []
    for name, spec, _, seed, sid, level, family in chosen:
        if sid in ("E1", "E2"):
            print(f"  note: {name} changes rates of the sensor file, which the renderer does not use")
        runs = a.runs or plan_runs.get(name, 1000)
        d = os.path.join(a.out, name)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(dict(set=name, scenario=sid, level=level, scenes=runs, seed=a.seed * 1000 + seed,
                           camera=a.camera, K=cam.K.tolist(), D=cam.D.tolist(), size=[cam.W, cam.H],
                           scale=a.scale, obstacles=a.obstacles if sid in ('train', 'gate', 'val') else 0,
                           obstacle_style=a.obstacle_style, obstacle_shapes=shapes, distractors=a.distractors,
                           visual=a.visual, masks=a.masks,
                           appearance=dict(asdict(Appearance.load(a.appearance)),
                                                          draw_markers=not a.no_markers)), f, indent=2)
        for k, start in enumerate(range(0, runs, SHARD)):
            tasks.append((name, asdict(spec), min(SHARD, runs - start), start, a.seed * 1000 + seed,
                          os.path.join(d, f"shard_{k:04d}.pkl"), a.model, CALIB[a.camera], a.scale, a.appearance,
                          a.jpeg_quality, a.no_markers, a.obstacles if sid in ('train', 'gate', 'val') else 0,
                          a.obstacle_style, shapes, a.distractors, a.visual, a.masks))
    print(f"{len(chosen)} set(s), {sum(t[2] for t in tasks)} scenes in {len(tasks)} shards on "
          f"{min(a.jobs, len(tasks))} process(es)", flush=True)
    with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
        for k, (name, path) in enumerate(pool.imap_unordered(render_shard, tasks)):
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                print(f"  {k + 1}/{len(tasks)} shards ({name})", flush=True)
    print("done")


# ====================================================================
# Loading
# ====================================================================

class BevDataset:
    """Scenes of one or more set folders of build(). Item: dict with
    images (N, 3, H, W) float32 in [0, 1], grids (N, len(HEIGHTS_M), G, G, 2), valid (N, len(HEIGHTS_M), G, G),
    view (N, 3, G, G), heat (G, G), off (2, G, G), mask (G, G), boxes (B, 2), seen (B,), run.

    seen: the boxes visible in at least one image (MIN_VISIBLE_PX pixels or more; sets rendered as masks record
    the visible pixels of every box). The targets hold only those: a box no robot can see is not asked for.
    Sets without that record (real scenes, older sets) count every box as seen.

    The shards are read when needed and only the last CACHE stay in memory (each data loader process keeps its
    own), so the training set never has to fit in memory; ShardSampler keeps the reads few while shuffling."""

    CACHE = 8
    MIN_VISIBLE_PX = 12               # of a 256 x 144 image: a sliver of a far box, a corner of a near one

    def __init__(self, set_dirs, max_scenes=None):
        self.index, self.meta, self.shards = [], {}, []
        for d in set_dirs:
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            self.meta[meta["set"]] = meta
            for k, path in enumerate(sorted(glob.glob(os.path.join(d, "shard_*.pkl")))):
                n = min(SHARD, meta["scenes"] - k * SHARD)
                self.index += [(len(self.shards), j) for j in range(n)]
                self.shards.append((path, meta["set"]))
        if max_scenes:
            self.index = self.index[:max_scenes]
        self.cells = geo.cell_centres()
        self._cache = {}
        self.pitch_noise_deg = 0.0        # training: the pitch and yaw given to the network are off by N(0, sd)
        self.yaw_noise_deg = 0.0          # from those of the image, so that it learns to live with real errors
        self.height_noise_m = 0.0         # same for the camera height (a wrong height scales all floor distances)
        self.mask_noise = 0.0             # training on masks: imperfect segmentation (see noisy_mask)
        self.min_visible_px = self.MIN_VISIBLE_PX

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
        imgs, grids, valid, view = [], [], [], []
        rng = np.random.default_rng() if (self.pitch_noise_deg or self.yaw_noise_deg or self.height_noise_m
                                          or self.mask_noise) else None
        for k, jpg in enumerate(s["jpgs"]):
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if self.mask_noise and rng is not None:
                img = noisy_mask(img, rng, self.mask_noise)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0)
            Kk = np.array(s["K"][k]) if "K" in s else K              # real frames: each robot its own camera
            Dk = np.array(s["D"][k]) if "D" in s else D
            pose, pitch, height = np.asarray(s["poses"][k], float), float(s["pitch"][k]), float(s["height"])
            if rng is not None:
                pitch += math.radians(rng.normal(0, self.pitch_noise_deg))
                pose = pose + [0.0, 0.0, math.radians(rng.normal(0, self.yaw_noise_deg))]
                height += rng.normal(0, self.height_noise_m)
            g, v, w = geo.sampling_grid(pose, pitch, float(s["roll"][k]), height, Kk, Dk, size, self.cells)
            grids.append(g)
            valid.append(v)
            view.append(w)
        seen = seen_boxes(s, self.min_visible_px)
        heat, off, mask = geo.heatmap_targets(s["boxes"][seen])
        return dict(images=np.stack(imgs), grids=np.stack(grids), valid=np.stack(valid), view=np.stack(view),
                    heat=heat, off=off, mask=mask, boxes=s["boxes"], seen=seen, run=s["run"], set=s["set"])


def seen_boxes(scene, min_px):
    """(B,) bool: the boxes with at least min_px visible pixels in at least one image of the scene."""
    n = len(scene["boxes"])
    px = scene.get("box_pixels")
    if px is None or len(px) == 0:
        return np.ones(n, bool)
    return np.asarray(px).reshape(-1, n).max(0) >= min_px


def noisy_mask(img, rng, level):
    """A box mask (white on black) made imperfect as a real segmentation is: edges grown or shrunk by a few
    pixels, small holes, small false blobs, and now and then a whole box missed. level 1 = the default amounts."""
    m = (img[..., 0] > 127).astype(np.uint8)
    H, W = m.shape
    k = int(rng.integers(0, 3))
    if k:
        kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        m = cv2.dilate(m, kernel) if rng.random() < 0.5 else cv2.erode(m, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    for c in range(1, n):                                  # a single small box missed (not a group of boxes)
        if stats[c, cv2.CC_STAT_AREA] < 0.02 * H * W and rng.random() < 0.05 * level:
            m[lab == c] = 0
    for _ in range(rng.poisson(1.0 * level)):              # holes
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        w, h = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        m[y:y + h, x:x + w] = 0
    for _ in range(rng.poisson(0.5 * level)):              # false blobs
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, int(0.8 * H)))
        cv2.ellipse(m, (cx, cy), (int(rng.integers(2, 12)), int(rng.integers(3, 18))), 0, 0, 360, 1, -1)
    return np.repeat((m * 255)[..., None], 3, axis=2)


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
    for key in ("images", "grids", "valid", "view"):
        out[key] = torch.from_numpy(np.concatenate([it[key] for it in items]))
    for key in ("heat", "off", "mask"):
        out[key] = torch.from_numpy(np.stack([it[key] for it in items]))
    return out


if __name__ == "__main__":
    build()
