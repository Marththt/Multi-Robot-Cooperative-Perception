"""Simulated instance-segmentation data set (imgsim/render.py) and YOLO-seg fine-tuning to segment the boxes.
Pure sim-to-real: no real laboratory frame is used for training.

Every simulated scene (sim/plan.py layouts, random obstacles and distractor objects, random appearance and a
camera pitch in [8, 32] deg) is rendered from every robot; the label of every box is its exact visible region
(occluded by nearer boxes, obstacles, robots and the gripper), written as a YOLO polygon.

    python -m yolo_gnn.train_seg --data-dir data/yolo_sim --out results/yolo_seg
      -> results/yolo_seg/box_seg/weights/best.pt
"""

import argparse
import copy
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import asdict, replace

import cv2
import numpy as np

CALIB = dict(P0="config/checkerboard_calib_P0.json", P1="config/checkerboard_calib_P1.json")


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


def render_instance_ids(renderer, scene, k, rng, style):
    """Exact per-box instance ID map (1..B) in the distorted camera frame, with proper occlusion by
    closer boxes, obstacles, robots and gripper."""
    from imgsim.render import _u, camera_rotation
    from sim.geometry import face_normal

    a = renderer.app
    x, y, yaw = scene.captures[k].pose
    pitch, roll = np.radians(_u(rng, a.pitch_deg)), np.radians(_u(rng, a.roll_deg))
    C = np.array([x, y, a.camera_height_m])
    R = camera_rotation(yaw, pitch, roll)

    canvas = np.zeros((renderer.cam.Hi, renderer.cam.Wi), np.float32)
    objects = []
    half = a.box_side_m / 2
    apothem = a.box_side_m / (2 * np.sqrt(3))

    for j, (c, th) in enumerate(zip(scene.boxes, scene.thetas)):
        for f in range(3):
            nrm = face_normal(th, f)
            nv = np.array([np.cos(nrm), np.sin(nrm)])
            fc = np.asarray(c, float) + apothem * nv
            if np.dot(nv, np.array([x, y]) - fc) <= 0:
                continue
            t = np.array([-nv[1], nv[0]])
            left, right = fc - t * half, fc + t * half
            quad = np.array([[*left, a.box_height_m], [*right, a.box_height_m], [*right, 0.0], [*left, 0.0]])
            depth = float(np.hypot(*(np.asarray(c) - [x, y])))
            objects.append((depth + 0.01 * f, "face", (j + 1, quad)))

    for q, poly in enumerate(scene.obstacles):
        poly = np.asarray(poly, float)
        oc, r = poly.mean(0), float(np.max(np.hypot(*(poly - poly.mean(0)).T)))
        if "obstacle_seed" in style and q in style.get("obstacle_looks", {}):
            objects.append((float(np.hypot(*(oc - [x, y]))), "solid", style["obstacle_looks"][q]))
        else:
            objects.append((float(np.hypot(*(oc - [x, y]))), "cylinder", (oc, r)))

    if a.draw_robots:
        L_, W_, H_ = a.robot_size_m
        for q, cap in enumerate(scene.captures):
            if q == k:
                continue
            qx, qy, qyaw = cap.pose
            bc = np.array([qx, qy]) - a.robot_body_behind_m * np.array([np.cos(qyaw), np.sin(qyaw)])
            objects.append((float(np.hypot(*(bc - [x, y]))), "robot", (bc, qyaw, L_, W_, H_)))

    objects.sort(key=lambda o: -o[0])
    for _, kind, data in objects:
        if kind == "face":
            box_id, quad = data
            uv, z, _ = renderer.project(quad, C, R)
            if (z >= 0.05).all():
                hull = cv2.convexHull(np.float32(uv)).reshape(-1, 2)
                cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), float(box_id), lineType=cv2.LINE_8)
        elif kind == "cylinder":
            oc, r = data
            ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            ring = np.stack([oc[0] + r * np.cos(ang), oc[1] + r * np.sin(ang)], 1)
            pts = np.vstack([np.column_stack([ring, np.zeros(32)]),
                             np.column_stack([ring, np.full(32, a.plain_obstacle_height_m)])])
            uv, z, _ = renderer.project(pts, C, R)
            if (z >= 0.05).all():
                hull = cv2.convexHull(np.float32(uv)).reshape(-1, 2)
                cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), 0.0, lineType=cv2.LINE_8)
        elif kind == "solid":
            F = np.asarray(data["footprint"], float)
            z0 = data.get("z0", 0.0)
            pts = np.vstack([np.column_stack([F, np.full(len(F), z0)]),
                             np.column_stack([F, np.full(len(F), z0 + data["height"])])])
            uv, z, _ = renderer.project(pts, C, R)
            if (z >= 0.05).all():
                hull = cv2.convexHull(np.float32(uv)).reshape(-1, 2)
                cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), 0.0, lineType=cv2.LINE_8)
        elif kind == "robot":
            bc, qyaw, L_, W_, H_ = data
            fwd = np.array([np.cos(qyaw), np.sin(qyaw)])
            side = np.array([-fwd[1], fwd[0]])
            corners = [bc + sx * fwd * L_ / 2 + sy * side * W_ / 2 for sx in (-1, 1) for sy in (-1, 1)]
            pts = np.array([[*p, zz] for p in corners for zz in (0.02, H_)])
            uv, z, _ = renderer.project(pts, C, R)
            if (z >= 0.05).all():
                hull = cv2.convexHull(np.float32(uv)).reshape(-1, 2)
                cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), 0.0, lineType=cv2.LINE_8)

    ids = cv2.remap(canvas, renderer.cam.map_x, renderer.cam.map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ids = np.round(ids).astype(np.int32)
    if renderer.grip_mask is not None:
        ids[renderer.grip_mask[..., 0] > 0.5] = 0
    return ids


def render_yolo_scene(task):
    (split, i, spec_dict, seed, out_dir, model_path, calib, scale, obstacles, distractors, min_px) = task
    from imgsim.render import Appearance, Renderer
    from sim.model import SensorModel
    from sim.plan import PlanSpec, plan_scene, run_model

    spec = PlanSpec(**spec_dict)
    m = run_model(SensorModel.load(model_path), spec)

    app = Appearance(
        pitch_deg=[8.0, 32.0],
        roll_deg=[-1.5, 1.5],
        floor_grey=[30.0, 90.0],
        wall_grey=[120.0, 220.0],
        light_ambient=[0.35, 0.85],
        exposure=[0.30, 1.05],
        vignetting=0.35,
        obstacle_style="random",
        visual_style="random",
        light_obstacle_share=0.30,
    )
    renderer = Renderer(calib, app, seed=seed, scale=scale)

    scene_seed = seed * 1_000_003 + i
    rng = np.random.default_rng(scene_seed)
    family = spec.families[i % len(spec.families)]
    h = int(np.random.default_rng(scene_seed + 104_729).integers(0, obstacles + 1)) if obstacles else 0
    spec_i = replace(spec, hidden_robots=h)
    scene, _ = plan_scene(spec_i, family, rng, split, f"{split}_{i:05d}", m)
    if distractors:
        add_distractors(scene, distractors, np.random.default_rng(scene_seed + 15_485_863))

    rrng = np.random.default_rng(scene_seed + 7_919)
    style = renderer.scene_style(rrng)

    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    written = 0

    for k, cap in enumerate(scene.captures):
        same = copy.deepcopy(rrng)
        rgb, _, _ = renderer.render(scene, k, rrng, style)
        ids = render_instance_ids(renderer, scene, k, same, style)

        H, W = rgb.shape[:2]
        lines = []
        for box_id in range(1, len(scene.boxes) + 1):
            bmask = (ids == box_id).astype(np.uint8)
            if int(bmask.sum()) < min_px:
                continue
            cnts, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            for cnt in cnts:
                if cv2.contourArea(cnt) < min_px:
                    continue
                approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True).reshape(-1, 2)
                if len(approx) < 3:
                    continue
                coords = [f"{np.clip(x / W, 0.0, 1.0):.5f} {np.clip(y / H, 0.0, 1.0):.5f}" for x, y in approx]
                lines.append("0 " + " ".join(coords))

        stem = f"{scene.run_id}_{cap.robot}"
        cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    return written


def write_yaml(data_dir):
    yaml_path = os.path.abspath(os.path.join(data_dir, "dataset.yaml"))
    with open(yaml_path, "w") as f:
        # forward slashes: a Windows path in double quotes is read as escapes ("\G...") by the YAML parser
        f.write(f"path: '{os.path.abspath(data_dir).replace(os.sep, '/')}'\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: box\n")
    return yaml_path


def generate_dataset(a):
    from sim.plan import select, set_table

    if os.path.isdir(a.data_dir):                     # never deleted: set aside in _to_delete/
        dest = os.path.join("_to_delete", f"{os.path.basename(os.path.normpath(a.data_dir))}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs("_to_delete", exist_ok=True)
        shutil.move(a.data_dir, dest)
        print(f"previous data set moved to {dest}")
    for split in ("train", "val"):
        os.makedirs(os.path.join(a.data_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.data_dir, "labels", split), exist_ok=True)

    big = 10 ** 9
    table = set_table(argparse.Namespace(train_runs=big, gate_runs=big, val_runs=big, test_runs=big, seed=a.seed))
    chosen = {name: (spec, seed) for name, spec, _, seed, *_ in select(table, "train,val")}

    tasks = []
    for split, n_scenes in (("train", a.train_scenes), ("val", a.val_scenes)):
        spec, s_seed = chosen[split]
        full_seed = a.seed * 1000 + s_seed
        for i in range(n_scenes):
            tasks.append((split, i, asdict(spec), full_seed, a.data_dir, a.sensor_model, CALIB[a.camera],
                          a.scale, a.obstacles, a.distractors, a.min_px))

    print(f"Generating {len(tasks)} simulated scenes ({a.train_scenes} train, {a.val_scenes} val) "
          f"at scale {a.scale} on {a.jobs} workers...", flush=True)
    total_imgs = 0
    with multiprocessing.Pool(a.jobs) as pool:
        for idx, count in enumerate(pool.imap_unordered(render_yolo_scene, tasks, chunksize=10)):
            total_imgs += count
            if (idx + 1) % 100 == 0 or idx + 1 == len(tasks):
                print(f"  {idx + 1}/{len(tasks)} scenes ({total_imgs} images)", flush=True)

    yaml_path = write_yaml(a.data_dir)
    print(f"Dataset ready: {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/yolo_sim", help="output folder for the YOLO dataset")
    ap.add_argument("--train-scenes", type=int, default=1500, help="simulated training scenes (~9,000 images)")
    ap.add_argument("--val-scenes", type=int, default=200, help="simulated validation scenes (~1,200 images)")
    ap.add_argument("--scale", type=float, default=0.75, help="0.75 -> 960x540 images")
    ap.add_argument("--min-px", type=int, default=25, help="minimum visible pixels to label a far box")
    ap.add_argument("--obstacles", type=int, default=2, help="0..N occluding obstacles per box")
    ap.add_argument("--distractors", type=int, default=4, help="0..N random distractor objects in the room")
    ap.add_argument("--camera", choices=sorted(CALIB), default="P0")
    ap.add_argument("--sensor-model", default="config/sensor_model.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-gen", action="store_true", help="reuse existing dataset in --data-dir")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="base Ultralytics segmentation model")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | mps | 0 | cpu")
    ap.add_argument("--out", default="results/yolo_seg", help="folder where the trained weights are saved")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)

    train_imgs = os.path.join(a.data_dir, "images", "train")
    if a.skip_gen and os.path.isdir(train_imgs) and os.listdir(train_imgs):
        yaml_path = write_yaml(a.data_dir)
        print(f"Reusing existing images, updated YAML: {yaml_path}")
    else:
        yaml_path = generate_dataset(a)

    import torch
    from ultralytics import YOLO

    if a.device == "auto":
        a.device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = YOLO(a.model)
    res = model.train(
        data=yaml_path,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        seed=a.seed,
        workers=min(8, a.jobs),
        device=a.device,
        project=os.path.abspath(a.out),
        name="box_seg",
        exist_ok=True,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.6,
        erasing=0.1,
    )
    best = os.path.join(str(res.save_dir), "weights", "best.pt")
    print(f"\nTraining complete! Best weights saved at: {best}")


if __name__ == "__main__":
    main()
