"""Renders the frames of simulated sets (sim/scaling.py) and segments them with the YOLO-seg model, as
yolo_gnn/segment_real.py does for the real frames.

Every scene of a set is redrawn from its geometry.jsonl (boxes, yaws, camera poses) by imgsim/render.py: the
other robots, the room around the area (walls 1.0-2.5 m beyond its sides), the gripper of the own robot. The
look is the default one of the renderer (the one matched to the real frames), not the randomized look the YOLO
model was trained on. Each robot takes --readings frames of its stop: one view (pitch and roll drawn once per
stop, as a robot standing still) and independent camera effects (exposure, blur, noise, JPEG) per reading.

The frames are not kept: one label image per frame is cached, <cache>/<set>/<run>/<robot>_r<k>_ids.png
(0 background, 1..n one index per box), plus <cache>/<set>/camera.json (camera matrix, distortion, image size,
the nominal pitch that the rays assume). Scenes already in the cache are skipped. A sheet of overlays is written
next to the cache (fig_yolo_sim.jpg).

    python -m yolo_gnn.segment_sim --sets "data/scale/tune" "data/scale/S*" \\
        --weights results/yolo_seg/box_seg/weights/best.pt --cache data/yolo_sim_eval/cache --device 0
"""

import argparse
import glob
import json
import multiprocessing
import os
import sys
from dataclasses import replace

import cv2
import numpy as np

CALIB = dict(P0="config/checkerboard_calib_P0.json", P1="config/checkerboard_calib_P1.json")
ROOM_BEYOND_M = (1.0, 2.5)          # walls this far beyond the sides of the area

_R = None                           # renderer of a worker process


def cache_dir(cache, set_dir, run):
    return os.path.join(cache, os.path.basename(os.path.normpath(set_dir)), run)


def frame_path(cache, set_dir, run, robot, reading):
    return os.path.join(cache_dir(cache, set_dir, run), f"{robot}_r{reading}_ids.png")


def _init(calib, scale, seed):
    global _R
    from imgsim.render import Appearance, Renderer
    _R = (Renderer(calib, Appearance(), seed=seed, scale=scale), Appearance())


def render_scene(task):
    """JPEG bytes of every frame of one scene: (run, [(robot, reading, bytes)])."""
    g, readings, quality = task
    from sim.scaling import scene_from_geometry
    renderer, base = _R
    half = float(g["side_m"]) / 2
    renderer.app = replace(base, room_half_m=[half + ROOM_BEYOND_M[0], half + ROOM_BEYOND_M[1]])
    scene = scene_from_geometry(g)
    style = renderer.scene_style(np.random.default_rng([int(g["seed"]), 7919]))
    out = []
    for k, cap in enumerate(scene.captures):
        r = np.random.default_rng([int(g["seed"]), 7919, k])
        clean, _, _ = renderer.render_clean(scene, k, r, dict(style))
        for q in range(readings):
            img = renderer.camera_effects(clean, r)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            out.append((cap.robot, q, buf.tobytes()))
    return g["run"], out


def write_camera(path, calib, scale, pitch_deg):
    from imgsim.render import Camera
    cam = Camera(calib, scale=scale)
    with open(path, "w") as f:
        json.dump(dict(camera_matrix=cam.K.tolist(), dist_coeffs=cam.D.tolist(), image_size=[cam.W, cam.H],
                       pitch_deg=pitch_deg, calib=calib, scale=scale), f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="set folders of sim/scaling.py (glob patterns)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", default="data/yolo_sim_eval/cache")
    ap.add_argument("--readings", type=int, default=3, help="frames per robot and stop")
    ap.add_argument("--max-scenes", type=int, default=None, help="only the first N scenes of each set")
    ap.add_argument("--camera", choices=sorted(CALIB), default="P0")
    ap.add_argument("--scale", type=float, default=1.0, help="image scale (1.0: 1280 x 720, as the real frames)")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=1.0, help="the simulated frames need no gamma lift")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=11, help="textures of the renderer")
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)

    from imgsim.render import Appearance
    from sim.scaling import read_geometry
    from yolo_gnn.segment import YoloSegmenter, overlay

    sets = sorted({d for p in a.sets for d in glob.glob(p) if os.path.exists(os.path.join(d, "geometry.jsonl"))})
    if not sets:
        raise SystemExit(f"no set with geometry.jsonl matches {a.sets}")
    seg = YoloSegmenter(a.weights, device=a.device, conf=a.conf, gamma=a.gamma)
    pitch = float(np.mean(Appearance().pitch_deg))
    calib = CALIB[a.camera]
    sheet = []
    with multiprocessing.Pool(a.jobs, initializer=_init, initargs=(calib, a.scale, a.seed)) as pool:
        for set_dir in sets:
            name = os.path.basename(os.path.normpath(set_dir))
            os.makedirs(os.path.join(a.cache, name), exist_ok=True)
            write_camera(os.path.join(a.cache, name, "camera.json"), calib, a.scale, pitch)
            geo = list(read_geometry(set_dir).values())[:a.max_scenes]
            todo = [g for g in geo
                    if not all(os.path.exists(frame_path(a.cache, set_dir, g["run"], f"P{k}", a.readings - 1))
                               for k in range(len(g["poses"])))]
            print(f"{name}: {len(geo)} scenes, {len(geo) - len(todo)} already segmented", flush=True)
            done = 0
            for run, frames in pool.imap_unordered(render_scene, [(g, a.readings, 92) for g in todo]):
                os.makedirs(cache_dir(a.cache, set_dir, run), exist_ok=True)
                for start in range(0, len(frames), a.batch):
                    chunk = frames[start:start + a.batch]
                    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for _, _, b in chunk]
                    for (robot, q, _), img, (mask, rects, labels) in zip(chunk, imgs, seg.many(imgs)):
                        cv2.imwrite(frame_path(a.cache, set_dir, run, robot, q), labels)
                        if q == 0 and len(sheet) < 16 and int(labels.max()) >= 2 and done % 7 == 0:
                            sheet.append(cv2.resize(overlay(img, labels, rects), (640, 360)))
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"  {name}: {done}/{len(todo)} scenes", flush=True)
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        fig = os.path.join(os.path.dirname(os.path.normpath(a.cache)), "fig_yolo_sim.jpg")
        cv2.imwrite(fig, np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"overlays: {fig}")


if __name__ == "__main__":
    main()
