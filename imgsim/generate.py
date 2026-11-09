"""Image-level sets: the scenes of sim/plan.py rendered (imgsim/render.py) and passed through the real ArUco
detector (vf/perception/aruco_detector.py, marker 77 mm, the robot's calibration), written in the format of
the real runs.

Every scene is drawn by sim.plan.plan_scene with the same seed as in the measurement-level set of the same
name, so scene i of data/imgsim/<set>/ has the same boxes, obstacles and robot poses as scene i of
data/plan/<set>/. What changes is where the reports come from: here a marker is reported when the detector
finds it in the rendered image, and its distance, bearing and viewing angle come from solvePnP, exactly as on
the robots; there they are drawn from the fitted sensor model. As on the robots, every stop takes 5 frames
and keeps one detection per group of detections closer than 0.2 m (acquisition/static_trial.capture_trial);
so the two faces of one box, about 15 cm apart, mostly become one report, as in the real runs. The reports are RAW values, as in the real
runs: imgsim/compare.py fits their range and bearing calibration on the set (as pipeline/calibrate.py does for
the real runs) and writes it as <set>/sensor_calib.json, to be passed as --sensor-calib to every reader.

Each detection is labelled with the marker it comes from (sim_box, sim_face) when its centre lies close to
the projected centre of a drawn marker, otherwise -1 (a false detection). The renderer has no source of false
detections beyond what the detector produces on its own.

Output per set (--out/<set>/): part*/ with P<k>.jsonl, ground_truth.json, runs.csv, geometry.jsonl (as
sim/plan.py), captures.jsonl (per capture: pitch, roll, markers drawn and detected), appearance.json, and
images/<run>_P<k>.jpg for the first --save-images scenes (--save-images -1: every scene, e.g. for training an
image model).

    python -m imgsim.generate --sets G0_nominal_circle --runs 200 --save-images 20
    python -m imgsim.generate --sets G0,E1,E3,S1 --runs 1000 --jobs 30
"""

import argparse
import copy
import csv
import json
import logging
import math
import multiprocessing
import os
import shutil
import sys
from dataclasses import asdict

import cv2
import numpy as np

from imgsim.render import Appearance, Renderer
from sim.model import SensorModel
from sim.plan import PlanSpec, plan_scene, run_model, select, set_table
from sim.writer import write_run

PART = 250
CALIB = dict(P0="config/checkerboard_calib_P0.json", P1="config/checkerboard_calib_P1.json")


def detect(detector, img):
    """[(distance, bearing, viewing angle, u, v, size)] of the markers the real detector finds."""
    out = []
    for _cls, _conf, bbox, (dist, bearing, view) in detector.detect_with_pose(img):
        x, y, w, h = bbox
        out.append((float(dist), float(bearing), float(view), x + w / 2, y + h / 2, max(w, h)))
    return out


def one_stop(renderer, detector, scene, k, rng, style, readings, threshold):
    """What a robot standing at capture k reports, as acquisition/static_trial.capture_trial does on the robot:
    `readings` frames (the same view, new sensor noise, blur and compression), the detections of all of
    them grouped by position (vf.perception.object_position.cluster_by_position, --instance-threshold-m of the
    real runs), and one detection per group, the one whose distance is closest to the group median.
    Returns (reports, first frame, markers drawn, camera settings)."""
    from vf.perception.object_position import cluster_by_position
    clean, markers, info = renderer.render_clean(scene, k, rng, style)
    x, y, yaw = scene.captures[k].pose
    found, first = [], None
    for r in range(readings):
        img = renderer.camera_effects(clean, rng)
        first = img if first is None else first
        found += label(detect(detector, img), markers)
    if not found:
        return [], first, markers, info
    pos = [(x + d["distance"] * math.cos(yaw - d["bearing"]), y + d["distance"] * math.sin(yaw - d["bearing"]))
           for d in found]
    groups = {}
    for d, g in zip(found, cluster_by_position(pos, threshold_m=threshold)):
        groups.setdefault(g, []).append(d)
    reports = []
    for g in groups.values():
        med = float(np.median([d["distance"] for d in g]))
        reports.append(min(g, key=lambda d: abs(d["distance"] - med)))
    return reports, first, markers, info


def label(dets, markers):
    """Box and face of every detection: the nearest drawn marker centre within its size (else -1)."""
    out = []
    for dist, bearing, view, u, v, size in dets:
        best, bd = None, max(12.0, 0.6 * size)
        for m in markers:
            d = math.hypot(m["u"] - u, m["v"] - v)
            if d < bd:
                best, bd = m, d
        out.append(dict(distance=dist, bearing=bearing, view=view,
                        box=best["box"] if best else -1, face=best["face"] if best else -1))
    return out


def generate_part(task):
    (name, spec_dict, runs, start, seed, part_dir, model_path, app_dict, calib, save_images, set_dir, readings,
     threshold) = task
    logging.disable(logging.WARNING)
    from vf.perception.aruco_detector import ArucoDetector, load_calibration
    spec = PlanSpec(**spec_dict)
    model = SensorModel.load(model_path)
    m = run_model(model, spec)
    app = Appearance(**app_dict)
    renderer = Renderer(calib, app, seed=seed)
    K, D = load_calibration(calib)
    detector = ArucoDetector(app.marker_size_m, K, D)
    writer_model = copy.deepcopy(model)
    writer_model.camera_offset_m = 0.0
    os.makedirs(part_dir, exist_ok=True)
    img_dir = os.path.join(set_dir, "images")
    truth, rows, geometry, caps = {}, [], [], []
    for i in range(start, start + runs):
        scene_seed = seed * 1_000_003 + i                   # the seed of sim/plan.py for this scene
        rng = np.random.default_rng(scene_seed)
        family = spec.families[i % len(spec.families)]
        scene, meta = plan_scene(spec, family, rng, name, f"{name}_{i:05d}", m)
        rrng = np.random.default_rng(scene_seed + 7_919)    # rendering draws, independent of the layout
        style = renderer.scene_style(rrng)
        detections = {}
        for k in range(len(scene.captures)):
            dets, img, markers, info = one_stop(renderer, detector, scene, k, rrng, style, readings, threshold)
            detections[k] = dets
            caps.append(dict(run=scene.run_id, robot=scene.captures[k].robot, **info,
                             markers_drawn=len(markers), detections=len(dets),
                             false=sum(d["box"] < 0 for d in dets)))
            if save_images < 0 or i < save_images:
                os.makedirs(img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(img_dir, f"{scene.run_id}_{scene.captures[k].robot}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth.update(write_run(part_dir, scene, detections, writer_model, folder=f"{name}_{i // 500:04d}"))
        rows.append(dict(run=scene.run_id, seed=scene_seed, **meta))
        geometry.append(dict(run=scene.run_id, boxes=scene.boxes.round(4).tolist(),
                             yaws=[round(float(t), 5) for t in scene.thetas],
                             obstacles=[o.round(4).tolist() for o in scene.obstacles],
                             poses=[[round(float(v), 5) for v in c.pose] for c in scene.captures]))
    for fname, data in (("ground_truth.json", truth),):
        with open(os.path.join(part_dir, fname), "w") as f:
            json.dump(data, f)
    for fname, items in (("geometry.jsonl", geometry), ("captures.jsonl", caps)):
        with open(os.path.join(part_dir, fname), "w") as f:
            for g in items:
                f.write(json.dumps(g) + "\n")
    with open(os.path.join(part_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return name, part_dir


def merge(set_dir, part_dirs):
    truth, rows = {}, []
    with open(os.path.join(set_dir, "geometry.jsonl"), "w") as g, \
            open(os.path.join(set_dir, "captures.jsonl"), "w") as c:
        for d in sorted(part_dirs):
            with open(os.path.join(d, "ground_truth.json")) as f:
                truth.update(json.load(f))
            with open(os.path.join(d, "runs.csv")) as f:
                rows += list(csv.DictReader(f))
            for name, out in (("geometry.jsonl", g), ("captures.jsonl", c)):
                with open(os.path.join(d, name)) as f:
                    out.write(f.read())
            for name in ("ground_truth.json", "runs.csv", "geometry.jsonl", "captures.jsonl"):
                os.remove(os.path.join(d, name))
    with open(os.path.join(set_dir, "ground_truth.json"), "w") as f:
        json.dump(truth, f)
    with open(os.path.join(set_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["run"]))
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True, help="as sim.plan --sets: scenario IDs, set names, tests, train, ...")
    ap.add_argument("--runs", type=int, default=None, help="scenes per set (default: as in sim/plan.py)")
    ap.add_argument("--model", default="config/sensor_model.json", help="only for the scene layout (field of view)")
    ap.add_argument("--camera", choices=sorted(CALIB), default="P0", help="calibration of the rendered camera")
    ap.add_argument("--appearance", default=None, help="JSON with Appearance fields to change (imgsim/render.py)")
    ap.add_argument("--save-images", type=int, default=20, help="save the images of the first N scenes (-1: all)")
    ap.add_argument("--readings", type=int, default=5, help="frames per stop (survey.py --n-readings)")
    ap.add_argument("--instance-threshold", type=float, default=0.2,
                    help="detections of one stop closer than this are one report (--instance-threshold-m of the "
                         "real runs) [m]")
    ap.add_argument("--out", default="data/imgsim/")
    ap.add_argument("--seed", type=int, default=1, help="as sim.plan --seed (same seeds = same scenes)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)
    big = 10 ** 9
    table = set_table(argparse.Namespace(train_runs=big, gate_runs=big, val_runs=big, test_runs=big, seed=a.seed))
    plan_runs = dict(train=50000, gate=10000, val=2000)
    chosen = select(table, a.sets)
    if not chosen:
        raise SystemExit(f"no set matches {a.sets} (see python -m sim.plan --list)")
    app = Appearance.load(a.appearance)
    tasks, parts = [], {}
    for name, spec, _, seed, sid, level, family in chosen:
        runs = a.runs or plan_runs.get(name, 1000)
        set_dir = os.path.join(a.out, name)
        if os.path.isdir(set_dir):
            shutil.rmtree(set_dir)
        os.makedirs(set_dir)
        app.save(os.path.join(set_dir, "appearance.json"))
        with open(os.path.join(set_dir, "spec.json"), "w") as f:
            json.dump(dict(asdict(spec), scenario=sid, level=level, seed=a.seed * 1000 + seed, camera=a.camera), f,
                      indent=2)
        for k, start in enumerate(range(0, runs, PART)):
            part_dir = os.path.join(set_dir, f"part{k:03d}")
            tasks.append((name, asdict(spec), min(PART, runs - start), start, a.seed * 1000 + seed, part_dir, a.model,
                          asdict(app), CALIB[a.camera], a.save_images, set_dir, a.readings, a.instance_threshold))
            parts.setdefault(name, []).append(part_dir)
    print(f"{len(chosen)} set(s), {sum(t[2] for t in tasks)} scenes: {len(tasks)} parts on "
          f"{min(a.jobs, len(tasks))} process(es)", flush=True)
    with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
        for name, done in pool.imap_unordered(generate_part, tasks):
            print(f"  {done}", flush=True)
    for name in parts:
        n = merge(os.path.join(a.out, name), parts[name])
        print(f"  {name}: {n} scenes", flush=True)
    print("next: python -m imgsim.compare --set-dir <set> (fits the range / bearing calibration of the set)")


if __name__ == "__main__":
    main()
