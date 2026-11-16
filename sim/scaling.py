"""Scalability sets: robots AND boxes placed at random over a square area large enough to hold them, from a
few robots up to 50, with as many boxes.

Scene ("area" family):
  side      L = max(MIN_SIDE_M, sqrt((N + B) / density)): the area grows with the number of objects, so the
            density of robots and boxes stays that of the lab (about 1.25 objects per m^2 in the lab disc)
  boxes     centres uniform in the square (MARGIN_M from its sides), at least s_min apart, yaw uniform in
            [0, 120) deg, no two prisms overlapping
  robots    camera positions uniform in the square, clear of the box centres and of each other
  heading   every robot looks where it sees the most boxes (the direction, among HEADING_STEPS, with the most
            boxes in its field of view and range; a random one among the ties), plus a uniform heading noise;
            a robot with no box in range looks in a random direction
  sensor    the nominal sensor file (config/sensor_model.json), as sim/plan.py; the bodies of the other robots
            hide the boxes behind them (occluders of the sensor, not obstacles of the scene)
  covered   boxes in the field of view and range of at least one robot. A box that no robot can see cannot be
            found by any method: the evaluation (yolo_gnn/evaluate_sim.py) scores the covered boxes. The
            ground truth of the set keeps every box, so the labels of the detections stay the box indices.

Sets, each in its own folder under --out with its own seed:
  train      20,000 scenes  N 3-50, B 2-50, s_min 20-60 cm, density 0.8-2.0 per m^2 (to train the marker network
                            on large teams too)
  val         2,000 scenes  same distribution, own seed (model selection of the marker network)
  tune          300 scenes  same distribution, own seed (thresholds of the YOLO rays in simulation)
  SC_n<N>       200 scenes  N = B = 4, 8, 16, 24, 32, 50 (robots and boxes grow together)
  SR_n<N>       200 scenes  B = 16, N = 4, 8, 32, 50 (more robots, same boxes; N = 16 is SC_n16)
  SB_b<B>       200 scenes  N = 16, B = 4, 8, 32, 50 (more boxes, same robots; B = 16 is SC_n16)
  Test scenes: s_min 30 cm, density 1.25 per m^2, heading noise 10 deg.

Every set folder holds the runs (P<k>.jsonl, as the real runs), ground_truth.json (every box), runs.csv (one row
per scene: N, B, covered boxes, s_min, density, side, seed) and geometry.jsonl (boxes, yaws, camera poses, covered
box indices, side of every scene: the renderer of yolo_gnn/segment_sim.py redraws the scenes from it).

    python -m sim.scaling --model config/sensor_model.json --out data/scale/            # every set
    python -m sim.scaling --sets train,val --out data/scale/
    python -m sim.scaling --sets tune,tests --out data/scale/
    python -m sim.scaling --list
"""

import argparse
import copy
import csv
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from typing import List

import numpy as np

from sim.geometry import box_polygon, polygons_overlap, rectangle, wrap
from sim.model import SensorModel
from sim.plan import merge, write_seeds
from sim.scene import Capture, Scene
from sim.sensor import capture, robot_bias
from sim.writer import write_run

MIN_SIDE_M = 3.0                    # smallest area (a few robots and boxes)
MARGIN_M = 0.20                     # box centres and cameras this far inside the square
ROBOT_CLEARANCE_M = 0.35            # camera and body centre from a box centre
ROBOT_SEPARATION_M = 0.45           # body centre from another body centre
ROBOT_SIZE_M = (0.32, 0.24)         # body footprint, length x width (imgsim.render.Appearance.robot_size_m)
ROBOT_BODY_BEHIND_M = 0.15          # body centre behind the camera (imgsim.render.Appearance.robot_body_behind_m)
HEADING_STEPS = 72                  # candidate directions of the heading rule (5 deg)
FOV_MARGIN_DEG = 4.0                # a box counts for the heading rule when this far inside the field of view
PART = 500                          # scenes per parallel part


@dataclass
class ScaleSpec:
    """Ranges of one set ([lo, hi], integers inclusive)."""
    n_robots: List[int] = field(default_factory=lambda: [16, 16])
    n_boxes: List[int] = field(default_factory=lambda: [16, 16])
    spacing_m: List[float] = field(default_factory=lambda: [0.30, 0.30])
    density: List[float] = field(default_factory=lambda: [1.25, 1.25])     # robots + boxes per m^2
    heading_noise_deg: float = 10.0
    false_factor: float = 1.0
    two_face_factor: float = 1.0


TRAINING = ScaleSpec(n_robots=[3, 50], n_boxes=[2, 50], spacing_m=[0.20, 0.60], density=[0.8, 2.0])
SWEEPS = {
    "SC": [(f"n{v}", dict(n_robots=[v, v], n_boxes=[v, v])) for v in (4, 8, 16, 24, 32, 50)],
    "SR": [(f"n{v}", dict(n_robots=[v, v], n_boxes=[16, 16])) for v in (4, 8, 32, 50)],
    "SB": [(f"b{v}", dict(n_robots=[16, 16], n_boxes=[v, v])) for v in (4, 8, 32, 50)],
}


def set_table(a):
    """[(name, spec, runs, seed, sweep, level, family)] of every set, in a fixed order (fixed seeds)."""
    out = [("train", TRAINING, a.train_runs, 501, "train", "", "area"),
           ("val", TRAINING, a.val_runs, 502, "val", "", "area"),
           ("tune", TRAINING, a.tune_runs, 503, "tune", "", "area")]
    k = 0
    for sid, levels in SWEEPS.items():
        for level, changes in levels:
            out.append((f"{sid}_{level}", replace(ScaleSpec(), **changes), a.test_runs, 600 + k, sid, level, "area"))
            k += 1
    return out


def select(table, text):
    wanted = [s.strip() for s in text.split(",") if s.strip()]
    return [row for row in table
            if "all" in wanted or row[0] in wanted or row[4] in wanted
            or ("tests" in wanted and row[0] not in ("train", "val", "tune"))]


# ====================================================================
# One scene
# ====================================================================

def _int(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _float(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def area_side(n_robots, n_boxes, density):
    return max(MIN_SIDE_M, math.sqrt((n_robots + n_boxes) / density))


def body_centre(x, y, yaw):
    return np.array([x - ROBOT_BODY_BEHIND_M * math.cos(yaw), y - ROBOT_BODY_BEHIND_M * math.sin(yaw)])


def robot_footprint(pose):
    x, y, yaw = pose
    return rectangle(body_centre(x, y, yaw), ROBOT_SIZE_M[0], ROBOT_SIZE_M[1], yaw)


def place_boxes(n, spacing, side, rng, tries=400, restarts=60):
    """n box centres uniform in the square, at least `spacing` apart, yaws in [0, 120) deg, no overlap."""
    h = side / 2 - MARGIN_M
    for _ in range(restarts):
        pts, yaws, polys = [], [], []
        for _ in range(n):
            for _ in range(tries):
                p = rng.uniform(-h, h, 2)
                if pts and np.min(np.hypot(*(np.asarray(pts) - p).T)) < spacing:
                    continue
                yaw = rng.uniform(0, 2 * math.pi / 3)
                poly = box_polygon(p, yaw)
                if any(math.hypot(*(p - q)) < 0.35 and polygons_overlap(poly, o) for q, o in zip(pts, polys)):
                    continue
                pts.append(p), yaws.append(yaw), polys.append(poly)
                break
            else:
                break
        if len(pts) == n:
            return np.array(pts), np.array(yaws)
    raise RuntimeError(f"cannot place {n} boxes {spacing} m apart in a {side:.1f} m square")


def place_robots(n, boxes, side, rng, tries=2000):
    """n camera positions uniform in the square, clear of the boxes and of each other whatever the heading (the
    body centre can lie anywhere ROBOT_BODY_BEHIND_M from the camera)."""
    h = side / 2 - MARGIN_M - ROBOT_BODY_BEHIND_M - ROBOT_SIZE_M[0] / 2     # the whole body inside the square
    pts = []
    for _ in range(n):
        for _ in range(tries):
            p = rng.uniform(-h, h, 2)
            if len(boxes) and np.min(np.hypot(*(boxes - p).T)) < ROBOT_CLEARANCE_M + ROBOT_BODY_BEHIND_M:
                continue
            if pts and np.min(np.hypot(*(np.asarray(pts) - p).T)) < ROBOT_SEPARATION_M + 2 * ROBOT_BODY_BEHIND_M:
                continue
            pts.append(p)
            break
        else:
            return None
    return np.array(pts)


def best_heading(cam, boxes, model, noise, rng):
    """The direction with the most boxes in the field of view and range (a random one among the ties), plus
    the heading noise; a random direction when no box is in range."""
    v = boxes - cam
    d = np.hypot(v[:, 0], v[:, 1])
    ok = (d >= model.r_min_m) & (d <= model.r_max_m)
    if not ok.any():
        return float(rng.uniform(-math.pi, math.pi))
    ang = np.arctan2(v[ok, 1], v[ok, 0])
    cand = np.linspace(-math.pi, math.pi, HEADING_STEPS, endpoint=False)
    half = math.radians(model.hfov_deg / 2 - FOV_MARGIN_DEG)
    counts = (np.abs(wrap(ang[None, :] - cand[:, None])) <= half).sum(1)
    best = cand[rng.choice(np.flatnonzero(counts == counts.max()))]
    return float(wrap(best + rng.uniform(-noise, noise)))


def covered_boxes(boxes, poses, model):
    """Indices of the boxes in the field of view and range of at least one camera."""
    half = math.radians(model.hfov_deg) / 2
    out = []
    for j, b in enumerate(boxes):
        for x, y, yaw in poses:
            d = math.hypot(b[0] - x, b[1] - y)
            if model.r_min_m <= d <= model.r_max_m and abs(float(wrap(math.atan2(b[1] - y, b[0] - x) - yaw))) <= half:
                out.append(j)
                break
    return out


def area_scene(spec, model, rng, layout_id, run_id, attempts=100):
    """One scene of the area family; returns (Scene, metadata, covered box indices, side)."""
    n_rob, n_box = _int(rng, spec.n_robots), _int(rng, spec.n_boxes)
    for _ in range(attempts):
        spacing, density = _float(rng, spec.spacing_m), _float(rng, spec.density)
        side = area_side(n_rob, n_box, density)
        try:
            boxes, yaws = place_boxes(n_box, spacing, side, rng)
        except RuntimeError:
            continue
        cams = place_robots(n_rob, boxes, side, rng)
        if cams is not None:
            break
    else:
        raise RuntimeError(f"no layout for {run_id}")
    noise = math.radians(spec.heading_noise_deg)
    poses = [(float(c[0]), float(c[1]), best_heading(c, boxes, model, noise, rng)) for c in cams]
    covered = covered_boxes(boxes, poses, model)
    captures = [Capture(round=0, robot=f"P{k}", pose=p, radius=float(math.hypot(p[0], p[1])),
                        angle_deg=math.degrees(math.atan2(p[1], p[0])) % 360) for k, p in enumerate(poses)]
    scene = Scene(layout_id=layout_id, run_id=run_id, boxes=boxes, thetas=yaws, obstacles=[], captures=captures)
    meta = dict(family="area", n_robots=n_rob, n_boxes=n_box, n_covered=len(covered), spacing_m=round(spacing, 4),
                density=round(density, 4), side_m=round(side, 3), heading_noise_deg=spec.heading_noise_deg,
                false_factor=spec.false_factor, two_face_factor=spec.two_face_factor)
    return scene, meta, covered, side


def simulate(scene, model, rng):
    """Detections of every capture (sim.sensor.capture), the bodies of the other robots in range hiding the
    boxes behind them."""
    feet = [robot_footprint(c.pose) for c in scene.captures]
    biases, out = {}, {}
    for ci, cap in enumerate(scene.captures):
        if cap.robot not in biases:
            biases[cap.robot] = robot_bias(model, rng)
        x, y, _ = cap.pose
        near = [f for k, f in enumerate(feet)
                if k != ci and math.hypot(*(f.mean(0) - [x, y])) <= model.r_max_m + 0.5]
        out[ci] = capture(replace(scene, obstacles=list(scene.obstacles) + near), cap, model, rng, biases[cap.robot])
    return out


def run_model(model, spec):
    m = copy.deepcopy(model)
    m.false_rate *= spec.false_factor
    m.two_face_q *= spec.two_face_factor
    return m


def scene_from_geometry(g, layout_id="scene"):
    """The Scene of one geometry.jsonl row (for the renderer)."""
    caps = [Capture(round=0, robot=f"P{k}", pose=tuple(p)) for k, p in enumerate(g["poses"])]
    return Scene(layout_id=layout_id, run_id=g["run"], boxes=np.array(g["boxes"], float),
                 thetas=np.array(g["yaws"], float), obstacles=[np.array(o, float) for o in g.get("obstacles", [])],
                 captures=caps)


def read_geometry(set_dir):
    """{run: geometry row} of one set."""
    with open(os.path.join(set_dir, "geometry.jsonl")) as f:
        return {g["run"]: g for g in map(json.loads, f) if g}


# ====================================================================
# Sets
# ====================================================================

def generate_part(task):
    name, spec_dict, runs, start, seed, part_dir, model_path = task
    spec = ScaleSpec(**spec_dict)
    m = run_model(SensorModel.load(model_path), spec)
    os.makedirs(part_dir, exist_ok=True)
    truth, rows, geometry = {}, [], []
    for i in range(start, start + runs):
        scene_seed = seed * 1_000_003 + i
        rng = np.random.default_rng(scene_seed)
        scene, meta, covered, side = area_scene(spec, m, rng, name, f"{name}_{i:05d}")
        truth.update(write_run(part_dir, scene, simulate(scene, m, rng), m, folder=f"{name}_{i // 500:04d}"))
        rows.append(dict(run=scene.run_id, seed=scene_seed, **meta))
        geometry.append(dict(run=scene.run_id, seed=scene_seed, side_m=round(side, 3), covered=covered,
                             boxes=scene.boxes.round(4).tolist(), yaws=[round(float(t), 5) for t in scene.thetas],
                             obstacles=[], poses=[[round(float(v), 5) for v in c.pose] for c in scene.captures]))
    with open(os.path.join(part_dir, "ground_truth.json"), "w") as f:
        json.dump(truth, f)
    with open(os.path.join(part_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(part_dir, "geometry.jsonl"), "w") as f:
        for g in geometry:
            f.write(json.dumps(g) + "\n")
    return name, part_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="config/sensor_model.json", help="nominal sensor parameter file")
    ap.add_argument("--sets", default="all", help="all | tests | train,val,tune | SC,SR,SB | set names (SC_n50,...)")
    ap.add_argument("--out", default="data/scale/")
    ap.add_argument("--seed-table", default="results/scale/seeds.csv")
    ap.add_argument("--train-runs", type=int, default=20000)
    ap.add_argument("--val-runs", type=int, default=2000)
    ap.add_argument("--tune-runs", type=int, default=300)
    ap.add_argument("--test-runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--list", action="store_true", help="print the sets and their seeds, generate nothing")
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)

    table = set_table(a)
    if a.list:
        for name, spec, runs, seed, *_ in table:
            print(f"{name:10s} {runs:6d} scenes  seed {a.seed * 1000 + seed}  N {spec.n_robots}  B {spec.n_boxes}")
        return
    chosen = select(table, a.sets)
    if not chosen:
        raise SystemExit(f"no set matches {a.sets}; see --list")
    tasks, parts = [], {}
    for name, spec, runs, seed, sid, level, family in chosen:
        set_dir = os.path.join(a.out, name)
        if os.path.isdir(set_dir):                     # never deleted: set aside in _to_delete/
            dest = os.path.join("_to_delete", f"scale_{name}_{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs("_to_delete", exist_ok=True)
            shutil.move(set_dir, dest)
            print(f"previous {set_dir} moved to {dest}")
        os.makedirs(set_dir)
        shutil.copy(a.model, os.path.join(set_dir, "sensor_model_used.json"))
        with open(os.path.join(set_dir, "spec.json"), "w") as f:
            json.dump(dict(asdict(spec), sweep=sid, level=level, seed=a.seed * 1000 + seed), f, indent=2)
        for k, start in enumerate(range(0, runs, PART)):
            part_dir = os.path.join(set_dir, f"part{k:02d}")
            tasks.append((name, asdict(spec), min(PART, runs - start), start, a.seed * 1000 + seed, part_dir, a.model))
            parts.setdefault(name, []).append(part_dir)
    print(f"{len(chosen)} set(s), {sum(r[2] for r in chosen)} scenes: {len(tasks)} parts on "
          f"{min(a.jobs, len(tasks))} process(es)", flush=True)
    with multiprocessing.Pool(min(a.jobs, len(tasks))) as pool:
        for name, done in pool.imap_unordered(generate_part, tasks):
            print(f"  {done}", flush=True)
    for name in parts:
        n = merge(os.path.join(a.out, name), parts[name])
        print(f"  {name}: {n} scenes", flush=True)
    write_seeds(table, a, [os.path.join(a.out, "seeds.csv"), a.seed_table])
    print(f"seeds: {a.seed_table}")


if __name__ == "__main__":
    main()
