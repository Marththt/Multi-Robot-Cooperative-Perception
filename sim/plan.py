"""Scenes and datasets of the research plan "MHT with a learned normality gate" (Sections 4-6, module M0).

Everything is in the world frame of a 6 m x 6 m arena, origin at its centre. Two scene families:

  circle   N robots evenly spaced on a circle of radius R centred at the origin, facing the centre
           (heading noise uniform in +-5 deg); the boxes inside a disc of radius r_b whose centre is
           shifted from the origin by up to 0.4 m in a random direction
  random   the disc centre uniform within 0.5 m of the origin; robots at random angles around it, at least
           20 deg apart, 1.0-2.5 m from its centre, facing it (heading noise uniform in +-10 deg)

Boxes: triangular prisms (side 30 cm, one marker per face), centres uniform in the disc and at least s_min
apart, yaw uniform in [0, 120) deg; a box whose prism would overlap another is drawn again. Robots keep
ROBOT_CLEARANCE_M from every box centre and from each other.

Error knobs (test sets only): k_f multiplies the false-detection rate of the sensor file, k_2 the two-face
rate, and h robots are hidden from each box by cylindrical obstacles (radius 10-15 cm, no marker) on the
line of sight, at 40-60% of the distance, chosen among the robots that have the box in range and field of
view. The sensor is the nominal parameter file (config/sensor_model.json), shared with MHT.

Sets, each in its own folder under --out with its own seed (seeds.csv, also copied to results/plan/):
  train      50,000 scenes   half circle, half random (alternating); N 3-16, B 2-16, s_min 20-60 cm,
                             R 1.2-2.0 m, r_b 0.5-1.0 m; nominal sensor
  gate       10,000 scenes   same distribution, own seed
  val         2,000 scenes   same distribution, own seed
  <ID>_<level>_<family>      the test scenarios of Section 5, 1,000 scenes each (52,000 in all):
             G0 nominal; G1 s_min 20/30/45/60 cm; G2 N 4/8/16; E1 k_f 1/2/4/8; E2 k_2 1/2/5; E3 h 0/1/2;
             S1 B 4/8/12/16 at s_min 20 cm; V1 B 20/24 in r_b 1.0 m; V2 N 20/24
  Nominal scene: N 6, B 5, s_min 30 cm, R 1.5 m, r_b 0.7 m, k_f 1, k_2 1, h 0.

Choices the plan leaves open (also in the module constants):
  - B is uniform; s_min and r_b are drawn again when the boxes would fill more than MAX_PACKING of the disc
    (e.g. 16 boxes 60 cm apart do not fit in a 0.5 m disc) or cannot be placed at random without touching,
    so in training large B goes with small s_min or large r_b;
  - in the random family N robots cannot be 20 deg apart when N > 17: the separation is then
    0.5 * 360 / N deg (9 deg for 20 robots, 7.5 deg for 24);
  - the disc offset of the circle family is uniform in [0, 0.4] m; it is drawn again if a box would come
    closer than ROBOT_CLEARANCE_M to a robot.

Every set folder holds the runs (P<k>.jsonl, as the real runs), ground_truth.json, runs.csv (one row per
scene: family, N, B, s_min, r_b, R, disc centre, k_f, k_2, h, obstacles, seed) and geometry.jsonl (boxes,
yaws, obstacles and camera poses of every scene, for sim/plan_check.py).

    python -m sim.plan --model config/sensor_model.json                  # every set
    python -m sim.plan --model config/sensor_model.json --sets train,gate,val
    python -m sim.plan --model config/sensor_model.json --sets tests     # the 52 test sets
    python -m sim.plan --list                                            # names and seeds only
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
from dataclasses import asdict, dataclass, field, replace
from typing import List

import numpy as np

from sim.generate import simulate_scene
from sim.geometry import box_polygon, polygons_overlap
from sim.model import SensorModel
from sim.scene import Capture, Scene
from sim.writer import write_run

CIRCLE_DISC_OFFSET_M = 0.4          # circle family: largest shift of the box disc from the origin
RANDOM_DISC_OFFSET_M = 0.5          # random family: disc centre within this distance of the origin
RANDOM_DISTANCE_M = (1.0, 2.5)      # random family: robot distance from the disc centre
RANDOM_MIN_SEP_DEG = 20.0           # random family: smallest angle between two robots
HEADING_NOISE_DEG = dict(circle=5.0, random=10.0)
ROBOT_CLEARANCE_M = 0.35            # robot from a box centre and from another robot
MAX_PACKING = 0.45                  # B (s_min/2)^2 / (r_b + s_min/2)^2 above this: the boxes do not fit
OBSTACLE_RADIUS_M = (0.10, 0.15)
OBSTACLE_AT = (0.4, 0.6)            # obstacle centre along the line of sight, share of the distance
OBSTACLE_SIDES = 16                 # a cylinder is drawn as a regular polygon
PART = 2500                         # scenes per parallel part


@dataclass
class PlanSpec:
    """Ranges of one set ([lo, hi], integers inclusive). families: scene i uses families[i % len]."""
    families: List[str] = field(default_factory=lambda: ["circle", "random"])
    n_robots: List[int] = field(default_factory=lambda: [6, 6])
    n_boxes: List[int] = field(default_factory=lambda: [5, 5])
    spacing_m: List[float] = field(default_factory=lambda: [0.30, 0.30])
    circle_radius_m: List[float] = field(default_factory=lambda: [1.5, 1.5])
    disc_radius_m: List[float] = field(default_factory=lambda: [0.7, 0.7])
    false_factor: float = 1.0            # k_f
    two_face_factor: float = 1.0         # k_2
    hidden_robots: int = 0               # h


NOMINAL = PlanSpec()
TRAINING = PlanSpec(n_robots=[3, 16], n_boxes=[2, 16], spacing_m=[0.20, 0.60], circle_radius_m=[1.2, 2.0],
                    disc_radius_m=[0.5, 1.0])

# test scenarios of Section 5: ID -> [(level name, changes to the nominal scene)]
SCENARIOS = {
    "G0": [("nominal", {})],
    "G1": [(f"s{round(v * 100)}", dict(spacing_m=[v, v])) for v in (0.20, 0.30, 0.45, 0.60)],
    "G2": [(f"n{v}", dict(n_robots=[v, v])) for v in (4, 8, 16)],
    "E1": [(f"kf{v}", dict(false_factor=float(v))) for v in (1, 2, 4, 8)],
    "E2": [(f"k2{v}", dict(two_face_factor=float(v))) for v in (1, 2, 5)],
    "E3": [(f"h{v}", dict(hidden_robots=v)) for v in (0, 1, 2)],
    "S1": [(f"b{v}", dict(n_boxes=[v, v], spacing_m=[0.20, 0.20])) for v in (4, 8, 12, 16)],
    "V1": [(f"b{v}", dict(n_boxes=[v, v], disc_radius_m=[1.0, 1.0])) for v in (20, 24)],
    "V2": [(f"n{v}", dict(n_robots=[v, v])) for v in (20, 24)],
}


def set_table(a):
    """[(name, spec, runs, seed, scenario, level, family)] of every set, in a fixed order (fixed seeds)."""
    out = [("train", TRAINING, a.train_runs, 1, "train", "", "both"),
           ("gate", TRAINING, a.gate_runs, 2, "gate", "", "both"),
           ("val", TRAINING, a.val_runs, 3, "val", "", "both")]
    k = 0
    for sid, levels in SCENARIOS.items():
        for level, changes in levels:
            for family in ("circle", "random"):
                spec = replace(NOMINAL, families=[family], **changes)
                out.append((f"{sid}_{level}_{family}", spec, a.test_runs, 100 + k, sid, level, family))
                k += 1
    return out


# ====================================================================
# One scene
# ====================================================================

def _int(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _float(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def draw_layout_params(spec, n, rng, tries=10000):
    """(s_min, r_b) for n boxes that fit: the boxes fill at most MAX_PACKING of the disc. B is drawn once per
    scene and kept, so its distribution stays uniform; s_min and r_b are drawn again until they fit."""
    for _ in range(tries):
        s, rb = _float(rng, spec.spacing_m), _float(rng, spec.disc_radius_m)
        if n * (s / 2) ** 2 / (rb + s / 2) ** 2 <= MAX_PACKING:
            return s, rb
    raise ValueError(f"{spec.n_boxes} boxes at {spec.spacing_m} m do not fit in a disc of {spec.disc_radius_m} m")


def place_boxes(n, spacing, radius, centre, rng, tries=500, restarts=400):
    """n box centres uniform in the disc, at least `spacing` apart, and yaws in [0, 120) deg such that no two
    prisms overlap. Sequential: a box that cannot be placed after `tries` draws restarts the layout."""
    for _ in range(restarts):
        pts, yaws, polys = [], [], []
        for _ in range(n):
            for _ in range(tries):
                r, a = radius * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
                p = np.array([centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)])
                if any(math.hypot(*(p - q)) < spacing for q in pts):
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
    raise RuntimeError(f"cannot place {n} boxes {spacing} m apart in a disc of {radius} m")


def spread_angles(n, min_sep, rng):
    """n angles uniform on the circle with every gap >= min_sep [rad]: the gaps are min_sep plus a uniformly
    random split (flat Dirichlet) of the free angle, from a random start."""
    free = 2 * math.pi - n * min_sep
    gaps = min_sep + free * rng.dirichlet(np.ones(n))
    return (rng.uniform(0, 2 * math.pi) + np.concatenate([[0.0], np.cumsum(gaps[:-1])])) % (2 * math.pi)


def robot_poses(family, n, disc_centre, spec, rng):
    """[(x, y, yaw, distance from the circle or disc centre, angle_deg)] of the n cameras."""
    noise = math.radians(HEADING_NOISE_DEG[family])
    out = []
    if family == "circle":
        R = _float(rng, spec.circle_radius_m)
        start = rng.uniform(0, 2 * math.pi)
        for k in range(n):
            a = start + 2 * math.pi * k / n
            x, y = R * math.cos(a), R * math.sin(a)
            out.append((x, y, math.atan2(-y, -x) + rng.uniform(-noise, noise), R, math.degrees(a) % 360))
        return out, R
    sep = math.radians(min(RANDOM_MIN_SEP_DEG, 0.5 * 360.0 / n)) if n * RANDOM_MIN_SEP_DEG >= 360 \
        else math.radians(RANDOM_MIN_SEP_DEG)
    for a in spread_angles(n, sep, rng):
        d = _float(rng, RANDOM_DISTANCE_M)
        x, y = disc_centre[0] + d * math.cos(a), disc_centre[1] + d * math.sin(a)
        yaw = math.atan2(disc_centre[1] - y, disc_centre[0] - x) + rng.uniform(-noise, noise)
        out.append((x, y, yaw, d, math.degrees(a) % 360))
    return out, float("nan")


def clear(poses, boxes):
    for k, (x, y, *_) in enumerate(poses):
        if any(math.hypot(x - b[0], y - b[1]) < ROBOT_CLEARANCE_M for b in boxes):
            return False
        if any(math.hypot(x - q[0], y - q[1]) < ROBOT_CLEARANCE_M for q in poses[:k]):
            return False
    return True


def cylinder(centre, radius):
    a = np.linspace(0, 2 * math.pi, OBSTACLE_SIDES, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], axis=1)


def in_view(pose, point, model):
    x, y, yaw = pose[:3]
    d = math.hypot(point[0] - x, point[1] - y)
    off = (math.atan2(point[1] - y, point[0] - x) - yaw + math.pi) % (2 * math.pi) - math.pi
    return model.r_min_m <= d <= model.r_max_m and abs(off) <= math.radians(model.hfov_deg) / 2


def hide_boxes(h, boxes, yaws, poses, model, rng, tries=20):
    """For every box, h of the robots that have it in range and field of view get a cylinder on their line of
    sight to the box centre, not overlapping a box or a robot. Returns the obstacle polygons."""
    if h <= 0:
        return []
    box_polys = [box_polygon(c, t) for c, t in zip(boxes, yaws)]
    obstacles = []
    for b in boxes:
        seeing = [k for k, p in enumerate(poses) if in_view(p, b, model)]
        for k in rng.permutation(seeing)[:h]:
            x, y = poses[k][:2]
            for _ in range(tries):
                t, r = _float(rng, OBSTACLE_AT), _float(rng, OBSTACLE_RADIUS_M)
                c = np.array([x + t * (b[0] - x), y + t * (b[1] - y)])
                poly = cylinder(c, r)
                if any(polygons_overlap(poly, bp) for bp in box_polys):
                    continue
                if any(math.hypot(c[0] - p[0], c[1] - p[1]) < r + 0.15 for p in poses):
                    continue
                obstacles.append(poly)
                break
    return obstacles


def plan_scene(spec, family, rng, layout_id, run_id, model, attempts=200):
    """One scene of `family` from the ranges of `spec`; returns (Scene, metadata)."""
    n_rob, n_box = _int(rng, spec.n_robots), _int(rng, spec.n_boxes)
    for _ in range(attempts):
        spacing, rb = draw_layout_params(spec, n_box, rng)
        if family == "circle":
            r, a = rng.uniform(0, CIRCLE_DISC_OFFSET_M), rng.uniform(0, 2 * math.pi)
        else:
            r, a = RANDOM_DISC_OFFSET_M * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        centre = np.array([r * math.cos(a), r * math.sin(a)])
        try:
            boxes, yaws = place_boxes(n_box, spacing, rb, centre, rng, restarts=40)
        except RuntimeError:                 # too dense for random sequential placement: draw again
            continue
        for _ in range(50):
            poses, R = robot_poses(family, n_rob, centre, spec, rng)
            if clear(poses, boxes):
                break
        else:
            continue
        break
    else:
        raise RuntimeError(f"no clear layout for {run_id}")
    obstacles = hide_boxes(spec.hidden_robots, boxes, yaws, poses, model, rng)
    captures = [Capture(round=0, robot=f"P{k}", pose=(x, y, yaw), radius=d, angle_deg=ang)
                for k, (x, y, yaw, d, ang) in enumerate(poses)]
    scene = Scene(layout_id=layout_id, run_id=run_id, boxes=boxes, thetas=yaws, obstacles=obstacles,
                  captures=captures)
    meta = dict(family=family, n_robots=n_rob, n_boxes=n_box, spacing_m=round(spacing, 4),
                disc_radius_m=round(rb, 4), circle_radius_m=round(R, 4), disc_x=round(float(centre[0]), 4),
                disc_y=round(float(centre[1]), 4), false_factor=spec.false_factor,
                two_face_factor=spec.two_face_factor, hidden_robots=spec.hidden_robots,
                n_obstacles=len(obstacles))
    return scene, meta


def run_model(model, spec):
    m = copy.deepcopy(model)
    m.false_rate *= spec.false_factor
    m.two_face_q *= spec.two_face_factor
    return m


# ====================================================================
# Sets
# ====================================================================

def generate_part(task):
    """Scenes start .. start + runs - 1 of one set into part_dir."""
    name, spec_dict, runs, start, seed, part_dir, model_path = task
    spec = PlanSpec(**spec_dict)
    model = SensorModel.load(model_path)
    m = run_model(model, spec)
    os.makedirs(part_dir, exist_ok=True)
    truth, rows, geometry = {}, [], []
    for i in range(start, start + runs):
        scene_seed = seed * 1_000_003 + i
        rng = np.random.default_rng(scene_seed)
        family = spec.families[i % len(spec.families)]
        scene, meta = plan_scene(spec, family, rng, name, f"{name}_{i:05d}", m)
        truth.update(write_run(part_dir, scene, simulate_scene(scene, m, rng), m, folder=f"{name}_{i // 500:04d}"))
        rows.append(dict(run=scene.run_id, seed=scene_seed, **meta))
        geometry.append(dict(run=scene.run_id, boxes=scene.boxes.round(4).tolist(),
                             yaws=[round(float(t), 5) for t in scene.thetas],
                             obstacles=[o.round(4).tolist() for o in scene.obstacles],
                             poses=[[round(float(v), 5) for v in c.pose] for c in scene.captures]))
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


def merge(set_dir, part_dirs):
    """One ground_truth.json, runs.csv and geometry.jsonl per set; the record folders stay in the parts."""
    truth, rows = {}, []
    with open(os.path.join(set_dir, "geometry.jsonl"), "w") as g:
        for d in sorted(part_dirs):
            with open(os.path.join(d, "ground_truth.json")) as f:
                truth.update(json.load(f))
            with open(os.path.join(d, "runs.csv")) as f:
                rows += list(csv.DictReader(f))
            with open(os.path.join(d, "geometry.jsonl")) as f:
                g.write(f.read())
            for name in ("ground_truth.json", "runs.csv", "geometry.jsonl"):
                os.remove(os.path.join(d, name))
    with open(os.path.join(set_dir, "ground_truth.json"), "w") as f:
        json.dump(truth, f)
    with open(os.path.join(set_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["run"]))
    return len(rows)


def write_seeds(table, a, paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["set", "scenario", "level", "family", "scenes", "seed", "scene_seed_rule"])
            for name, spec, runs, seed, sid, level, family in table:
                w.writerow([name, sid, level, family, runs, a.seed * 1000 + seed,
                            "seed * 1000003 + scene index"])


def select(table, text):
    wanted = [s.strip() for s in text.split(",") if s.strip()]
    out = []
    for row in table:
        name, sid = row[0], row[4]
        if "all" in wanted or name in wanted or sid in wanted or (
                "tests" in wanted and name not in ("train", "gate", "val")):
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="config/sensor_model.json", help="nominal sensor parameter file")
    ap.add_argument("--sets", default="all",
                    help="all | tests | train,gate,val | scenario IDs (G1,E3,...) | set names (E1_kf4_circle,...)")
    ap.add_argument("--out", default="data/plan/")
    ap.add_argument("--seed-table", default="results/plan/seeds.csv")
    ap.add_argument("--train-runs", type=int, default=50000)
    ap.add_argument("--gate-runs", type=int, default=10000)
    ap.add_argument("--val-runs", type=int, default=2000)
    ap.add_argument("--test-runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--list", action="store_true", help="print the sets and their seeds, generate nothing")
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)

    table = set_table(a)
    if a.list:
        for name, spec, runs, seed, sid, level, family in table:
            print(f"{name:22s} {runs:6d} scenes  seed {a.seed * 1000 + seed}")
        print(f"{len(table)} sets, {sum(r[2] for r in table)} scenes")
        return
    chosen = select(table, a.sets)
    if not chosen:
        raise SystemExit(f"no set matches {a.sets}; see --list")
    tasks, parts = [], {}
    for name, spec, runs, seed, sid, level, family in chosen:
        set_dir = os.path.join(a.out, name)
        if os.path.isdir(set_dir):
            shutil.rmtree(set_dir)
        os.makedirs(set_dir)
        shutil.copy(a.model, os.path.join(set_dir, "sensor_model_used.json"))
        with open(os.path.join(set_dir, "spec.json"), "w") as f:
            json.dump(dict(asdict(spec), scenario=sid, level=level, seed=a.seed * 1000 + seed), f, indent=2)
        for k, start in enumerate(range(0, runs, PART)):
            part_dir = os.path.join(set_dir, f"part{k:02d}")
            tasks.append((name, asdict(spec), min(PART, runs - start), start, a.seed * 1000 + seed, part_dir,
                          a.model))
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
