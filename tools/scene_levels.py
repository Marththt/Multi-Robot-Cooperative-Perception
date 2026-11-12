"""Difficulty level of every real scene, from its geometry only (OptiTrack positions of boxes and cameras),
independently of any method, and the correct scenes of every method per level.

A box is in CLEAR VIEW of a robot if it is within 40 deg of the optical axis and 3 m of the camera, and the line
of sight to its centre does not pass within 0.8 circumradius of another box closer to the camera.
  easy    every pair of boxes at least 25 cm apart, every box in clear view of at least two robots,
          no obstacles or white objects
  medium  as easy, but at least one box in clear view of a single robot
  hard    two boxes less than 25 cm apart, a box in clear view of no robot, obstacles (block C) or white
          objects (block D)

    python -m tools.scene_levels                       # results/scene_levels.csv and results/levels_summary.csv
"""

import argparse
import collections
import csv
import glob
import json
import math
import os

import numpy as np

BOX_SIDE_M = 0.235
APOTHEM_M = BOX_SIDE_M / (2 * math.sqrt(3))
CIRCUMRADIUS_M = BOX_SIDE_M / math.sqrt(3)
HALF_FOV = math.radians(40)
MAX_RANGE_M = 3.0
TOUCH_M = 0.25
LENS_FIX_M = 0.03                   # recorded camera pose (12 cm ahead of the body) -> lens (15 cm)


def clear_views(boxes, cams):
    """Per box: the number of cameras (x, y, yaw) that have it in clear view."""
    B = np.asarray(boxes, float)
    views = []
    for k in range(len(B)):
        v = 0
        for x, y, yaw in cams:
            d = math.hypot(B[k][0] - x, B[k][1] - y)
            if d < 1e-6 or d > MAX_RANGE_M:
                continue
            u = (B[k] - [x, y]) / d
            if abs((math.atan2(u[1], u[0]) - yaw + math.pi) % (2 * math.pi) - math.pi) > HALF_FOV:
                continue
            hidden = False
            for j in range(len(B)):
                if j == k:
                    continue
                t = float(np.dot(B[j] - [x, y], u))
                across = abs(u[0] * (B[j][1] - y) - u[1] * (B[j][0] - x))
                if 0 < t < d - APOTHEM_M and across < 0.8 * CIRCUMRADIUS_M:
                    hidden = True
                    break
            v += not hidden
        views.append(v)
    return views


def scene_level(boxes, cams, block=""):
    """(level, closest centre distance [m], fewest clear views of a box)."""
    B = np.asarray(boxes, float)
    n = len(B)
    gap = min([float(np.hypot(*(B[i] - B[j]))) for i in range(n) for j in range(i + 1, n)] or [9.0])
    mv = min(clear_views(B, cams))
    if gap < TOUCH_M or mv == 0 or (block and block[0] in "CD"):
        return "hard", gap, mv
    return ("medium" if mv == 1 else "easy"), gap, mv


def run_scenes(run_dir, gt_key=None):
    """Boxes and, per round, the camera poses of reading 0 (no yaw offset) of one run."""
    with open(os.path.join(run_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    v = gt.get(gt_key) if gt_key else None
    v = v or list(gt.values())[0]
    boxes = [(b["x"], b["y"]) for b in v if b["class_name"] == "box"]
    cams = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(run_dir, "raw", "P*.jsonl")):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("reading", 0) != 0 or r["trial_id"].split("/")[-1] not in ("+0", "0"):
                    continue
                x, y, yaw = r["camera_pose"]
                cams[int(r["round"])][r["robot_id"]] = (x + LENS_FIX_M * math.cos(yaw),
                                                        y + LENS_FIX_M * math.sin(yaw), yaw)
    return boxes, cams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", default="data/survey_rounds/*", help="runs of the first campaign")
    ap.add_argument("--second", default="data/survey_v2/*", help="runs of the second campaign")
    ap.add_argument("--runs-list", default="results/gnn_mixed_large/detection_full_ring_h3/test_scenes_real_v2.csv",
                    help="CSV with a 'run' column: the run kept for each layout of the second campaign")
    ap.add_argument("--plans", nargs="+", default=["docs/capture_plan_v2.csv", "docs/capture_plan_v3.json"])
    ap.add_argument("--out", default="results/scene_levels.csv")
    a = ap.parse_args()
    block = {}
    for p in a.plans:
        if not os.path.exists(p):
            continue
        if p.endswith(".csv"):
            with open(p, newline="", encoding="utf-8") as f:
                block.update({r["id"]: r["block"] for r in csv.DictReader(f)})
        else:
            with open(p, encoding="utf-8") as f:
                block.update({L["id"]: L["block"] for L in json.load(f)})
    rows = []
    for d in sorted(glob.glob(a.first)):
        if not os.path.isdir(os.path.join(d, "raw")):
            continue
        L = os.path.basename(os.path.normpath(d)).split("_")[0]
        boxes, cams = run_scenes(d)
        for rnd, c in sorted(cams.items()):
            lv, gap, mv = scene_level(boxes, list(c.values()))
            rows.append(dict(campaign="first", layout=L, round=rnd, block="", n_boxes=len(boxes), gap_m=round(gap, 3),
                             min_clear_views=mv, level=lv))
    keep = None
    if a.runs_list and os.path.exists(a.runs_list):
        with open(a.runs_list, newline="") as f:
            keep = {r["run"] for r in csv.DictReader(f)}
    dirs = {os.path.basename(os.path.normpath(d)): d for d in glob.glob(a.second) if os.path.isdir(os.path.join(d, "raw"))}
    from pipeline.build_scans import best_runs
    chosen = []
    for L in sorted({n.split("_")[0] for n in dirs}):
        cand = {n: d for n, d in dirs.items() if n.split("_")[0] == L}
        # the run used by the other evaluations; a folder renamed with a final F may have lost a digit
        hit = [d for n, d in cand.items() if keep and any(r.startswith(n.rstrip("F")) for r in keep)]
        chosen += hit[:1] if hit else best_runs(list(cand.values()))
    for d in sorted(chosen):
        name = os.path.basename(os.path.normpath(d))
        L = name.split("_")[0]
        boxes, cams = run_scenes(d, name.rstrip("F"))
        for rnd, c in sorted(cams.items()):
            lv, gap, mv = scene_level(boxes, list(c.values()), block.get(L, ""))
            rows.append(dict(campaign="second", layout=L, round=rnd, block=block.get(L, ""), n_boxes=len(boxes),
                             gap_m=round(gap, 3), min_clear_views=mv, level=lv))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = collections.Counter((r["campaign"], r["level"]) for r in rows)
    print(f"{len(rows)} scenes -> {a.out}")
    for k in sorted(c):
        print(f"  {k[0]:<7} {k[1]:<7} {c[k]}")


if __name__ == "__main__":
    main()
