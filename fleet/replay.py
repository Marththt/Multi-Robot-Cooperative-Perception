"""Replays the reports of a lab run (fleet/lab.py) through the brain, offline: the same map and online object list
as in the lab, scored against the OptiTrack boxes of its summary.json. To try changes of the fusion on real data
(the aims are replayed as they were: what the gimbal measured cannot change).

    python -m fleet.replay data/fleet_lab/result_20261125_174008
    python -m fleet.replay data/fleet_lab/result --cam-range-scale 1.0      # without the monocular correction

The folder holds summary.json (the truth, the robots, the captain) and reports/<robot>.jsonl.
"""

import argparse
import json
import math
import os

import numpy as np

from fleet import calib, fusion
from fleet.brain import Brain, BrainParams
from fleet.lab import CAM_FLOOR_M, CAM_RANGE_SCALE, IR_ACROSS_M, cam_detections, lab_camera
from fleet.sensing import Frame, GimbalIR
from fleet.world import World, lab_arena


def replay(folder, cam_range_scale=CAM_RANGE_SCALE, regions="config/optitrack_regions.json", picture=True,
           calib_s=20.0):
    with open(os.path.join(folder, "summary.json")) as f:
        s = json.load(f)
    robots, captain, truth = s["robots"], s["captain"], s["truth"]
    rows = []
    for r in robots:
        with open(os.path.join(folder, "reports", f"{r}.jsonl")) as f:
            for line in f:
                if line.strip():
                    rows.append((r, json.loads(line)))
    rows.sort(key=lambda x: x[1]["t"])
    t0 = rows[0][1]["t"]
    fix = dict(s.get("yaw_fix") or {})
    if fix:
        # a run with the correction: the nodes applied it; the rows sent before (the calibration) are left out
        rows = [(r, row) for r, row in rows if abs(float(row.get("yf", fix.get(r, 0.0))) - fix.get(r, 0.0)) < 1e-3]
    elif calib_s > 0:
        # as fleet/lab.py does at the start: the gimbal heading of every robot from its first calib_s seconds
        start = {r: min(row["t"] for q, row in rows if q == r) for r in robots}
        for r in robots:
            first = [row for q, row in rows if q == r and row["t"] - start[r] < calib_s]
            e = calib.estimate([x for x in first if x["type"] == "ir"], [x for x in first if x["type"] == "cam"], truth)
            fix[r] = e[0] if e else 0.0
            print(f"{r}: gimbal heading correction {fix[r]:+.2f} deg" + (f" (IR {e[1]}, camera {e[2]} votes)" if e else
                                                                          " (too few votes: none)"))
        rows = [(r, calib.rotate_row(row, fix[r])) for r, row in rows if row["t"] - start[r] >= calib_s]
    arena = lab_arena(regions)
    cam, irm = lab_camera(), GimbalIR()
    brain = Brain(arena, robots, captain, cam, BrainParams(mode=s.get("mode", "coverage"), sensors="gimbal",
                                                                  cam_floor_m=CAM_FLOOR_M, ir_across_m=IR_ACROSS_M))
    pose = {r: None for r in robots}
    traj = {r: [] for r in robots}
    next_foot = 0.0
    for r, row in rows:
        t = row["t"] - t0
        brain.t = t
        k = row.get("type")
        known = all(q is not None for q in pose.values())
        if k == "status":
            if row.get("pose") is not None:
                pose[r] = tuple(row["pose"])
        elif k == "ir" and known:
            brain.add_ir(r, row["o"], row["h"], row["d"], irm.r_max_m, dict(pose))
        elif k == "cam" and known:
            lens = tuple(row["lens"])
            brain.camera_frame(Frame(r, t, lens, detections=cam_detections(row, cam_range_scale)), dict(pose))
        elif k == "aim" and row.get("o") is not None:
            brain.ir_measure(r, row["task"], row["o"], row["h"], row["d"], irm)
        if known and t >= next_foot:
            for q in robots:
                brain.add_footprint(pose[q])
                traj[q].append(pose[q])
            next_foot = t + 0.25
    tr = brain.tracker
    est = [tuple(float(v) for v in k.xy) for k in tr.confirmed()]
    sc = fusion.score(est, truth)
    if picture:
        from fleet.sim import draw
        objs = [dict(xy=tuple(float(v) for v in k.xy), kept=k.state == "confirmed", state=k.state)
                for k in tr.tracks if k.state != "merged"]
        out = dict(s, **{k: v for k, v in sc.items() if k != "matches"})
        draw(os.path.join(folder, "replay.png"), World(arena, [], truth), brain.fine,
             (1.0 - cam.p_d_near) ** brain.views, {r: v or [(0.0, 0.0, 0.0)] for r, v in traj.items()}, objs, out,
             brain)
    return brain, est, sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--cam-range-scale", type=float, default=CAM_RANGE_SCALE)
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--calib-s", type=float, default=20.0,
                    help="gimbal heading correction from the first seconds, as the lab does (0 = none; a run that "
                         "had it, summary.json yaw_fix, is replayed as it was)")
    a = ap.parse_args()
    brain, est, sc = replay(a.folder, a.cam_range_scale, a.regions, calib_s=a.calib_s)
    tr = brain.tracker
    print(f"{len(est)} boxes reported, {len(tr.tentative())} tentative, OptiTrack {sc['n_true']}: TP {sc['tp']} "
          f"FP {sc['fp']} FN {sc['fn']}" + (f", mean error {100 * sc['mean_err_m']:.1f} cm"
                                            if sc.get("mean_err_m") is not None else ""))
    for k in tr.tracks:
        if k.state in ("confirmed", "tentative"):
            print(f"  {k.state:9s} ({k.xy[0]:+.2f}, {k.xy[1]:+.2f})  camera {k.n_cam}/{k.looks}  IR {k.n_ir} "
                  f"(missed {k.ir_miss})  {sorted(k.robots)}")
    print(f"-> {os.path.join(a.folder, 'replay.png')}")


if __name__ == "__main__":
    main()
