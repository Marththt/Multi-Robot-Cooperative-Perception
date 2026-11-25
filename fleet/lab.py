"""The fleet in the lab: the brain of fleet/brain.py (the same as in the simulation) driving the real robots.

It starts one fleet/robot_node.py per robot (one after the other), reads what they report (poses, IR beams,
YOLO rays, aimed IR readings: --dir/reports/<robot>.jsonl), keeps the map and the online object list, and twice a
second writes every robot's goal (--dir/goals/<robot>.json: where to go, where to face, fan or aim the gimbal).
The live window (fleet/live.py) shows the map, the robots, the beams and the objects; the boxes OptiTrack tracks
are drawn as the truth (they are used for nothing else, except the robots' bumper). At the end (coverage goal
reached and nothing tentative, --t-max-s, Ctrl+C or the window closed) every robot stops and re-centres its
gimbal, and --dir/result_<date>_<time>/ (one per run) gets summary.json (the objects, scored against the OptiTrack boxes), map.png and the
reports.

First, every run calibrates (--calib-s, 20 s): the robots stand with the gimbal fanning all around, and the
heading of every gimbal is measured against the OptiTrack boxes (fleet/calib.py: the SDK's gimbal yaw moved by
10-17 deg between two sessions); the correction goes to the robot (yaw_fix of its goal) and is in summary.json.
A robot without a clear heading tries again every 5 s and goes on without one after 15 s more (it says so).

The tests, one step at a time (same command, fewer robots / more freedom):

  1  one robot standing, fan only: is the IR map right?
        python -m fleet.lab --robot G11:8 --no-drive --no-camera
  2  one robot standing, camera too: YOLO rays, objects, aimed IR readings, nothing moves
        python -m fleet.lab --robot G11:8 --no-drive
  3  one robot exploring alone
        python -m fleet.lab --robot G11:8
  4  two robots standing, both fanning: do their IR beams disturb each other? (--ir-check prints, per robot,
     how many readings end on a box OptiTrack tracks, on the arena border, or on nothing known)
        python -m fleet.lab --robot G11:8 --robot G12:7 --no-drive --no-camera --ir-check
  5  the fleet: captain + followers
        python -m fleet.lab --robot G11:8 --robot G12:7 --robot G14:10 --captain G11

Every test runs here without hardware first: --sim (the robots of --robot ID:_:X,Y,YAW_DEG, the scene of
--sim-world or a random one of --seed):
        python -m fleet.lab --sim --robot S1::-2.2,-0.6,0 --robot S2::-2.2,0.1,0 --robot S3::-2.2,-1.3,0

Options not listed here go to every robot node (e.g. --v-max 0.1, --fan-dps 20, --ir-index 2; --sim with
--sim-yaw-error-deg 14 gives the simulated gimbals the error the calibration has to find).

A run can be replayed offline through the brain (python -m fleet.replay data/fleet_lab/result_<date>_<time>).
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from fleet import calib
from fleet.brain import Brain, BrainParams
from fleet.robot_node import CONNECTED_MSG
from fleet.sensing import CameraModel, Frame, GimbalIR
from fleet.world import World, lab_arena
from vf.control.goto import wrap_pi

CONNECT_TIMEOUT_S = 120.0

# The monocular range of YOLO (yolo_gnn/hybrid.frame_rays) is divided by this. Lab test of 2026-11-25, once the
# gimbal heading was corrected (fleet/calib.py): 0.92-0.96 x the distance of the box centre: left as it comes.
CAM_RANGE_SCALE = 1.0
# Calibration at the start (fleet/calib.py): the robots stand and fan the gimbal all around, so that each sees the
# boxes near it whatever its heading; one that has no clear heading after CALIB_EXTRA_S more goes on without.
CALIB_FAN_DEG = 170.0
CALIB_FAN_DPS = 40.0
CALIB_EXTRA_S = 15.0
# The rays of one robot share a bias (heading residual, monocular range): an object's camera position is never
# trusted below this, so that an aimed IR reading 10-20 cm away is still accepted and then dominates.
CAM_FLOOR_M = 0.08
# An aimed reading gives the box centre one apothem behind the point the beam hit: right along the beam, up to half
# a box off across it (the sweep of the aim picks the beam nearest the object, not the face centre).
IR_ACROSS_M = 0.04


def lab_camera():
    """The camera model of the brain in the lab: a wider monocular range error than in the simulation (the
    residual after CAM_RANGE_SCALE was 0.1-0.3 x the range)."""
    return CameraModel(sd_range_m=(0.05, 0.25), sd_bearing_deg=2.0, p_d_near=0.85, p_d_far=0.57, fp_rate=0.1)


def cam_detections(row, scale=CAM_RANGE_SCALE):
    """A 'cam' report -> detections for the brain: the rays from the lens, the range divided by scale."""
    lens = row["lens"]
    out = []
    for ang, d in row["rays"]:
        d = d / scale
        out.append(dict(xy=(lens[0] + d * math.cos(ang), lens[1] + d * math.sin(ang)), range=d,
                        bearing=wrap_pi(ang - lens[2]), true=None))
    return out


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for _ in range(20):                       # Windows cannot replace a file another process has open
        try:
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.01)
    return False


def pump(proc, tag, connected):
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line.strip():
            print(f"{tag:>4} | {line}", flush=True)
        if CONNECTED_MSG in line:
            connected.set()
    connected.set()


class Tail:
    """New lines of a file that another process appends to."""

    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, ""

    def lines(self):
        try:
            with open(self.path) as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        self.buf += data
        *done, self.buf = self.buf.split("\n")
        out = []
        for ln in done:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out


class IRCheck:
    """Per robot: where its IR readings end, against what OptiTrack tracks (test 4, interference)."""

    def __init__(self, robots):
        self.n = {r: dict(box=0, border=0, unknown=0, free=0) for r in robots}

    def add(self, r, row, boxes, arena):
        if row["d"] is None:
            self.n[r]["free"] += 1
            return
        ex = row["o"][0] + row["d"] * math.cos(row["h"])
        ey = row["o"][1] + row["d"] * math.sin(row["h"])
        if any(math.hypot(ex - b[0], ey - b[1]) < 0.15 for b in boxes):
            self.n[r]["box"] += 1
        elif arena and min(abs(ex - arena[0]), abs(ex - arena[1]), abs(ey - arena[2]), abs(ey - arena[3])) < 0.15:
            self.n[r]["border"] += 1
        else:
            self.n[r]["unknown"] += 1

    def line(self):
        out = []
        for r, c in self.n.items():
            tot = sum(c.values()) or 1
            out.append(f"{r}: {tot} readings, on a box {100 * c['box'] / tot:.0f}%, border "
                       f"{100 * c['border'] / tot:.0f}%, nothing known {100 * c['unknown'] / tot:.0f}%, "
                       f"no echo {100 * c['free'] / tot:.0f}%")
        return "IR check | " + "; ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", action="append", required=True, metavar="ID:RIGID_BODY[:X,Y,YAW_DEG]")
    ap.add_argument("--captain", default=None, help="default: the first --robot")
    ap.add_argument("--mode", choices=["coverage", "formation"], default="coverage")
    ap.add_argument("--dir", default="data/fleet_lab")
    ap.add_argument("--no-drive", action="store_true", help="nothing moves: sensing, map, objects and plans only")
    ap.add_argument("--no-camera", action="store_true", help="no YOLO: the IR map only")
    ap.add_argument("--no-aim", action="store_true", help="never aim the IR at objects (fan only)")
    ap.add_argument("--ir-check", action="store_true", help="print where the IR readings end, every 5 s")
    ap.add_argument("--t-max-s", type=float, default=600.0)
    ap.add_argument("--calib-s", type=float, default=20.0,
                    help="seconds standing and fanning at the start to measure the gimbal heading of every robot "
                         "against the OptiTrack boxes (fleet/calib.py; 0 = none; --sim: against its world)")
    ap.add_argument("--cam-range-scale", type=float, default=CAM_RANGE_SCALE,
                    help="the monocular YOLO range is divided by this (1 = as it comes)")
    ap.add_argument("--tick-s", type=float, default=0.5)
    ap.add_argument("--coverage-goal", type=float, default=0.98)
    ap.add_argument("--no-live", action="store_true", help="no window")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--optitrack-server", default="127.0.0.1")
    ap.add_argument("--optitrack-local", default="127.0.0.1")
    ap.add_argument("--sim", action="store_true", help="no hardware: simulated robots")
    ap.add_argument("--sim-world", default=None, help="--sim: scene file (default: random, --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--obstacles", type=int, default=3)
    a, rest = ap.parse_known_args()

    robots = []
    for spec in a.robot:
        parts = spec.split(":")
        robots.append(dict(id=parts[0], rb=int(parts[1]) if len(parts) > 1 and parts[1] else None,
                           start=[float(v) for v in parts[2].split(",")] if len(parts) > 2 and parts[2] else None))
    ids = [r["id"] for r in robots]
    captain = a.captain or ids[0]
    if captain not in ids:
        ap.error(f"--captain {captain} is not one of the robots {ids}")
    arena = lab_arena(a.regions)

    for sub in ("reports", "goals"):
        shutil.rmtree(os.path.join(a.dir, sub), ignore_errors=True)
        os.makedirs(os.path.join(a.dir, sub), exist_ok=True)
    stop_file = os.path.join(a.dir, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)

    # the truth: the world file (sim) or the boxes OptiTrack tracks (only drawn and scored)
    hub = None
    if a.sim:
        for k, r in enumerate(robots):
            r["start"] = r["start"] or [arena[0] + 0.45, (arena[2] + arena[3]) / 2 + (k - (len(robots) - 1) / 2) * 0.7,
                                        0.0]
        if a.sim_world:
            world = World.load(a.sim_world)
        else:
            world = World.random(np.random.default_rng(a.seed), arena, a.targets, a.obstacles,
                                 keep_free=[r["start"][:2] for r in robots])
        world_path = os.path.join(a.dir, "world.json")
        world.save(world_path)
        truth = lambda: world.targets.tolist()
    else:
        from acquisition.mocap_all import AllBodiesHub
        if any(r["rb"] is None for r in robots):
            ap.error("every --robot needs its rigid body: ID:RIGID_BODY")
        hub = AllBodiesHub([r["rb"] for r in robots], a.optitrack_server, a.optitrack_local)
        hub.wait_ready()
        time.sleep(0.3)
        rbs = {r["rb"] for r in robots}
        truth = lambda: [[b["x"], b["y"]] for b in hub.boxes(arena, exclude=rbs)]
        world = World(arena, [], truth())
        print(f"boxes OptiTrack tracks inside the arena (drawn as the truth): {len(world.targets)}", flush=True)

    # robot nodes, one after the other
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    procs = []
    for r in robots:
        if stop.is_set():
            break
        cmd = [sys.executable, "-u", "-m", "fleet.robot_node", "--robot-id", r["id"], "--dir", a.dir,
               "--regions", a.regions]
        if a.no_drive:
            cmd += ["--no-drive"]
        if a.no_camera:
            cmd += ["--no-camera"]
        if a.sim:
            x, y, yaw = r["start"]
            cmd += ["--sim", "--sim-world", world_path, "--sim-pose", str(x), str(y), str(yaw), "--seed",
                    str(a.seed + len(procs))]
        else:
            cmd += ["--rb", str(r["rb"]), "--optitrack-server", a.optitrack_server, "--optitrack-local",
                    a.optitrack_local, "--robot-rbs"] + [str(q["rb"]) for q in robots if q is not r]
        cmd += rest
        print(f"{r['id']}: starting" + (" (simulated)" if a.sim else f" (rigid body {r['rb']})"), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        connected = threading.Event()
        threading.Thread(target=pump, args=(proc, r["id"], connected), daemon=True).start()
        procs.append(proc)
        t0 = time.time()
        while not connected.is_set() and not stop.is_set() and time.time() - t0 < CONNECT_TIMEOUT_S:
            time.sleep(0.1)
        if proc.poll() is not None:
            print(f"{r['id']}: the node exited (see its lines above)", flush=True)
        elif not connected.is_set():
            print(f"{r['id']}: not running after {CONNECT_TIMEOUT_S:.0f} s", flush=True)

    try:
        result = run(a, robots, ids, captain, arena, world, truth, procs, stop)
    finally:
        open(stop_file, "w").close()
        t0 = time.time()
        while any(p.poll() is None for p in procs) and time.time() - t0 < 15:
            time.sleep(0.2)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        if hub is not None:
            hub.close()
    print("stopped", flush=True)
    return result


def run(a, robots, ids, captain, arena, world, truth, procs, stop):
    cam = lab_camera() if not a.sim else CameraModel(sd_range_m=(0.02, 0.15), sd_bearing_deg=1.0, p_d_near=0.85,
                                                      p_d_far=0.57, fp_rate=0.1)
    scale = a.cam_range_scale if not a.sim else 1.0        # the simulated camera has no such bias
    irm = GimbalIR()
    p = BrainParams(mode=a.mode, sensors="gimbal", captain_plan="frontier", coverage_goal=a.coverage_goal,
                    cam_floor_m=0.0 if a.sim else CAM_FLOOR_M, ir_across_m=0.0 if a.sim else IR_ACROSS_M)
    tails = {r: Tail(os.path.join(a.dir, "reports", f"{r}.jsonl")) for r in ids}
    pose = {r: None for r in ids}
    gim = {r: dict(gy=0.0, task=None, sent=0.0) for r in ids}
    check = IRCheck(ids) if a.ir_check else None
    counts = dict(ir=0, cam=0, aim=0)
    traj = {r: [] for r in ids}
    boxes_now = truth()

    # Gimbal heading of every robot (fleet/calib.py). state: "calib" = standing, gimbal fanning all around,
    # nothing to the brain; "late" = no clear heading after CALIB_EXTRA_S more: it goes on without correction and
    # its heading is estimated again every 5 s; "ok" = it has its correction. Every reading is logged: when a late
    # robot gets its correction, the map and the objects are rebuilt from the log with the right headings.
    fix = {r: 0.0 for r in ids}
    a.yaw_fix = fix
    state = {r: "calib" if a.calib_s > 0 else "ok" for r in ids}
    raw = {r: dict(ir=[], cam=[]) for r in ids}      # readings without correction, for the estimate
    log = []                                         # (kind, robot, t, row, poses) for a rebuild
    started = False
    brain = None

    def feed(kind, r, t, row, poses):
        """One reading into the brain, turned by the correction the robot did not apply itself yet."""
        if kind == "foot":
            for q in ids:
                brain.add_footprint(poses[q])
            return
        row = calib.rotate_row(row, fix[r] - float(row.get("yf", 0.0)))
        brain.t = t
        if kind == "ir":
            brain.add_ir(r, row["o"], row["h"], row["d"], irm.r_max_m, poses)
        elif kind == "cam":
            brain.camera_frame(Frame(r, t, tuple(row["lens"]), detections=cam_detections(row, scale)), poses)
        elif kind == "aim" and row.get("o") is not None:
            brain.ir_measure(r, row["task"], row["o"], row["h"], row["d"], irm)

    def rebuild(t):
        nonlocal brain
        brain = Brain(arena, ids, captain, cam, p)
        for kind, r, tr, row, poses in log:
            feed(kind, r, tr, row, poses)
        brain.t = t
        for r in ids:
            gim[r]["task"] = None

    view = None
    if not a.no_live:
        from fleet.live import LiveView
        view = LiveView(world, Brain(arena, ids, captain, cam, p).fine, cam, ids, captain, a.mode)
    t0 = time.time()
    seq, next_tick, next_view, next_check, next_foot = 0, 0.0, 0.0, 5.0, 0.0
    next_calib, next_wide = a.calib_s, 0.0
    why = "time limit"
    if a.calib_s > 0:
        print(f"calibration: {a.calib_s:.0f} s standing, the gimbal fanning all around, then the heading of every "
              f"robot from the OptiTrack boxes", flush=True)
    while not stop.is_set():
        t = time.time() - t0
        if t > a.t_max_s:
            break
        if all(pr.poll() is not None for pr in procs):
            why = "every robot node exited"
            break
        if any(v == "calib" for v in state.values()) and t >= next_wide:
            for r in ids:                            # still calibrating: standing, gimbal fanning all around
                if state[r] == "calib":
                    write_json(os.path.join(a.dir, "goals", f"{r}.json"), dict(
                        seq=-int(t * 10), t=time.time(), target=None, yaw=None, v_max=None,
                        gimbal={"mode": "fan", "fan_deg": CALIB_FAN_DEG, "fan_dps": CALIB_FAN_DPS}))
            next_wide = t + 0.5
        if any(v != "ok" for v in state.values()) and t >= next_calib:
            boxes_now = truth()
            late_fixed = False
            for r in ids:
                if state[r] == "ok":
                    continue
                e = calib.estimate(raw[r]["ir"], raw[r]["cam"], boxes_now)
                if e is None:
                    if state[r] == "calib" and t >= a.calib_s + CALIB_EXTRA_S:
                        state[r] = "late"
                        print(f"calibration {r}: no clear heading after {t:.0f} s ({len(raw[r]['ir'])} beams, "
                              f"{len(raw[r]['cam'])} frames): it goes on WITHOUT correction for now, and tries "
                              f"again every 5 s on what it sees (boxes nearer than ~1.5 m help)", flush=True)
                    elif state[r] == "calib" and t < a.calib_s + 1:
                        print(f"calibration {r}: no clear heading yet ({len(raw[r]['ir'])} beams, "
                              f"{len(raw[r]['cam'])} frames), trying again every 5 s", flush=True)
                    continue
                late_fixed |= state[r] == "late"
                fix[r], state[r] = round(e[0], 2), "ok"
                print(f"calibration {r} at {t:.0f} s: gimbal heading correction {fix[r]:+.2f} deg "
                      f"({e[1]} IR beams and {e[2]} YOLO rays agree with the boxes)", flush=True)
            if started and late_fixed:
                rebuild(t)
                print(f"map and objects rebuilt with the new correction ({len(log)} readings)", flush=True)
            next_calib = t + 5.0
        poses = {r: pose[r] for r in ids}
        for r in ids:
            for row in tails[r].lines():
                k = row.get("type")
                if k == "status":
                    if row.get("pose") is not None:
                        pose[r] = tuple(row["pose"])
                    gim[r]["gy"] = row.get("gy") or 0.0
                    continue
                if k not in ("ir", "cam", "aim") or any(q is None for q in poses.values()):
                    continue
                if state[r] != "ok" and k in ("ir", "cam") and not row.get("yf"):
                    raw[r][k].append(row)
                entry = (k, r, t, row, dict(poses))
                log.append(entry)
                if started:
                    feed(*entry)
                counts[k] += 1
                if k == "ir" and check:
                    check.add(r, calib.rotate_row(row, fix[r] - float(row.get("yf", 0.0))), boxes_now, arena)
                if k == "aim" and gim[r]["task"] == row["task"]:
                    gim[r]["task"] = None
        if not started and all(v != "calib" for v in state.values()) and all(q is not None for q in pose.values()):
            started = True                           # the brain starts: everything seen so far, corrected
            rebuild(t)
            next_tick = t
        known = started and all(q is not None for q in pose.values())
        if known and t >= next_foot:
            log.append(("foot", None, t, None, dict(pose)))
            feed("foot", None, t, None, dict(pose))
            for r in ids:
                traj[r].append(pose[r])
            next_foot = t + 0.25
        if known and t >= next_tick:
            poses = {r: pose[r] for r in ids}
            brain.tick(t, poses)
            for r in ids:                                    # an aim not answered in 10 s is dropped
                if gim[r]["task"] is not None and t - gim[r]["sent"] > 10.0:
                    gim[r]["task"] = None
            if not a.no_aim:
                idle = [r for r in ids if gim[r]["task"] is None and state[r] == "ok"]
                for r, tid in brain.aim_tasks(poses, idle).items():
                    gim[r].update(task=tid, sent=t)
            seq += 1
            for r in ids:
                g = brain.goals.get(r) or {}
                k = brain.tracker.get(gim[r]["task"]) if gim[r]["task"] is not None else None
                gimbal = {"mode": "fan"} if k is None else {"mode": "aim", "task": k.id,
                                                            "xy": [float(v) for v in k.xy]}
                write_json(os.path.join(a.dir, "goals", f"{r}.json"), dict(
                    seq=seq, t=time.time(), target=None if a.no_drive else (list(g["target"]) if g else None),
                    yaw=g.get("yaw"), v_max=g.get("v_max"), gimbal=gimbal, yaw_fix=fix[r]))
            next_tick = t + a.tick_s
            if brain.finished() and not a.no_drive and all(v == "ok" for v in state.values()):
                why = "coverage goal reached, nothing tentative"
                break
        if check and t >= next_check:
            boxes_now = truth()
            print(check.line(), flush=True)
            next_check = t + 5.0
        if view is not None and t >= next_view and known:
            if not view.alive():
                why = "window closed"
                break
            view.update(t, pose, brain, [tuple(k.xy) for k in brain.tracker.confirmed()], gim)
            next_view = t + 0.2
        time.sleep(0.02)
    else:
        why = "Ctrl+C"
    if brain is None:
        rebuild(time.time() - t0)
    return finish(a, brain, world, truth, traj, counts, captain, why, time.time() - t0, view)


def describe(a, robots):
    """What the run was, in words, for summary.json."""
    where = "simulated robots (--sim)" if a.sim else "real robots"
    n = len(robots)
    if a.no_drive:
        what = f"{n} robot(s) standing (--no-drive)"
    else:
        what = "one robot exploring alone" if n == 1 else f"the fleet: {n} robots, captain + followers"
    sensors = "IR fan only (--no-camera)" if a.no_camera else "IR fan + camera (YOLO)"
    extra = [x for x, on in (("no aiming (--no-aim)", a.no_aim), ("IR interference check (--ir-check)", a.ir_check))
             if on]
    return ", ".join([what, sensors, where] + extra)


def finish(a, brain, world, truth, traj, counts, captain, why, t, view):
    from fleet import fusion
    from fleet.sim import draw
    tr = brain.tracker
    objs = [dict(xy=tuple(float(v) for v in k.xy), kept=k.state == "confirmed", state=k.state, n_det=k.n_cam,
                 looks=k.looks, n_ir=k.n_ir, ir_miss=k.ir_miss, robots=sorted(k.robots))
            for k in tr.tracks if k.state != "merged"]
    est = [o["xy"] for o in objs if o["kept"]]
    boxes = truth()
    sc = fusion.score(est, boxes)
    s = dict(yaw_fix=getattr(a, "yaw_fix", None), run=dict(what=describe(a, traj), command="python -m fleet.lab " + " ".join(sys.argv[1:]),
                      date=time.strftime("%Y-%m-%d %H:%M:%S")),
             why=why, t_s=round(t, 1), robots=list(traj), captain=captain, mode=a.mode, seed=a.seed,
             coverage_brain=brain.coverage(), coverage_true=brain.coverage(), reports=counts,
             contacts=0, **{k: v for k, v in sc.items() if k != "matches"},
             estimates=est, tentative=[o["xy"] for o in objs if o["state"] == "tentative"], objects=objs,
             truth=[list(b) for b in boxes])
    out = os.path.join(a.dir, "result_" + time.strftime("%Y%m%d_%H%M%S"))     # one folder per run, never overwritten
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    w = World(world.arena, world.obstacles, boxes)
    phi = (1.0 - brain.cam.p_d_near) ** brain.views
    tj = {r: v if v else [(0.0, 0.0, 0.0)] for r, v in traj.items()}
    draw(os.path.join(out, "map.png"), w, brain.fine, phi, tj, objs, s, brain)
    src, dst = os.path.join(a.dir, "reports"), os.path.join(out, "reports")
    os.makedirs(dst, exist_ok=True)                  # not copytree(dirs_exist_ok=...): the lab PC has Python 3.7
    for name in os.listdir(src) if os.path.isdir(src) else []:
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    print(f"\n{why} after {t:.0f} s: {len(est)} boxes reported, {len(s['tentative'])} still tentative; "
          f"OptiTrack tracks {len(boxes)} (TP {sc['tp']} FP {sc['fp']} FN {sc['fn']}"
          + (f", mean error {100 * sc['mean_err_m']:.1f} cm" if sc.get("mean_err_m") is not None else "") + ")")
    print(f"reports: {counts['ir']} IR beams, {counts['cam']} camera frames, {counts['aim']} aimed readings")
    print(f"-> {out}/summary.json, map.png", flush=True)
    if view is not None and view.alive():
        print("close the window to end", flush=True)
        view.hold()
    return s


if __name__ == "__main__":
    main()
