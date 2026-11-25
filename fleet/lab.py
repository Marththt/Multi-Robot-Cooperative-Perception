"""The fleet in the lab: the brain of fleet/brain.py (the same as in the simulation) driving the real robots.

It starts one fleet/robot_node.py per robot (one after the other), reads what they report (poses, IR beams,
YOLO rays, aimed IR readings: --dir/reports/<robot>.jsonl), keeps the map and the online object list, and twice a
second writes every robot's goal (--dir/goals/<robot>.json: where to go, where to face, fan or aim the gimbal).
The live window (fleet/live.py) shows the map, the robots, the beams and the objects; the boxes OptiTrack tracks
are drawn as the truth (they are used for nothing else, except the robots' bumper). At the end (coverage goal
reached and nothing tentative, --t-max-s, Ctrl+C or the window closed) every robot stops and re-centres its
gimbal, and --dir/result/ gets summary.json (the objects, scored against the OptiTrack boxes), map.png and the
reports.

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

Options not listed here go to every robot node (e.g. --v-max 0.1, --fan-dps 20, --ir-index 2).
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

from fleet.brain import Brain, BrainParams
from fleet.robot_node import CONNECTED_MSG
from fleet.sensing import CameraModel, Frame, GimbalIR
from fleet.world import World, lab_arena
from vf.control.goto import wrap_pi

CONNECT_TIMEOUT_S = 120.0


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
    cam = CameraModel(sd_range_m=(0.02, 0.15), sd_bearing_deg=1.0, p_d_near=0.85, p_d_far=0.57, fp_rate=0.1)
    irm = GimbalIR()
    p = BrainParams(mode=a.mode, sensors="gimbal", captain_plan="frontier", coverage_goal=a.coverage_goal)
    brain = Brain(arena, ids, captain, cam, p)
    tails = {r: Tail(os.path.join(a.dir, "reports", f"{r}.jsonl")) for r in ids}
    pose = {r: None for r in ids}
    gim = {r: dict(gy=0.0, task=None, sent=0.0) for r in ids}
    check = IRCheck(ids) if a.ir_check else None
    view = None
    if not a.no_live:
        from fleet.live import LiveView
        view = LiveView(world, brain.fine, cam, ids, captain, a.mode)
    t0 = time.time()
    seq, next_tick, next_view, next_check, next_foot = 0, 0.0, 0.0, 5.0, 0.0
    counts = dict(ir=0, cam=0, aim=0)
    traj = {r: [] for r in ids}
    boxes_now = truth()
    why = "time limit"
    while not stop.is_set():
        t = time.time() - t0
        if t > a.t_max_s:
            break
        if all(pr.poll() is not None for pr in procs):
            why = "every robot node exited"
            break
        poses = {r: pose[r] for r in ids}
        for r in ids:
            for row in tails[r].lines():
                k = row.get("type")
                if k == "status":
                    if row.get("pose") is not None:
                        pose[r] = tuple(row["pose"])
                    gim[r]["gy"] = row.get("gy") or 0.0
                elif k == "ir" and all(q is not None for q in poses.values()):
                    brain.add_ir(r, row["o"], row["h"], row["d"], irm.r_max_m, poses)
                    counts["ir"] += 1
                    if check:
                        check.add(r, row, boxes_now, arena)
                elif k == "cam" and all(q is not None for q in poses.values()):
                    lens = tuple(row["lens"])
                    dets = [dict(xy=(lens[0] + d * math.cos(ang), lens[1] + d * math.sin(ang)), range=d,
                                 bearing=wrap_pi(ang - lens[2]), true=None) for ang, d in row["rays"]]
                    brain.t = t
                    brain.camera_frame(Frame(r, t, lens, detections=dets), poses)
                    counts["cam"] += 1
                elif k == "aim":
                    if row.get("o") is not None:
                        brain.t = t
                        brain.ir_measure(r, row["task"], row["o"], row["h"], row["d"], irm)
                    if gim[r]["task"] == row["task"]:
                        gim[r]["task"] = None
                    counts["aim"] += 1
        known = all(q is not None for q in pose.values())
        if known and t >= next_foot:
            for r in ids:
                brain.add_footprint(pose[r])
                traj[r].append(pose[r])
            next_foot = t + 0.25
        if known and t >= next_tick:
            poses = {r: pose[r] for r in ids}
            brain.tick(t, poses)
            for r in ids:                                    # an aim not answered in 10 s is dropped
                if gim[r]["task"] is not None and t - gim[r]["sent"] > 10.0:
                    gim[r]["task"] = None
            if not a.no_aim:
                for r, tid in brain.aim_tasks(poses, [r for r in ids if gim[r]["task"] is None]).items():
                    gim[r].update(task=tid, sent=t)
            seq += 1
            for r in ids:
                g = brain.goals.get(r) or {}
                k = brain.tracker.get(gim[r]["task"]) if gim[r]["task"] is not None else None
                gimbal = {"mode": "fan"} if k is None else {"mode": "aim", "task": k.id,
                                                            "xy": [float(v) for v in k.xy]}
                write_json(os.path.join(a.dir, "goals", f"{r}.json"), dict(
                    seq=seq, t=time.time(), target=None if a.no_drive else (list(g["target"]) if g else None),
                    yaw=g.get("yaw"), v_max=g.get("v_max"), gimbal=gimbal))
            next_tick = t + a.tick_s
            if brain.finished() and not a.no_drive:
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
    return finish(a, brain, world, truth, traj, counts, captain, why, time.time() - t0, view)


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
    s = dict(why=why, t_s=round(t, 1), robots=list(traj), captain=captain, mode=a.mode, seed=a.seed,
             coverage_brain=brain.coverage(), coverage_true=brain.coverage(), reports=counts,
             contacts=0, **{k: v for k, v in sc.items() if k != "matches"},
             estimates=est, tentative=[o["xy"] for o in objs if o["state"] == "tentative"], objects=objs,
             truth=[list(b) for b in boxes])
    out = os.path.join(a.dir, "result")
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
