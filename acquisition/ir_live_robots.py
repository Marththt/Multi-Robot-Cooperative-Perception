"""Several robots ranging the boxes with the IR on their gimbal, live, for the demo page (MHT and the graph network).

Robot side, as demo/live_robots.py but with the IR instead of the markers: one acquisition/ir_scan.py process per
robot (started one after the other: the SDK finds a robot through a port only one process at a time can use), and
<out>/boxes.json with the boxes OptiTrack tracks, in the format of demo/live_robots.py. The boxes are found by
themselves: every rigid body tracked inside the arena that is not one of the robots (acquisition/mocap_all.py),
so boxes can be added or taken away during the run; --box restricts them to a list. Then the page:

    terminal 1 (capture environment: SDK, natnet)
        python -m acquisition.ir_live_robots --robot G11:8 --robot G12:7 --robot G14:10 --out data/live_ir/
    terminal 2 (demo environment)
        python -m demo.live --live-dir data/live_ir/ --sensor-calib config/sensor_calib_ir.json --window-s 12 \\
            --stale-s 4 --no-images

--window-s must cover one sweep of a robot (about 1.5 s per box); config/sensor_calib_ir.json leaves the IR ranges
as they are (the ArUco range correction of config/sensor_calib.json must not be applied to them). Model C needs the
frames and is left out (--no-images).

--robot is ID:RIGID_BODY. Every robot uses its own gimbal aiming offset and IR range correction
(acquisition/ir_live.py YAW_OFFSET_DEG, IR_FIT); a robot not in those tables gets the values measured on G11 (same
camera and IR model). Its gimbal aiming offset also depends on how its rigid body was defined in Motive: if its
aim is off, measure it with ir_live sweep and add it to YAW_OFFSET_DEG. Other options (--ir-index, --ir-samples, --max-range, ...) are passed to
every robot. Ctrl+C stops every robot (the gimbals are re-centred).

Without hardware: --sim --sim-layout L4 --robot S1:0:-1.6,-0.4,-20 --robot S2:0:0.6,-1.8,110 (ID:_:X,Y,YAW_DEG):
fake gimbals and IR on the boxes of the layout, boxes.json written from it.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

CONNECT_TIMEOUT_S = 180.0         # connection + IR slot + aim calibration (a yaw sweep, about 40 s)


def pump(proc, tag, connected):
    from acquisition.ir_scan import CONNECTED_MSG
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        try:
            if line.strip():
                print(f"{tag:>4} | {line}", flush=True)
        except Exception:
            pass
        if CONNECTED_MSG in line:
            connected.set()
    connected.set()


def truth_loop(a, robots, stop, hz=30.0):
    """<out>/boxes.json, as demo/live_robots.truth_loop writes it, with the boxes found by acquisition/mocap_all."""
    from acquisition.mocap_all import AllBodiesHub
    from demo.methods import write_json
    from vf.fusion.sensor_model import arena_from_regions
    arena = arena_from_regions(a.regions)
    rbs = [r["rb"] for r in robots]
    hub = AllBodiesHub(rbs, a.optitrack_server, a.optitrack_local)
    seen = None
    try:
        while not stop.is_set():
            t0 = time.time()
            boxes = hub.boxes(arena, exclude=set(rbs), only=set(a.box) if a.box else None)
            ids = [b["id"] for b in boxes]
            if ids != seen:
                print(f"boxes tracked inside the arena: {ids}", flush=True)
                seen = ids
            poses = {}
            for r in robots:
                q = hub.get_pose(r["rb"])
                if q is not None:
                    poses[r["robot_id"]] = [round(float(v), 4) for v in q[:3]]
            write_json(os.path.join(a.out, "boxes.json"), dict(
                t=time.time(), robots=poses, boxes=[dict(class_name="box", rb_id=b["id"], x=round(b["x"], 4),
                                                         y=round(b["y"], 4), yaw=round(b["yaw"], 4)) for b in boxes]))
            stop.wait(max(0.0, 1.0 / hz - (time.time() - t0)))
    finally:
        hub.close()


def coordinator(a, robots, stop, timeout_s=8.0):
    """Turn taking: one robot at a time ranges one box (acquisition/ir_scan.py --turns), round robin, so that the
    beams of two robots never meet on a box (the IR cannot be switched off: the others point theirs upwards)."""
    from demo.methods import read_json, write_json
    seq, n = int(time.time() * 1000), 0
    while not stop.is_set():
        for r in robots:
            if stop.is_set():
                return
            seq += 1
            write_json(os.path.join(a.out, "turn.json"), dict(robot=r["robot_id"], seq=seq, t=time.time()))
            t0 = time.time()
            while not stop.is_set():
                d = read_json(os.path.join(a.out, f"{r['robot_id']}.done.json"))
                if d and d.get("seq") == seq:
                    break
                if time.time() - t0 > timeout_s:
                    print(f"{r['robot_id']}: no answer to its turn in {timeout_s:.0f} s, next robot", flush=True)
                    break
                time.sleep(0.02)
        n += 1
        if n == 1:
            print("first round of turns done", flush=True)


def sim_truth(a, stop):
    from demo.methods import write_json
    with open(a.ground_truth) as f:
        gt = json.load(f)[a.sim_layout]
    boxes = [dict(class_name="box", rb_id=11 + k, x=float(b["x"]), y=float(b["y"]), yaw=0.0) for k, b in enumerate(gt)]
    while not stop.is_set():
        write_json(os.path.join(a.out, "boxes.json"), dict(t=time.time(), boxes=boxes, robots={}))
        stop.wait(0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", action="append", required=True, metavar="ID:RIGID_BODY")
    ap.add_argument("--box", type=int, nargs="+", default=None,
                    help="only these OptiTrack rigid bodies are boxes (default: every tracked rigid body inside the "
                         "arena that is not a robot)")
    ap.add_argument("--out", default="data/live_ir/", help="folder of the live states (demo.live --live-dir)")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--optitrack-server", default="127.0.0.1")
    ap.add_argument("--optitrack-local", default="127.0.0.1")
    ap.add_argument("--no-turns", action="store_true",
                    help="every robot ranges when it likes (beams can meet on a box); default: one robot at a time")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-layout", default="L4")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    a, rest = ap.parse_known_args()
    os.makedirs(a.out, exist_ok=True)
    stop_file = os.path.join(a.out, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    for name in os.listdir(a.out):                     # states of an earlier session would look alive for a moment
        if name.endswith(".json") and name != "boxes.json":
            os.remove(os.path.join(a.out, name))
    robots = []
    for spec in a.robot:
        parts = spec.split(":")
        robots.append(dict(robot_id=parts[0], rb=int(parts[1]) if len(parts) > 1 and parts[1] else 0,
                           sim_pose=parts[2] if len(parts) > 2 else None))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if a.sim:
        truth = threading.Thread(target=sim_truth, args=(a, stop), daemon=True)
    else:
        truth = threading.Thread(target=truth_loop, args=(a, robots, stop), daemon=True)
    truth.start()
    procs = []
    for r in robots:
        if stop.is_set():
            break
        cmd = [sys.executable, "-u", "-m", "acquisition.ir_scan", "--robot-id", r["robot_id"],
               "--state-out", os.path.join(a.out, f"{r['robot_id']}.json"), "--stop-file", stop_file]
        if not a.no_turns:
            cmd += ["--turns"]
        if a.sim:
            x, y, yaw = (r["sim_pose"] or "-1.6,-0.4,0").split(",")
            cmd += ["--sim", "--sim-layout", a.sim_layout, "--ground-truth", a.ground_truth,
                    "--sim-pose", x, y, yaw, "--settle-s", "0"]
        else:
            cmd += ["--rb", str(r["rb"]), "--regions", a.regions, "--optitrack-server", a.optitrack_server,
                    "--optitrack-local", a.optitrack_local,
                    "--robot-rbs"] + [str(q["rb"]) for q in robots if q is not r]
            if a.box:
                cmd += ["--box-rb"] + [str(b) for b in a.box]
        cmd += rest
        print(f"{r['robot_id']}: " + ("simulated" if a.sim else f"rigid body {r['rb']}"), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        connected = threading.Event()
        threading.Thread(target=pump, args=(proc, r["robot_id"], connected), daemon=True).start()
        procs.append(proc)
        t0 = time.time()
        while not connected.is_set() and not stop.is_set() and time.time() - t0 < CONNECT_TIMEOUT_S:
            time.sleep(0.1)
        if not connected.is_set() and not stop.is_set():
            print(f"{r['robot_id']}: not running after {CONNECT_TIMEOUT_S:.0f} s, starting the next robot", flush=True)
        time.sleep(1.0)
    window = 20 if a.no_turns else max(20, int(2 * 1.5 * len(robots) * 4))
    if not a.no_turns:
        threading.Thread(target=coordinator, args=(a, robots, stop), daemon=True).start()
    print(f"all robots started: now run\n  python -m demo.live --live-dir {a.out} --sensor-calib "
          f"config/sensor_calib_ir.json --window-s {window} --stale-s {window} --no-images\n"
          f"  python -m tools.ir_view --live-dir {a.out}\n(Ctrl+C here to stop)", flush=True)
    while not stop.is_set() and any(p.poll() is None for p in procs):
        time.sleep(0.3)
    stop.set()
    open(stop_file, "w").close()                       # every robot finishes its reading and re-centres its gimbal
    t0 = time.time()
    while any(p.poll() is None for p in procs) and time.time() - t0 < 15:
        time.sleep(0.3)
    for p in procs:
        if p.poll() is None:
            p.terminate()
    print("stopped")


if __name__ == "__main__":
    main()
