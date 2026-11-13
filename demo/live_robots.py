"""Robot side of the continuous live demo: every robot streams what it sees, and OptiTrack gives the true boxes.

Place the robots by hand, or send them from the page (below), and move the boxes at any time: each robot runs
acquisition/static_trial.py --live, which shows its camera with the detected markers and writes, a few times per
second, its live state (camera pose from OptiTrack, detections) and its last frame in --out. This script
starts one such process per robot, one after the other (the robot SDK finds a robot through a port that only one
process at a time can use), and meanwhile writes, 30 times per second, the positions of the boxes that OptiTrack
sees inside the arena and the camera poses of the robots in <out>/boxes.json: the ground truth of the page and
the poses with which the page moves the robots smoothly. demo/live.py reads these files and runs the methods on
them continuously.

Run it in the usual environment of the captures (robot SDK, natnet):

    python -m demo.live_robots --robot P0:8 --robot P1:7 --robot P3:10:config/checkerboard_calib_P1.json \\
        --robot P5:6:config/checkerboard_calib_P1.json

Every robot also drives where the page of the demo sends it (click a robot, then a point of the map: see
acquisition/live_drive.py), on a path clear of the boxes and of the other robots; --no-drive turns this off.

--robot is ID:RIGID_BODY[:CAMERA_CALIB], as in launch_all.py. The camera windows of the robots are not shown
(the page shows the frames); --windows shows them, and 'q' in a window stops that robot only. Ctrl+C here stops
every robot process.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

from acquisition.launch_all import parse_robot

CONNECTED_MSG = "Waiting for the first frame"        # printed by static_trial.py --live once the robot is connected
CONNECT_TIMEOUT_S = 60.0
MOCAP_HZ = 30.0                                       # boxes and robot poses written to boxes.json


def pump(proc, tag, connected):
    """Prints the output of one robot process (bytes, never stopping on a line it cannot decode: if this thread
    died, the pipe would fill up and the robot process would freeze at its next print)."""
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        try:
            if line.strip():
                print(f"{tag:>4} | {line}", flush=True)
        except Exception:
            pass
        if CONNECTED_MSG in line:
            connected.set()
    connected.set()                                  # process ended: do not keep the others waiting


def write_json(path, data):
    """Atomic write, so that a reader never opens a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                      # Windows: the file is being scanned or read right now
            time.sleep(0.02 * (attempt + 1))


def truth_loop(a, robots, stop):
    """Writes <out>/boxes.json MOCAP_HZ times per second: the boxes OptiTrack tracks inside the arena and the
    camera pose of every robot (rigid body + the camera offset, as static_trial.py computes it)."""
    from vf.fusion.sensor_model import arena_from_regions
    from vf.localization.mocap_pose import MocapHub
    x0, x1, y0, y1 = arena_from_regions(a.regions)
    hub = MocapHub(list(a.box) + [r["rb"] for r in robots], a.optitrack_server, a.optitrack_local)
    path = os.path.join(a.out, "boxes.json")
    off = a.camera_offset_forward_m
    try:
        while not stop.is_set():
            t0 = time.time()
            boxes = []
            for rb in a.box:
                q = hub.get_pose(rb)
                if q is not None and x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                    boxes.append(dict(class_name="box", rb_id=rb, x=round(float(q[0]), 4), y=round(float(q[1]), 4),
                                      yaw=round(float(q[2]), 4)))
            poses = {}
            for r in robots:
                q = hub.get_pose(r["rb"])
                if q is not None:
                    x, y, yaw = (float(v) for v in q[:3])
                    poses[r["robot_id"]] = [round(x + off * math.cos(yaw), 4), round(y + off * math.sin(yaw), 4),
                                            round(yaw, 4)]
            write_json(path, dict(t=time.time(), boxes=boxes, robots=poses))
            stop.wait(max(0.0, 1.0 / MOCAP_HZ - (time.time() - t0)))
    finally:
        hub.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", action="append", required=True, metavar="ID:RIGID_BODY[:CAMERA_CALIB]")
    ap.add_argument("--box", type=int, nargs="+", default=[11, 12, 13, 14, 15, 16, 17],
                    help="OptiTrack rigid bodies of the boxes; those inside the arena are the true boxes")
    ap.add_argument("--out", default="data/live/", help="folder of the live states (demo.live --live-dir)")
    ap.add_argument("--frame-every-s", type=float, default=0.25, help="seconds between two frames written per robot")
    ap.add_argument("--windows", action="store_true", help="show the camera window of every robot")
    ap.add_argument("--no-drive", action="store_true",
                    help="the robots ignore the move commands of the page (they are then moved only by hand)")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--aruco-marker-size", type=float, default=0.077)
    ap.add_argument("--camera-offset-forward-m", type=float, default=0.12,
                    help="camera ahead of the rigid body, as in the captures (the lens offset is corrected later)")
    ap.add_argument("--optitrack-server", default="127.0.0.1")
    ap.add_argument("--optitrack-local", default="127.0.0.1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    robots = [parse_robot(s) for s in a.robot]
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    truth = threading.Thread(target=truth_loop, args=(a, robots, stop), daemon=True)
    truth.start()
    procs = []
    for r in robots:
        if stop.is_set():
            break
        cmd = [sys.executable, "-u", "-m", "acquisition.static_trial", "--robot-id", r["robot_id"], "--live",
               "--state-out", os.path.join(a.out, f"{r['robot_id']}.json"),
               "--frame-out", os.path.join(a.out, f"{r['robot_id']}.jpg"), "--frame-every-s", str(a.frame_every_s),
               "--camera-resolution", "720", "--regions", a.regions,
               "--localization", "optitrack", "--optitrack-rb-id", str(r["rb"]),
               "--optitrack-server", a.optitrack_server, "--optitrack-local", a.optitrack_local,
               "--detector", "aruco", "--aruco-marker-size", str(a.aruco_marker_size),
               "--camera-offset-forward-m", str(a.camera_offset_forward_m)]
        if r["calib"]:
            cmd += ["--camera-calib", r["calib"]]
        if not a.windows:
            cmd += ["--no-window"]
        if not a.no_drive:
            cmd += ["--commands", os.path.join(a.out, f"{r['robot_id']}.cmd.json")]
        print(f"{r['robot_id']}: rigid body {r['rb']}, calibration {r['calib'] or 'NONE'}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        connected = threading.Event()
        threading.Thread(target=pump, args=(proc, r["robot_id"], connected), daemon=True).start()
        procs.append(proc)
        t0 = time.time()
        while not connected.is_set() and not stop.is_set() and time.time() - t0 < CONNECT_TIMEOUT_S:
            time.sleep(0.1)
        if not connected.is_set() and not stop.is_set():
            print(f"{r['robot_id']}: not connected after {CONNECT_TIMEOUT_S:.0f} s, starting the next robot", flush=True)
        time.sleep(1.0)                              # let the SDK release the discovery port
    print("all robots started: now run  python -m demo.live --live-dir " + a.out + "  (Ctrl+C here to stop)",
          flush=True)
    while not stop.is_set() and any(p.poll() is None for p in procs):
        time.sleep(0.3)
    stop.set()
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    print("stopped")


if __name__ == "__main__":
    main()
