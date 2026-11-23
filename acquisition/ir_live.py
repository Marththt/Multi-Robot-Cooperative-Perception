"""Live tests of the infrared (ToF) ranging of one robot, against the boxes OptiTrack tracks right now.

One robot, its IR sensors (robomaster_sdk/ir.py) and OptiTrack (the robot and the box rigid bodies). For the
ranging sensor (--ir-index, mounted as --ir-forward-m / --ir-right-m / --ir-yaw-deg) every reading is compared
with what it should read: the nearest box inside the cone of the sensor, at its centre distance minus the apothem
(the face). Four modes:

  watch   the robot stands still (move it, or the boxes, by hand): 5 lines per second with the 4 sensors, the
          expected box and distance, the error. Ctrl+C ends and prints the statistics.
  aim     the robot turns on the spot to aim the IR at every box in turn (OptiTrack positions, so the camera plays
          no part), reads --ir-samples samples, and moves to the next; --repeats rounds. Range error per box.
  sweep   around one box (--box RIGID_BODY, default the one nearest the heading) the robot turns from -span to
          +span in steps and reads the IR at each: the reading against the angle off the box tells how wide the
          beam really is and when it jumps to another box or the wall.
  camera  the full loop of acquisition/ir_ranging.py: YOLO finds the boxes in the camera frame, the robot aims
          the IR at each one, the box centre comes from the camera ray and the IR range; every result is compared
          with OptiTrack. Needs the YOLO packages (ultralytics, torch) in this environment and --camera-calib.

Every sample goes to <out>/<robot>_<mode>_<time>.csv.

    python -m acquisition.ir_live watch  --robot-id P0 --rb 8 --ir-index 0 --ir-forward-m 0.10
    python -m acquisition.ir_live aim    --robot-id P0 --rb 8 --ir-index 0 --ir-forward-m 0.10 --repeats 3
    python -m acquisition.ir_live sweep  --robot-id P0 --rb 8 --ir-index 0 --ir-forward-m 0.10 --span-deg 25
    python -m acquisition.ir_live camera --robot-id P0 --rb 8 --ir-index 0 --ir-forward-m 0.10 \\
        --camera-calib config/checkerboard_calib_P0.json
    # any mode without hardware: --sim --sim-layout L4 (boxes of config/ground_truth_positions.json)
"""

import argparse
import csv
import math
import os
import signal
import threading
import time

import numpy as np

from acquisition.ir_ranging import (FakeIR, SimRays, YoloRays, add_ir_arguments, aim_from_args, aim_yaw,
                                    camera_pitch, ir_origin, measure_targets, mount_from_args, turn_in_place, unit)
from vf.control.goto import FakeRobot, stop, wrap_pi
from yolo_gnn.camera import APOTHEM_M

STOP = threading.Event()


def expected(origin, axis, boxes, beam_deg):
    """What the sensor should read: the nearest box whose footprint enters the cone. Returns a dict (box id,
    face and centre distance, angle of the box off the axis), the fields None when no box is in the cone."""
    half = math.radians(beam_deg) / 2
    best = None
    for b in boxes:
        v = np.array([b["x"], b["y"]]) - origin
        d = float(np.hypot(*v))
        off = wrap_pi(math.atan2(v[1], v[0]) - axis)
        if abs(off) <= half + math.atan2(APOTHEM_M, max(d, 1e-3)) and (best is None or d < best[1]):
            best = (b["id"], d, off)
    if best is None:
        return dict(exp_box=None, exp_face_m=None, exp_centre_m=None, off_deg=None)
    return dict(exp_box=best[0], exp_face_m=best[1] - APOTHEM_M, exp_centre_m=best[1], off_deg=math.degrees(best[2]))


class Rig:
    """Robot, IR, OptiTrack (real), or FakeRobot, FakeIR, fixed boxes (--sim)."""

    def __init__(self, a):
        self.a, self.mount = a, mount_from_args(a)
        self.cleanup = []
        self.camera = None
        rng = np.random.default_rng(a.seed)
        if a.sim:
            import json
            with open(a.ground_truth) as f:
                gt = json.load(f)[a.sim_layout]
            self.fixed = [dict(id=11 + k, x=float(b["x"]), y=float(b["y"])) for k, b in enumerate(gt)]
            x, y, yaw = a.sim_pose
            self.fake = FakeRobot(x, y, math.radians(yaw), speedup=a.sim_speedup)
            self.chassis, self.body_pose = self.fake, self.fake.pose
            self.camera_pose = self.fake.pose
            xy = [[b["x"], b["y"]] for b in self.fixed]
            self.ir = FakeIR(xy, self.fake.pose, self.mount, rng)
            self.sim_rays = SimRays(xy, self.fake.pose, rng)
            self.ir.latest = lambda: (time.time(), [self.ir.measure(0, n=1)["m"] or 0.0] * 4)
            return
        from robomaster_sdk import Robot
        from robomaster_sdk.ir import IRSensors
        from vf.fusion.sensor_model import arena_from_regions
        from vf.localization.mocap_pose import MocapHub
        self.arena = arena_from_regions(a.regions)
        self.hub = MocapHub([a.rb] + list(a.box_rb), a.optitrack_server, a.optitrack_local)
        self.hub.wait_ready()
        self.cleanup.append(self.hub.close)
        off = math.radians(a.yaw_offset_deg)

        def body_pose():
            q = self.hub.get_pose(a.rb)
            return None if q is None else (float(q[0]), float(q[1]), wrap_pi(float(q[2]) + off))

        def camera_pose():
            q = body_pose()
            if q is None:
                return None
            c, s = math.cos(q[2]), math.sin(q[2])
            return (q[0] + a.camera_offset_forward_m * c, q[1] + a.camera_offset_forward_m * s, q[2])

        self.body_pose, self.camera_pose = body_pose, camera_pose
        self.robot = Robot(a.robot_id, sn=a.sn, ip=a.ip).connect()
        self.cleanup.insert(0, self.robot.disconnect)
        self.chassis = self.robot.chassis
        self.cleanup.insert(0, lambda: stop(self.chassis))
        self.ir = IRSensors.from_robot(self.robot, freq=a.ir_freq).start()
        self.cleanup.insert(1, self.ir.stop)
        time.sleep(0.5)
        print(f"first IR message from the SDK: {self.ir.first_raw}", flush=True)

    def boxes(self):
        if self.a.sim:
            return self.fixed
        x0, x1, y0, y1 = self.arena
        out = []
        for rb in self.a.box_rb:
            q = self.hub.get_pose(rb)
            if q is not None and x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                out.append(dict(id=rb, x=float(q[0]), y=float(q[1])))
        return out

    def start_camera(self):
        if self.a.sim:
            return self.sim_rays
        if self.a.camera_calib is None:
            raise SystemExit("camera mode needs --camera-calib")
        from robomaster_sdk.camera import Camera as CameraSDK
        from vf.perception.aruco_detector import load_calibration
        from yolo_gnn.segment import YoloSegmenter
        self.camera = CameraSDK(self.robot.camera).start(resolution=self.a.camera_resolution)
        self.cleanup.insert(1, self.camera.stop)
        K, D = load_calibration(self.a.camera_calib)
        return YoloRays(self.camera, YoloSegmenter(self.a.yolo_weights, device=self.a.yolo_device), K, D,
                        camera_pitch(self.a), self.camera_pose)

    def sample(self, n):
        """(body pose, IR origin, axis, measurement of the ranging sensor, expected)."""
        body = self.body_pose()
        r = self.ir.measure(self.mount.index, n=n)
        if body is None:
            return None, None, None, r, expected(np.zeros(2), 0.0, [], self.mount.beam_deg)
        o, axis = ir_origin(body, self.mount)
        return body, o, axis, r, expected(o, axis, self.boxes(), self.mount.beam_deg)

    def turn_to(self, yaw):
        return turn_in_place(self.body_pose, self.chassis, yaw, math.radians(self.a.aim_tol_deg),
                             self.a.aim_timeout_s, STOP.is_set)

    def close(self):
        for fn in self.cleanup:
            try:
                fn()
            except Exception as e:
                print(f"cleanup: {e}")


class Log:
    def __init__(self, a, mode):
        os.makedirs(a.out, exist_ok=True)
        self.path = os.path.join(a.out, f"{a.robot_id}_{mode}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self.rows = []

    def add(self, **row):
        row = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
        self.rows.append(row)
        return row

    def write(self):
        if not self.rows:
            return
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        print(f"written {self.path} ({len(self.rows)} rows)")


def row_of(body, o, axis, r, e, **extra):
    err = None if (r["m"] is None or e["exp_face_m"] is None) else r["m"] - e["exp_face_m"]
    return dict(t=time.time(), body_x=None if body is None else body[0], body_y=None if body is None else body[1],
                body_yaw_deg=None if body is None else math.degrees(body[2]),
                ir_x=None if o is None else float(o[0]), ir_y=None if o is None else float(o[1]),
                ir_axis_deg=None if axis is None else math.degrees(axis), ir_m=r["m"], ir_sd_m=r["sd_m"],
                ir_n=r["n"], ir_n_total=r["n_total"], err_m=err, **e, **extra)


def stats(rows, label):
    e = np.array([r["err_m"] for r in rows if r.get("err_m") is not None], float)
    none = sum(1 for r in rows if r.get("ir_m") is None)
    if len(e):
        print(f"{label}: {len(e)} reading(s) with a box in the cone: error median {np.median(e) * 100:+.1f} cm, "
              f"median |error| {np.median(np.abs(e)) * 100:.1f} cm, sd {np.std(e) * 100:.1f} cm, "
              f"max |error| {np.max(np.abs(e)) * 100:.1f} cm; {none} reading(s) without a valid range")
    else:
        print(f"{label}: no reading with a box in the cone; {none} reading(s) without a valid range")


def fmt(v, unit="", nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}{unit}"


# ====================================================================
# Modes
# ====================================================================

def watch(rig, a, log):
    print("all 4 sensors [m] | ranging sensor: reading, expected box and face distance, error   (Ctrl+C ends)")
    while not STOP.is_set():
        t0 = time.time()
        latest = rig.ir.latest()
        body, o, axis, r, e = rig.sample(n=1)
        four = latest[1] if latest else [None] * 4
        row = log.add(**row_of(body, o, axis, r, e, all_m=" ".join(fmt(v) for v in four)))
        print(" ".join(f"{fmt(v):>6}" for v in four) + f" | IR {fmt(r['m'])}  expected box "
              f"{e['exp_box'] or '-'} {fmt(e['exp_face_m'])} (off {fmt(e['off_deg'], ' deg', 1)})  "
              f"error {fmt(None if row['err_m'] is None else row['err_m'] * 100, ' cm', 1)}", flush=True)
        STOP.wait(max(0.0, 0.2 - (time.time() - t0)))
    stats(log.rows, "watch")


def aim(rig, a, log):
    start = rig.body_pose()
    for rep in range(a.repeats):
        boxes = [b for b in rig.boxes() if a.boxes is None or b["id"] in a.boxes]
        body = rig.body_pose()
        if body is None or not boxes:
            print("no pose or no box")
            return
        boxes.sort(key=lambda b: wrap_pi(math.atan2(b["y"] - body[1], b["x"] - body[0]) - body[2]))
        print(f"--- round {rep + 1}/{a.repeats}: boxes {[b['id'] for b in boxes]}")
        for b in boxes:
            if STOP.is_set():
                return
            ok = rig.turn_to(aim_yaw(rig.body_pose(), (b["x"], b["y"]), rig.mount))
            time.sleep(a.settle_s)
            body, o, axis, r, e = rig.sample(a.ir_samples)
            off = None if o is None else math.degrees(wrap_pi(math.atan2(b["y"] - o[1], b["x"] - o[0]) - axis))
            true_face = None if o is None else float(np.hypot(b["x"] - o[0], b["y"] - o[1])) - APOTHEM_M
            err = None if (r["m"] is None or true_face is None) else r["m"] - true_face
            log.add(**row_of(body, o, axis, r, e, round=rep, target=b["id"], target_face_m=true_face,
                             target_off_deg=off, target_err_m=err, turn_ok=ok))
            print(f"  box {b['id']}: IR {fmt(r['m'])} ({r['n']}/{r['n_total']} valid, sd "
                  f"{fmt(None if r['sd_m'] is None else r['sd_m'] * 1000, ' mm', 1)}), true face {fmt(true_face)}, "
                  f"error {fmt(None if err is None else err * 100, ' cm', 1)}, aim off {fmt(off, ' deg', 1)}"
                  + ("" if e["exp_box"] in (None, b["id"]) else f"  <- nearer box {e['exp_box']} in the cone"))
    tgt = [dict(err_m=r["target_err_m"], ir_m=r["ir_m"]) for r in log.rows]
    stats(tgt, "aim")
    if start is not None:
        rig.turn_to(start[2])


def sweep(rig, a, log):
    body = rig.body_pose()
    boxes = rig.boxes()
    if body is None or not boxes:
        print("no pose or no box")
        return
    if a.box is not None:
        b = next((x for x in boxes if x["id"] == a.box), None)
        if b is None:
            raise SystemExit(f"box {a.box} not tracked")
    else:
        b = min(boxes, key=lambda x: abs(wrap_pi(math.atan2(x["y"] - body[1], x["x"] - body[0]) - body[2])))
    centre = aim_yaw(body, (b["x"], b["y"]), rig.mount)
    steps = np.arange(-a.span_deg, a.span_deg + 1e-6, a.step_deg)
    print(f"sweep around box {b['id']}: {len(steps)} steps of {a.step_deg} deg")
    for s in steps:
        if STOP.is_set():
            break
        rig.turn_to(wrap_pi(centre + math.radians(s)))
        time.sleep(a.settle_s)
        body, o, axis, r, e = rig.sample(a.ir_samples)
        off = None if o is None else math.degrees(wrap_pi(math.atan2(b["y"] - o[1], b["x"] - o[0]) - axis))
        face = None if o is None else float(np.hypot(b["x"] - o[0], b["y"] - o[1])) - APOTHEM_M
        log.add(**row_of(body, o, axis, r, e, target=b["id"], step_deg=float(s), target_off_deg=off,
                         target_face_m=face, on_target=None if (r["m"] is None or face is None)
                         else abs(r["m"] - face) <= a.on_target_m and e["exp_box"] in (None, b["id"])))
        print(f"  axis {fmt(off, ' deg', 1):>9} off box {b['id']}: IR {fmt(r['m'])}  (box face {fmt(face)}"
              + (f", expected nearest box {e['exp_box']}" if e["exp_box"] not in (None, b["id"]) else "") + ")",
              flush=True)
    on = [r["target_off_deg"] for r in log.rows if r.get("on_target")]
    if on:
        print(f"the IR reads box {b['id']} (within {a.on_target_m * 100:.0f} cm of its face) from "
              f"{min(on):+.1f} to {max(on):+.1f} deg off its centre: {max(on) - min(on):.1f} deg in all")
    else:
        print(f"the IR never read box {b['id']}")
    rig.turn_to(centre)


def camera(rig, a, log):
    rays = rig.start_camera()
    for rep in range(a.repeats):
        if STOP.is_set():
            break
        print(f"--- round {rep + 1}/{a.repeats}")
        got = measure_targets(rays.rays, rig.ir, rig.body_pose, rig.chassis, rig.mount, aim_from_args(a),
                              cancel=STOP.is_set)
        boxes = rig.boxes()
        xy = np.array([[b["x"], b["y"]] for b in boxes], float).reshape(-1, 2)
        for g in got:
            o, u = np.asarray(g["camera_pose"][:2]), unit(g["ray_angle_rad"])
            k, perr, merr = None, None, None
            if len(xy):
                v = xy - o
                along = v @ u
                perp = np.abs(v[:, 0] * u[1] - v[:, 1] * u[0])
                k = int(np.argmin(np.where(along > 0, perp + 0.2 * np.abs(along - g["mono_m"]), np.inf)))
                if g["position"] is not None:
                    perr = float(np.hypot(*(xy[k] - np.asarray(g["position"]))))
                merr = float(np.hypot(*(xy[k] - (o + g["mono_m"] * u))))
            true_face = (None if (k is None or g["ir_origin"] is None)
                         else float(np.hypot(*(xy[k] - np.asarray(g["ir_origin"])))) - APOTHEM_M)
            err = None if (g["ir_m"] is None or true_face is None) else g["ir_m"] - true_face
            log.add(round=rep, target=g["target"], true_box=None if k is None else boxes[k]["id"], ir_m=g["ir_m"],
                    true_face_m=true_face, err_m=err, centre_err_m=perr, mono_centre_err_m=merr,
                    aim_residual_deg=g["aim_residual_deg"], aim_lost=g["aim_lost"], consistent=g["consistent"],
                    ray_angle_deg=math.degrees(g["ray_angle_rad"]), mono_m=g["mono_m"])
            print(f"  YOLO box {g['target']} -> true box {None if k is None else boxes[k]['id']}: IR error "
                  f"{fmt(None if err is None else err * 100, ' cm', 1)}, centre error "
                  f"{fmt(None if perr is None else perr * 100, ' cm', 1)} (camera alone "
                  f"{fmt(None if merr is None else merr * 100, ' cm', 1)})")
    stats(log.rows, "camera (IR range)")
    c = [r["centre_err_m"] for r in log.rows if r.get("centre_err_m") is not None]
    m = [r["mono_centre_err_m"] for r in log.rows if r.get("mono_centre_err_m") is not None]
    if c:
        print(f"box centre error: median {np.median(c) * 100:.1f} cm with the IR, "
              f"{np.median(m) * 100:.1f} cm with the camera alone")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["watch", "aim", "sweep", "camera"])
    ap.add_argument("--robot-id", required=True)
    ap.add_argument("--sn", default=None)
    ap.add_argument("--ip", default=None)
    ap.add_argument("--rb", type=int, help="OptiTrack rigid body of the robot")
    ap.add_argument("--box-rb", type=int, nargs="+", default=[11, 12, 13, 14, 15, 16, 17],
                    help="OptiTrack rigid bodies of the boxes (those inside the arena are used)")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--optitrack-server", default="127.0.0.1")
    ap.add_argument("--optitrack-local", default="127.0.0.1")
    ap.add_argument("--yaw-offset-deg", type=float, default=0.0)
    ap.add_argument("--camera-offset-forward-m", type=float, default=0.12)
    ap.add_argument("--camera-calib", default=None)
    ap.add_argument("--camera-resolution", choices=["360", "540", "720"], default="720")
    ap.add_argument("--repeats", type=int, default=1, help="aim / camera: rounds")
    ap.add_argument("--boxes", type=int, nargs="+", default=None, help="aim: only these box rigid bodies")
    ap.add_argument("--box", type=int, default=None, help="sweep: the box rigid body (default: nearest the heading)")
    ap.add_argument("--span-deg", type=float, default=25.0, help="sweep: from -span to +span around the box")
    ap.add_argument("--step-deg", type=float, default=2.0, help="sweep: step")
    ap.add_argument("--on-target-m", type=float, default=0.05, help="sweep: reading within this of the box face")
    ap.add_argument("--settle-s", type=float, default=0.3, help="wait after turning, before reading")
    ap.add_argument("--out", default="data/ir_live/")
    ap.add_argument("--sim", action="store_true", help="no hardware: fake robot and IR, boxes of --sim-layout")
    ap.add_argument("--sim-layout", default="L4")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sim-pose", type=float, nargs=3, default=[-1.6, -0.6, 0.0], metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--sim-speedup", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    add_ir_arguments(ap)
    a = ap.parse_args()
    if not a.sim and a.rb is None:
        ap.error("--rb (OptiTrack rigid body of the robot) is required, or --sim")
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    rig = Rig(a)
    log = Log(a, a.mode)
    try:
        dict(watch=watch, aim=aim, sweep=sweep, camera=camera)[a.mode](rig, a, log)
    finally:
        stop(rig.chassis)
        log.write()
        rig.close()


if __name__ == "__main__":
    main()
