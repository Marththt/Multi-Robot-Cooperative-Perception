"""Live tests of the infrared (ToF) sensor on the gimbal of one robot, against the boxes OptiTrack tracks now.

The camera and the IR sit on the gimbal; the gimbal points them, the chassis never moves (the robot is set to the
SDK "free" mode while the script runs, and back to "chassis_lead" at the end). Every IR reading is compared with
what it should read (acquisition/ir_ranging.expected): the nearest box whose face the beam hits, from the robot
pose (OptiTrack), the gimbal angles (read back from the gimbal) and the mount (--ir-arm-m, --ir-height-m, ...).
The IR reads a slant distance; with the gimbal pitched down, the horizontal distance is d cos(pitch) and the
beam meets the face at height h - d sin(pitch). Four modes:

  watch   the gimbal stays where it is (or at --yaw-deg / --pitch-deg): 5 lines per second with the reading, the
          expected box and distance, the error. Move the boxes by hand. Ctrl+C ends with the statistics.
  aim     the gimbal points the IR at every box in turn (OptiTrack positions: the camera plays no part), mid-face
          (--aim-height-m), reads --ir-samples samples; --repeats rounds. Error per box, slant and horizontal.
  sweep   around one box (--box RIGID_BODY, default the one nearest the gimbal direction): --axis yaw turns the
          gimbal from -span to +span (how wide the beam is, when it jumps to another box); --axis pitch tilts it
          from --pitch-from-deg to --pitch-to-deg (where the beam leaves the top of the box, where it meets the
          floor).
  camera  the full loop of acquisition/ir_ranging.py: YOLO finds the boxes, the gimbal points the IR at each one,
          the box centre comes from the camera ray and the IR range; compared with OptiTrack. Needs the YOLO
          packages (ultralytics, torch) in this environment and --camera-calib.

Every reading goes to <out>/<robot>_<mode>_<time>.csv.

    python -m acquisition.ir_live watch --robot-id P0 --rb 8
    python -m acquisition.ir_live aim   --robot-id P0 --rb 8 --repeats 3
    python -m acquisition.ir_live sweep --robot-id P0 --rb 8 --axis yaw --span-deg 20
    python -m acquisition.ir_live sweep --robot-id P0 --rb 8 --axis pitch
    python -m acquisition.ir_live camera --robot-id P0 --rb 8 --camera-calib config/checkerboard_calib_P0.json
    # any mode without hardware: --sim --sim-layout L4 (boxes of config/ground_truth_positions.json)

Mount defaults: IR 10 cm ahead of the gimbal axis, 30 cm high; gimbal axis above the rigid body origin
(--gimbal-forward-m 0). Check the gimbal directions first: python -m tools.gimbal_check --robot-id P0.
"""

import argparse
import csv
import json
import math
import os
import signal
import threading
import time

import numpy as np

from acquisition.ir_ranging import (FakeGimbal, FakeIR, SimRays, YoloRays, add_ir_arguments, aim_angles, aim_from_args,
                                    expected, ir_pose, measure_targets, mount_from_args, slant_to_floor, unit)
from vf.control.goto import wrap_pi
from yolo_gnn.camera import APOTHEM_M, BOX_HEIGHT_M

STOP = threading.Event()

# gimbal aiming offset per robot [deg], measured with `sweep --axis yaw` (centre of the plateau of a box against
# where OptiTrack puts it): the gimbal zero is not the forward axis of the rigid body. G1, 2026-11-23: boxes 16, 13,
# 12 gave +14.0, +13.9, +12.1.
YAW_OFFSET_DEG = {"G1": 13.3}

# range correction of the IR per robot: IR = K x true + Q [m], fitted with `aim` against OptiTrack; the readings
# are corrected as true = (IR - Q) / K. G1, 2026-11-23: boxes at 0.94-1.96 m from one position (to refine).
IR_FIT = {"G1": (1.0728, -0.010)}


class CorrectedIR:
    """The IR with its range correction: measure() gives the corrected median in "m" and the raw one in "m_raw"
    (the samples stay raw). latest() / valid() / stop() pass through."""

    def __init__(self, ir, k=1.0, q=0.0):
        self.ir, self.k, self.q = ir, k, q

    def measure(self, index, n=10, timeout_s=3.0):
        r = dict(self.ir.measure(index, n=n, timeout_s=timeout_s))
        r["m_raw"] = r["m"]
        if r["m"] is not None:
            r["m"] = (r["m"] - self.q) / self.k
            r["sd_m"] = None if r["sd_m"] is None else r["sd_m"] / self.k
        return r

    def __getattr__(self, name):
        return getattr(self.ir, name)


class Rig:
    """Robot (only its gimbal and IR are used), OptiTrack (robot and boxes); or fakes with fixed boxes (--sim)."""

    def __init__(self, a):
        self.a, self.mount = a, mount_from_args(a)
        self.cleanup = []
        self.camera = None
        rng = np.random.default_rng(a.seed)
        if a.sim:
            with open(a.ground_truth) as f:
                gt = json.load(f)[a.sim_layout]
            self.fixed = [dict(id=11 + k, x=float(b["x"]), y=float(b["y"])) for k, b in enumerate(gt)]
            x, y, yaw = a.sim_pose
            true = (x, y, math.radians(yaw))                          # where the fakes really point from
            seen = (x, y, wrap_pi(math.radians(yaw - a.sim_yaw_error_deg + (a.yaw_offset_deg or 0.0))))
            self.body_pose = lambda: seen                             # what the script believes
            self.gimbal = FakeGimbal()
            xy = [[b["x"], b["y"]] for b in self.fixed]
            self.ir = CorrectedIR(FakeIR(xy, lambda: true, self.gimbal, self.mount, rng), *a.ir_fit)
            self.sim_rays = SimRays(xy, lambda: true, self.gimbal, self.mount, rng)
            return
        from robomaster_sdk import Robot
        from robomaster_sdk.gimbal import PITCH_SIGN, YAW_SIGN, Gimbal
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

        self.body_pose = body_pose
        self.robot = Robot(a.robot_id, sn=a.sn, ip=a.ip).connect()
        self.cleanup.insert(0, self.robot.disconnect)
        self.gimbal = Gimbal.from_robot(self.robot, yaw_sign=a.gimbal_yaw_sign or YAW_SIGN,
                                        pitch_sign=a.gimbal_pitch_sign or PITCH_SIGN).start()
        self.cleanup.insert(0, self.gimbal.close)
        raw_ir = IRSensors.from_robot(self.robot, freq=a.ir_freq).start()
        self.cleanup.insert(0, raw_ir.stop)
        self.ir = CorrectedIR(raw_ir, *a.ir_fit)
        time.sleep(0.5)
        print(f"first IR message from the SDK: {self.ir.first_raw}", flush=True)
        print(f"first gimbal message from the SDK: {self.gimbal.first_raw}", flush=True)

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
        self.cleanup.insert(0, self.camera.stop)
        K, D = load_calibration(self.a.camera_calib)
        return YoloRays(self.camera, YoloSegmenter(self.a.yolo_weights, device=self.a.yolo_device), K, D,
                        self.body_pose, self.gimbal, self.mount)

    def snap(self, name, text=""):
        """--snap: saves the camera frame (the camera is on the gimbal) with a cross at the centre of the image and
        `text`, in <out>/snaps_<time>/<name>.jpg: where the gimbal really points."""
        if not self.a.snap or self.a.sim:
            return
        import cv2
        if self.camera is None:
            from robomaster_sdk.camera import Camera as CameraSDK
            self.camera = CameraSDK(self.robot.camera).start(resolution=self.a.camera_resolution)
            self.cleanup.insert(0, self.camera.stop)
            self.snap_dir = os.path.join(self.a.out, f"snaps_{time.strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(self.snap_dir, exist_ok=True)
            time.sleep(1.0)
        time.sleep(0.2)
        img, _ = self.camera.read()
        if img is None:
            print(f"  (no camera frame for {name})")
            return
        img = img.copy()
        h, w = img.shape[:2]
        cv2.line(img, (w // 2 - 40, h // 2), (w // 2 + 40, h // 2), (0, 0, 255), 2)
        cv2.line(img, (w // 2, h // 2 - 40), (w // 2, h // 2 + 40), (0, 0, 255), 2)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(self.snap_dir, f"{name}.jpg"), img)

    def point(self, yaw, pitch):
        if not STOP.is_set():
            self.gimbal.point(yaw, pitch)
            time.sleep(self.a.settle_s)

    def sample(self, n):
        """Reading of the IR with the pose of its beam and what it should read."""
        body, ang = self.body_pose(), self.gimbal.angles()
        r = self.ir.measure(self.mount.ir_index, n=n)
        raw = r.get("raw_mm") or [None] * 4
        row = dict(t=time.time(), ir_m=r["m"], ir_raw_m=r.get("m_raw"), ir_sd_m=r["sd_m"], ir_n=r["n"],
                   ir_n_total=r["n_total"],
                   **{f"raw{k}_mm": raw[k] for k in range(4)})
        if body is None or ang is None:
            return row
        o, head, pitch = ir_pose(body, ang[0], ang[1], self.mount)
        e = expected(o, head, pitch, self.mount.ir_height_m, self.boxes(), self.mount.beam_deg)
        horiz = hit = None
        if r["m"] is not None:
            horiz, hit = slant_to_floor(r["m"], pitch, self.mount.ir_height_m)
        row.update(body_x=body[0], body_y=body[1], body_yaw_deg=math.degrees(body[2]),
                   gimbal_yaw_deg=math.degrees(ang[0]), gimbal_pitch_deg=math.degrees(ang[1]),
                   ir_x=float(o[0]), ir_y=float(o[1]), ir_heading_deg=math.degrees(head),
                   ir_pitch_deg=math.degrees(pitch), ir_horiz_m=horiz, ir_hit_height_m=hit, **e,
                   err_slant_m=None if (r["m"] is None or e["exp_slant_m"] is None) else r["m"] - e["exp_slant_m"])
        return row

    def target_truth(self, row, b):
        """Horizontal / slant distance of box b's face from the beam of `row`, and the box off the axis."""
        if row.get("ir_x") is None:
            return dict(target=b["id"])
        o = np.array([row["ir_x"], row["ir_y"]])
        horiz = float(np.hypot(b["x"] - o[0], b["y"] - o[1])) - APOTHEM_M
        pitch = math.radians(row["ir_pitch_deg"])
        off = math.degrees(wrap_pi(math.atan2(b["y"] - o[1], b["x"] - o[0]) - math.radians(row["ir_heading_deg"])))
        out = dict(target=b["id"], target_horiz_m=horiz, target_slant_m=horiz / math.cos(pitch),
                   target_hit_height_m=self.mount.ir_height_m - horiz * math.tan(pitch), target_off_deg=off)
        if row["ir_m"] is not None:
            out.update(target_err_slant_m=row["ir_m"] - out["target_slant_m"],
                       target_err_horiz_m=row["ir_horiz_m"] - horiz)
        return out

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

    def add(self, row):
        self.rows.append({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
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


def fmt(v, unit="", nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}{unit}"


def raw_txt(r):
    """The 4 slots of the SDK message [mm], shown when the ranging slot gives nothing valid."""
    if r.get("ir_m") is not None:
        return ""
    return "  [SDK slots, mm: " + " ".join("-" if r.get(f"raw{k}_mm") is None else f"{r[f'raw{k}_mm']:.0f}"
                                         for k in range(4)) + "]"


def cm(v):
    return fmt(None if v is None else v * 100, " cm", 1)


def stats(errs, none, label):
    e = np.array([x for x in errs if x is not None], float)
    if len(e):
        print(f"{label}: {len(e)} reading(s): error median {np.median(e) * 100:+.1f} cm, median |error| "
              f"{np.median(np.abs(e)) * 100:.1f} cm, sd {np.std(e) * 100:.1f} cm, max |error| "
              f"{np.max(np.abs(e)) * 100:.1f} cm; {none} reading(s) without a valid range")
    else:
        print(f"{label}: no reading to compare; {none} reading(s) without a valid range")


def plateaus(angles, readings, flat_m=0.03, min_steps=3):
    """Runs of consecutive steps whose readings stay within flat_m of their median: the surfaces the beam rests on
    (the steps across an edge mix two surfaces and give values in between). [(first angle, last angle, median)]."""
    out, i, n = [], 0, len(angles)
    while i < n:
        if readings[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and readings[j + 1] is not None and \
                max(readings[i:j + 2]) - min(readings[i:j + 2]) <= 2 * flat_m:
            j += 1
        if j - i + 1 >= min_steps:
            out.append((angles[i], angles[j], float(np.median(readings[i:j + 1]))))
        i = j + 1
    return out


def target_plateau(rows, true_m, rel=0.25):
    """The plateau of the target box: the flat run whose reading is closest to the true face distance (within rel)."""
    rows = sorted(rows, key=lambda r: r["gimbal_yaw_deg"])
    runs = plateaus([r["gimbal_yaw_deg"] for r in rows], [r["ir_m"] for r in rows])
    runs = [p for p in runs if abs(p[2] - true_m) <= rel * true_m]
    return min(runs, key=lambda p: abs(p[2] - true_m)) if runs else None


def nearest_box(rig, prefer=None):
    body, ang, boxes = rig.body_pose(), rig.gimbal.angles(), rig.boxes()
    if body is None or ang is None or not boxes:
        return None
    if prefer is not None:
        b = next((x for x in boxes if x["id"] == prefer), None)
        if b is None:
            raise SystemExit(f"box {prefer} not tracked")
        return b
    head = body[2] + ang[0]
    return min(boxes, key=lambda x: abs(wrap_pi(math.atan2(x["y"] - body[1], x["x"] - body[0]) - head)))


# ====================================================================
# Modes
# ====================================================================

def watch(rig, a, log):
    if a.yaw_deg is not None or a.pitch_deg is not None:
        y, p = rig.gimbal.angles() or (0.0, 0.0)
        rig.point(math.radians(a.yaw_deg) if a.yaw_deg is not None else y,
                  math.radians(a.pitch_deg) if a.pitch_deg is not None else p)
    print("gimbal yaw/pitch | IR slant, flat, hit height | expected box, slant | error   (Ctrl+C ends)")
    while not STOP.is_set():
        t0 = time.time()
        r = log.add(rig.sample(n=1))
        print(f"{fmt(r.get('gimbal_yaw_deg'), '', 1):>6}/{fmt(r.get('gimbal_pitch_deg'), '', 1):<5} deg | "
              f"IR {fmt(r['ir_m'])} flat {fmt(r.get('ir_horiz_m'))} h {fmt(r.get('ir_hit_height_m'), '', 2)} | "
              f"box {r.get('exp_box') or '-'} {fmt(r.get('exp_slant_m'))}"
              + ("" if r.get("exp_box") else f" (floor at {fmt(r.get('exp_floor_m'))})")
              + f" | {cm(r.get('err_slant_m'))}" + raw_txt(r), flush=True)
        STOP.wait(max(0.0, 0.2 - (time.time() - t0)))
    stats([r.get("err_slant_m") for r in log.rows], sum(r["ir_m"] is None for r in log.rows), "watch")


def aim(rig, a, log):
    start = rig.gimbal.angles()
    for rep in range(a.repeats):
        body, boxes = rig.body_pose(), [b for b in rig.boxes() if a.boxes is None or b["id"] in a.boxes]
        if body is None or not boxes:
            print("no robot pose or no box")
            return
        boxes.sort(key=lambda b: wrap_pi(math.atan2(b["y"] - body[1], b["x"] - body[0]) - body[2]))
        print(f"--- round {rep + 1}/{a.repeats}: boxes {[b['id'] for b in boxes]}")
        for b in boxes:
            if STOP.is_set():
                break
            rig.point(*aim_angles(body, (b["x"], b["y"]), rig.mount))
            r = rig.sample(a.ir_samples)
            r.update(rig.target_truth(r, b), round=rep)
            log.add(r)
            rig.snap(f"aim_r{rep}_box{b['id']}", f"box {b['id']}: IR {fmt(r['ir_m'])} m, true {fmt(r.get('target_slant_m'))} m, "
                     f"gimbal {fmt(r.get('gimbal_yaw_deg'), '', 1)}/{fmt(r.get('gimbal_pitch_deg'), '', 1)} deg")
            print(f"  box {b['id']}: gimbal {fmt(r.get('gimbal_yaw_deg'), '', 1)}/{fmt(r.get('gimbal_pitch_deg'), '', 1)}"
                  f" deg, IR {fmt(r['ir_m'])} ({r['ir_n']}/{r['ir_n_total']}, sd {cm(r['ir_sd_m'])}), true "
                  f"{fmt(r.get('target_slant_m'))}: error {cm(r.get('target_err_slant_m'))} slant, "
                  f"{cm(r.get('target_err_horiz_m'))} flat, hit at {fmt(r.get('target_hit_height_m'), ' m', 2)}"
                  + ("" if r.get("exp_box") in (None, b["id"]) else f"  <- box {r['exp_box']} nearer in the beam") + raw_txt(r))
    stats([r.get("target_err_slant_m") for r in log.rows], sum(r["ir_m"] is None for r in log.rows), "aim")
    pts = [(r["target_slant_m"], r.get("ir_raw_m") if r.get("ir_raw_m") is not None else r["ir_m"]) for r in log.rows
           if r.get("ir_m") is not None and r.get("target_slant_m") is not None]
    if len({round(t, 1) for t, _ in pts}) >= 2:
        t, m = np.array(pts).T
        k, q = np.polyfit(t, m, 1)
        scale = float(np.sum(t * m) / np.sum(t * t))
        print(f"RAW IR against the true face distance ({len(pts)} readings, {t.min():.2f}-{t.max():.2f} m): "
              f"IR = {k:.3f} x true {q * 100:+.1f} cm, or IR = {scale:.3f} x true (residual "
              f"{np.std(m - scale * t) * 100:.1f} cm); use it with --ir-fit {k:.4f} {q:.4f} or in IR_FIT")
    if start is not None:
        rig.point(*start)


def sweep(rig, a, log):
    b = nearest_box(rig, a.box)
    body = rig.body_pose()
    if b is None or body is None:
        print("no robot pose, no gimbal angles or no box")
        return
    yaw0, pitch0 = aim_angles(body, (b["x"], b["y"]), rig.mount)
    if a.axis == "yaw":
        steps = [(yaw0 + math.radians(s), pitch0, s) for s in np.arange(-a.span_deg, a.span_deg + 1e-6, a.step_deg)]
    else:
        steps = [(yaw0, math.radians(s), s) for s in np.arange(a.pitch_from_deg, a.pitch_to_deg + 1e-6, a.step_deg)]
    print(f"{a.axis} sweep on box {b['id']}: {len(steps)} steps of {a.step_deg} deg")
    for yaw, pitch, s in steps:
        if STOP.is_set():
            break
        rig.point(yaw, pitch)
        r = rig.sample(a.ir_samples)
        r.update(rig.target_truth(r, b), step_deg=float(s))
        rig.snap(f"sweep_{a.axis}_{s:+05.1f}", f"{a.axis} {s:+.1f} deg: IR {fmt(r['ir_m'])} m, box {b['id']} face "
                 f"{fmt(r.get('target_slant_m'))} m")
        r["on_target"] = (None if r.get("target_err_slant_m") is None else
                          abs(r["target_err_slant_m"]) <= a.on_target_m and r.get("exp_box") in (None, b["id"])
                          and 0.0 <= r.get("target_hit_height_m", -1) <= BOX_HEIGHT_M)
        log.add(r)
        print(f"  {a.axis} {s:+6.1f} deg (box {fmt(r.get('target_off_deg'), '', 1)} deg off the axis, beam at "
              f"{fmt(r.get('target_hit_height_m'), ' m', 2)} on its face): IR {fmt(r['ir_m'])}, box face "
              f"{fmt(r.get('target_slant_m'))}" + ("  ON THE BOX" if r["on_target"] else "")
              + (f"  (box {r['exp_box']} nearer)" if r.get("exp_box") not in (None, b["id"]) else "") + raw_txt(r),
              flush=True)
    on = [r["step_deg"] for r in log.rows if r.get("on_target")]
    if on:
        print(f"the IR reads box {b['id']} (within {a.on_target_m * 100:.0f} cm) for {a.axis} from {min(on):+.1f} to "
              f"{max(on):+.1f} deg ({len(on)}/{len(log.rows)} steps)")
    else:
        print(f"the IR never read box {b['id']} (as the geometry expects it)")
    # the plateau of the target (the flat run of readings closest to its face distance), whatever box the geometry
    # thinks is in the beam: its centre against where the geometry put the box is the aiming error of the gimbal
    # (constant offset between the gimbal zero and the rigid body); its reading against the true face is the
    # range error of the IR
    true = float(np.median([r["target_slant_m"] for r in log.rows if r.get("target_slant_m") is not None] or [0]))
    p = target_plateau([r for r in log.rows if r.get("gimbal_yaw_deg") is not None], true) if a.axis == "yaw" else None
    if p is not None:
        g0 = math.degrees(yaw0)
        c = 0.5 * (p[0] + p[1])
        print(f"plateau of box {b['id']}: gimbal {p[0]:+.1f} to {p[1]:+.1f} deg, reading {p[2]:.3f} m against a true face "
              f"at {true:.3f} m ({(p[2] - true) * 100:+.1f} cm, x{p[2] / true:.3f})\n"
              f"  centre {c:+.1f} deg, the geometry put the box at {g0:+.1f} deg: the beam points {abs(c - g0):.1f} deg "
              f"to the {'LEFT' if c < g0 else 'RIGHT'} of where it should\n"
              f"  -> correction: --yaw-offset-deg {a.yaw_offset_deg - (c - g0):+.1f}")
    elif a.axis == "yaw":
        print(f"no flat run of readings near the face distance of box {b['id']} ({true:.2f} m)")
    rig.point(yaw0, pitch0)


def camera(rig, a, log):
    rays = rig.start_camera()
    for rep in range(a.repeats):
        if STOP.is_set():
            break
        print(f"--- round {rep + 1}/{a.repeats}")
        got = measure_targets(rays.rays, rig.ir, rig.body_pose, rig.gimbal, rig.mount, aim_from_args(a),
                              cancel=STOP.is_set)
        boxes = rig.boxes()
        xy = np.array([[b["x"], b["y"]] for b in boxes], float).reshape(-1, 2)
        for g in got:
            o, u = np.asarray(g["camera_pose"][:2]), unit(g["ray_angle_rad"])
            k = perr = merr = err = None
            if len(xy):
                v = xy - o
                along = v @ u
                perp = np.abs(v[:, 0] * u[1] - v[:, 1] * u[0])
                k = int(np.argmin(np.where(along > 0, perp + 0.2 * np.abs(along - g["mono_m"]), np.inf)))
                if g["position"] is not None:
                    perr = float(np.hypot(*(xy[k] - np.asarray(g["position"]))))
                merr = float(np.hypot(*(xy[k] - (o + g["mono_m"] * u))))
                if g["ir_horiz_m"] is not None:
                    err = g["ir_horiz_m"] - (float(np.hypot(*(xy[k] - np.asarray(g["ir_origin"])))) - APOTHEM_M)
            log.add(dict(round=rep, target=g["target"], true_box=None if k is None else boxes[k]["id"],
                         ir_m=g["ir_m"], ir_horiz_m=g["ir_horiz_m"], ir_hit_height_m=g["ir_hit_height_m"],
                         err_horiz_m=err, centre_err_m=perr, mono_centre_err_m=merr, on_face=g["on_face"],
                         aim_residual_deg=g["aim_residual_deg"], aim_lost=g["aim_lost"], consistent=g["consistent"],
                         gimbal_yaw_deg=g["gimbal_yaw_deg"], gimbal_pitch_deg=g["gimbal_pitch_deg"]))
            print(f"  YOLO box {g['target']} -> true box {None if k is None else boxes[k]['id']}: IR error "
                  f"{cm(err)} (flat), centre error {cm(perr)} (camera alone {cm(merr)})")
    stats([r.get("err_horiz_m") for r in log.rows], sum(r["ir_m"] is None for r in log.rows), "camera (IR range)")
    c = [r["centre_err_m"] for r in log.rows if r.get("centre_err_m") is not None]
    m = [r["mono_centre_err_m"] for r in log.rows if r.get("mono_centre_err_m") is not None]
    if c:
        print(f"box centre error: median {np.median(c) * 100:.1f} cm with the IR, "
              f"{np.median(m) * 100:.1f} cm with the camera alone")


def make_parser(with_mode=True, description=__doc__):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if with_mode:
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
    ap.add_argument("--yaw-offset-deg", type=float, default=None,
                    help="correction of the OptiTrack yaw of the robot, i.e. the gimbal aiming offset (default: "
                         "YAW_OFFSET_DEG of the robot, else 0)")
    ap.add_argument("--camera-calib", default=None)
    ap.add_argument("--camera-resolution", choices=["360", "540", "720"], default="720")
    ap.add_argument("--snap", action="store_true",
                    help="aim / sweep: save the camera frame after every pointing, with a cross at the image centre")
    ap.add_argument("--yaw-deg", type=float, default=None, help="watch: point the gimbal here first (+ left)")
    ap.add_argument("--pitch-deg", type=float, default=None, help="watch: point the gimbal here first (+ down)")
    ap.add_argument("--repeats", type=int, default=1, help="aim / camera: rounds")
    ap.add_argument("--boxes", type=int, nargs="+", default=None, help="aim: only these box rigid bodies")
    ap.add_argument("--box", type=int, default=None, help="sweep: the box rigid body (default: nearest the gimbal)")
    ap.add_argument("--axis", choices=["yaw", "pitch"], default="yaw", help="sweep: which gimbal axis moves")
    ap.add_argument("--span-deg", type=float, default=20.0, help="sweep yaw: from -span to +span around the box")
    ap.add_argument("--pitch-from-deg", type=float, default=-10.0, help="sweep pitch: from (+ down)")
    ap.add_argument("--pitch-to-deg", type=float, default=25.0, help="sweep pitch: to (+ down)")
    ap.add_argument("--step-deg", type=float, default=1.0, help="sweep: step")
    ap.add_argument("--on-target-m", type=float, default=0.05, help="sweep: reading within this of the box face")
    ap.add_argument("--out", default="data/ir_live/")
    ap.add_argument("--sim", action="store_true", help="no hardware: fake gimbal and IR, boxes of --sim-layout")
    ap.add_argument("--sim-layout", default="L4")
    ap.add_argument("--ground-truth", default="config/ground_truth_positions.json")
    ap.add_argument("--sim-pose", type=float, nargs=3, default=[-1.6, -0.6, 0.0], metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--sim-yaw-error-deg", type=float, default=0.0,
                    help="with --sim: the gimbal really points this much more to the left than the script believes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ir-fit", type=float, nargs=2, default=None, metavar=("K", "Q"),
                    help="IR range correction IR = K x true + Q [m] (default: IR_FIT of the robot; 1 0 = none)")
    add_ir_arguments(ap)
    return ap


def resolve(a, ap):
    """Per-robot defaults: gimbal aiming offset and IR range correction."""
    if a.yaw_offset_deg is None:
        a.yaw_offset_deg = YAW_OFFSET_DEG.get(a.robot_id, 0.0) if not a.sim else 0.0
        if a.yaw_offset_deg:
            print(f"gimbal aiming offset of {a.robot_id}: {a.yaw_offset_deg:+.1f} deg (YAW_OFFSET_DEG)")
    if a.ir_fit is None:
        a.ir_fit = IR_FIT.get(a.robot_id, (1.0, 0.0)) if not a.sim else (1.0, 0.0)
        if tuple(a.ir_fit) != (1.0, 0.0):
            print(f"IR range correction of {a.robot_id}: IR = {a.ir_fit[0]:.4f} x true {a.ir_fit[1] * 100:+.1f} cm "
                  f"(IR_FIT)")
    if not a.sim and a.rb is None:
        ap.error("--rb (OptiTrack rigid body of the robot) is required, or --sim")
    return a


def main():
    ap = make_parser()
    a = resolve(ap.parse_args(), ap)
    signal(signal.SIGINT, lambda *_: STOP.set())
    rig = Rig(a)
    log = Log(a, a.mode + (f"_{a.axis}" if a.mode == "sweep" else ""))
    try:
        dict(watch=watch, aim=aim, sweep=sweep, camera=camera)[a.mode](rig, a, log)
    finally:
        log.write()
        rig.close()


if __name__ == "__main__":
    main()
