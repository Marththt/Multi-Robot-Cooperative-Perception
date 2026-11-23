"""Range from the infrared (ToF) sensor, bearing from the camera: the robot turns to aim the IR at every box.

The ToF sensors of the RoboMaster EP measure only along their own axis (a narrow cone), so one sensor cannot
range the boxes spread over the camera's field of view. At a stop:

  1. camera + YOLO (yolo_gnn/segment.py, yolo_gnn/hybrid.frame_rays): one ray per segmented box, i.e. a world
     direction from the lens and a rough monocular range (floor contact / box height). No marker is needed.
  2. for every ray, the heading that puts the IR axis through the box (aim_yaw: the sensor is not at the body
     centre, so the monocular range is used for the parallax); the robot turns on the spot to it (turn_in_place,
     OptiTrack heading, fine tolerance).
  3. the camera checks the aim: the same box is found again (nearest ray within match_deg) and, if the IR axis
     still misses it by more than aim_tol_deg, the robot turns again (at most max_fine times).
  4. IR reading: median of n samples of the front sensor (robomaster_sdk/ir.py).
  5. box centre: the point of the camera ray whose distance from the IR origin is the IR range + the apothem
     of the box (the IR hits the face, not the centre); without a solution, along the IR axis. The IR range is
     checked against the monocular one (consistent): a beam that hit another box, the floor or nothing is
     flagged, not dropped.
  6. the robot turns back to the heading of the stop.

Every target gives one row: the raw quantities (rays, poses, IR samples, aim residual) and, as the marker rows,
distance and bearing_rad of the box centre from the camera pose of the stop (bearing positive to the right).

The same code runs without hardware (SimRays + FakeIR + vf.control.goto.FakeRobot): see acquisition/ir_live.py
--sim.
"""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from vf.control.goto import SIGN_Z, stop, wrap_pi
from yolo_gnn.camera import APOTHEM_M, CAMERA_FORWARD_FIX_M


@dataclass
class IRMount:
    """Where the IR sensor used for ranging sits, in the frame of the OptiTrack body (x forward, y left)."""
    index: int = 0              # position of the sensor in the sub_distance list (tools/ir_check.py)
    forward_m: float = 0.10     # ahead of the body origin
    right_m: float = 0.0        # to the right of the body origin
    yaw_deg: float = 0.0        # axis turned from the body heading, positive to the left
    beam_deg: float = 20.0      # full angle of the cone (only used by FakeIR)


@dataclass
class AimParams:
    aim_tol_deg: float = 1.5    # IR axis this close to the box direction: aimed
    max_fine: int = 2           # camera-checked corrections after the first turn
    match_deg: float = 6.0      # the same box in the next frame: ray within this angle of the expected one
    turn_timeout_s: float = 8.0
    n_ir: int = 10              # IR samples per box (median)
    consistency_m: float = 0.5  # IR range and monocular range further apart: flagged
    min_sep_deg: float = 3.0    # rays of one frame closer than this are one box
    restore_heading: bool = True


def turn_in_place(pose_fn, chassis, goal_yaw, tol_rad=math.radians(1.5), timeout_s=8.0, cancel=None,
                  kp=2.0, w_max_deg=40.0, w_min_deg=4.0, rate_hz=20.0):
    """Turns the robot on the spot to goal_yaw (OptiTrack heading), finer than vf/control/goto.go_to (5 deg): a
    proportional turn rate with a floor (w_min_deg), so that small errors still move the robot. Returns True when
    within tol_rad (held for 3 checks), False on timeout, lost pose or cancel. The robot is always stopped."""
    t0, lost, good = time.time(), 0, 0
    try:
        while time.time() - t0 < timeout_s:
            if cancel is not None and cancel():
                return False
            pose = pose_fn()
            if pose is None:
                lost += 1
                if lost >= 10:
                    return False
                time.sleep(1.0 / rate_hz)
                continue
            lost = 0
            err = wrap_pi(goal_yaw - pose[2])
            if abs(err) <= tol_rad:
                good += 1
                chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
                if good >= 3:
                    return True
            else:
                good = 0
                w = max(w_min_deg, min(w_max_deg, math.degrees(kp * abs(err)))) * (1 if err > 0 else -1)
                chassis.drive_speed(x=0, y=0, z=SIGN_Z * w, timeout=0.3)
            time.sleep(1.0 / rate_hz)
        return False
    finally:
        stop(chassis)


def unit(a):
    return np.array([math.cos(a), math.sin(a)])


def ir_origin(body, mount):
    """(x, y) of the sensor and the world yaw of its axis, for a body pose (x, y, yaw)."""
    x, y, yaw = body
    c, s = math.cos(yaw), math.sin(yaw)
    return (np.array([x + mount.forward_m * c + mount.right_m * s, y + mount.forward_m * s - mount.right_m * c]),
            wrap_pi(yaw + math.radians(mount.yaw_deg)))


def aim_yaw(body, target_xy, mount, iters=6):
    """Body heading that puts the IR axis through target_xy, turning on the spot (fixed-point iteration: the
    sensor moves with the heading)."""
    yaw = body[2]
    for _ in range(iters):
        o, _ = ir_origin((body[0], body[1], yaw), mount)
        d = np.asarray(target_xy, float) - o
        yaw = wrap_pi(math.atan2(d[1], d[0]) - math.radians(mount.yaw_deg))
    return yaw


def point_at_range(ray_o, ray_angle, centre, r):
    """Point of the ray (origin, angle) at distance r from centre (the farther solution ahead), or None."""
    u = unit(ray_angle)
    w = np.asarray(ray_o, float) - np.asarray(centre, float)
    b = float(w @ u)
    disc = b * b - (float(w @ w) - r * r)
    if disc < 0:
        return None
    t = -b + math.sqrt(disc)
    return None if t <= 0 else np.asarray(ray_o, float) + t * u


def dedupe(rays, min_sep_rad):
    """Rays closer than min_sep_rad in angle are one box (the one with the shorter range is kept)."""
    out = []
    for ang, d in sorted(rays, key=lambda r: r[1]):
        if all(abs(wrap_pi(ang - a)) >= min_sep_rad for a, _ in out):
            out.append((ang, d))
    return out


def nearest_ray(rays, angle, max_rad):
    best = min(rays, key=lambda r: abs(wrap_pi(r[0] - angle)), default=None)
    return best if best is not None and abs(wrap_pi(best[0] - angle)) <= max_rad else None


# ====================================================================
# Ray sources: real camera + YOLO, simulated
# ====================================================================

class YoloRays:
    """rays() -> (lens pose (x, y, yaw), [(world angle, monocular range), ...], image) from the latest frame."""

    def __init__(self, camera, segmenter, camera_matrix, dist_coeffs, pitch_deg, camera_pose_fn,
                 max_age_s=0.5, wait_s=0.15):
        from yolo_gnn.hybrid import Params
        self.camera, self.seg = camera, segmenter
        self.K = np.asarray(camera_matrix, float)
        self.D = np.asarray(dist_coeffs, float).ravel()
        self.pitch = math.radians(pitch_deg)
        self.pose_fn, self.max_age_s, self.wait_s = camera_pose_fn, max_age_s, wait_s
        self.p = Params()

    def rays(self):
        from yolo_gnn.hybrid import frame_rays
        time.sleep(self.wait_s)                     # a frame taken after the robot stopped turning
        img, age = self.camera.read()
        pose = self.pose_fn()
        if img is None or pose is None:
            return pose, [], img
        x, y, yaw = pose
        lens = (x + CAMERA_FORWARD_FIX_M * math.cos(yaw), y + CAMERA_FORWARD_FIX_M * math.sin(yaw), yaw)
        _, _, labels = self.seg(img)
        return lens, [(a, d) for a, d, _ in frame_rays(labels, self.K, self.D, lens, self.pitch, self.p)], img


class SimRays:
    """Simulated camera: every box centre in the field of view and range, not hidden behind a nearer box,
    with a bearing error and a monocular range error."""

    def __init__(self, boxes_xy, pose_fn, rng, hfov_deg=88.0, range_m=(0.3, 3.0), sd_bearing_deg=0.5,
                 sd_range_rel=0.15):
        self.boxes = np.asarray(boxes_xy, float).reshape(-1, 2)
        self.pose_fn, self.rng = pose_fn, rng
        self.half, self.range_m = math.radians(hfov_deg) / 2, range_m
        self.sd_b, self.sd_r = math.radians(sd_bearing_deg), sd_range_rel

    def rays(self):
        pose = self.pose_fn()
        x, y, yaw = pose
        v = self.boxes - [x, y]
        d = np.hypot(v[:, 0], v[:, 1])
        ang = np.arctan2(v[:, 1], v[:, 0])
        out = []
        for j in np.argsort(d):
            if not (self.range_m[0] <= d[j] <= self.range_m[1]) or abs(wrap_pi(ang[j] - yaw)) > self.half:
                continue
            hidden = any(d[k] < d[j] and abs(wrap_pi(ang[k] - ang[j])) < math.atan2(APOTHEM_M, d[k])
                         for k in range(len(d)) if k != j)
            if not hidden:
                out.append((wrap_pi(ang[j] + self.rng.normal(0, self.sd_b)),
                            float(d[j] * (1 + self.rng.normal(0, self.sd_r)))))
        return pose, out, None


class FakeIR:
    """Simulated ToF: the nearest box face inside the cone of the sensor, with a range error."""

    def __init__(self, boxes_xy, pose_fn, mount, rng, sd_m=0.01, sd_rel=0.01):
        self.boxes = np.asarray(boxes_xy, float).reshape(-1, 2)
        self.pose_fn, self.mount, self.rng = pose_fn, mount, rng
        self.sd_m, self.sd_rel = sd_m, sd_rel

    def measure(self, index, n=10, timeout_s=None):
        o, axis = ir_origin(self.pose_fn(), self.mount)
        v = self.boxes - o
        d = np.hypot(v[:, 0], v[:, 1])
        off = np.abs([wrap_pi(a - axis) for a in np.arctan2(v[:, 1], v[:, 0])])
        half = math.radians(self.mount.beam_deg) / 2
        hit = [d[j] - APOTHEM_M for j in range(len(d)) if off[j] <= half + math.atan2(APOTHEM_M, d[j])]
        if not hit:
            return dict(m=None, sd_m=None, n=0, n_total=n, valid=False, samples=[0.0] * n)
        r = min(hit)
        s = [float(r + self.rng.normal(0, self.sd_m + self.sd_rel * r)) for _ in range(n)]
        return dict(m=float(np.median(s)), sd_m=float(np.std(s)), n=n, n_total=n, valid=True, samples=s)


# ====================================================================
# Aim and measure
# ====================================================================

def measure_targets(rays_fn, ir, body_pose, chassis, mount, p=AimParams(), cancel=None, log=print,
                    on_frame=None):
    """Aims the IR at every box the camera sees from the current stop and measures it. Returns one dict per box
    (see the module doc). rays_fn() -> (lens pose, rays, image); ir.measure(index, n) -> dict;
    on_frame(tag, image) is called with every frame used (to save it)."""
    body0 = body_pose()
    if body0 is None:
        log("[IR] no pose: nothing measured")
        return []
    cam0, rays0, img = rays_fn()
    if on_frame is not None and img is not None:
        on_frame("stop", img)
    if cam0 is None:
        log("[IR] no camera pose: nothing measured")
        return []
    rays0 = dedupe(rays0, math.radians(p.min_sep_deg))
    rays0.sort(key=lambda r: wrap_pi(r[0] - body0[2]))          # right to left: one sweep
    here = (body0[0], body0[1])
    out = []
    for i, (ang0, d0) in enumerate(rays0):
        if cancel is not None and cancel():
            break
        target = np.asarray(cam0[:2]) + d0 * unit(ang0)
        cam, ang, d_mono, iters, turned, lost = cam0, ang0, d0, 0, True, False
        for it in range(1 + p.max_fine):
            iters = it + 1
            body = body_pose()
            if body is None:
                lost = True
                break
            goal = aim_yaw(body, target, mount)
            turned = turn_in_place(body_pose, chassis, goal, math.radians(p.aim_tol_deg), p.turn_timeout_s, cancel)
            cam_now, rays_now, img = rays_fn()
            if on_frame is not None and img is not None:
                on_frame(f"t{i}_a{it}", img)
            if cam_now is None:
                lost = True
                break
            expect = math.atan2(target[1] - cam_now[1], target[0] - cam_now[0])
            m = nearest_ray(rays_now, expect, math.radians(p.match_deg))
            if m is None:                                         # the box left the frame or was not segmented
                lost = True
                break
            cam, (ang, d_mono) = cam_now, m
            target = np.asarray(cam[:2]) + d_mono * unit(ang)
            body = body_pose()
            if body is None:
                lost = True
                break
            if abs(wrap_pi(aim_yaw(body, target, mount) - body[2])) <= math.radians(p.aim_tol_deg):
                break
        body = body_pose()
        o, axis = ir_origin(body, mount) if body is not None else (None, None)
        resid = (math.degrees(wrap_pi(aim_yaw(body, target, mount) - body[2])) if body is not None else None)
        r = ir.measure(mount.index, n=p.n_ir)
        est, how, consistent = None, None, None
        if r["m"] is not None and o is not None:
            rng = r["m"] + APOTHEM_M
            est = point_at_range(cam[:2], ang, o, rng)
            how = "camera ray"
            if est is None:
                est, how = o + rng * unit(axis), "IR axis"
            consistent = abs(rng - float(np.hypot(*(target - o)))) <= p.consistency_m
        row = dict(target=i, ray0_angle_rad=ang0, mono0_m=d0, ray_angle_rad=ang, mono_m=d_mono,
                   camera_pose0=list(cam0), camera_pose=list(cam), body_pose=None if body is None else list(body),
                   aim_iters=iters, aim_turn_ok=bool(turned), aim_lost=lost, aim_residual_deg=resid,
                   ir_origin=None if o is None else [float(v) for v in o], ir_axis_rad=axis,
                   ir_m=r["m"], ir_sd_m=r["sd_m"], ir_n=r["n"], ir_n_total=r["n_total"], ir_valid=r["valid"],
                   ir_samples=r["samples"], consistent=consistent, est_from=how,
                   position=None if est is None else [float(v) for v in est], distance=None, bearing_rad=None)
        if est is not None:
            v = est - np.asarray(cam0[:2])
            row["distance"] = float(np.hypot(*v))
            row["bearing_rad"] = float(wrap_pi(cam0[2] - math.atan2(v[1], v[0])))
        out.append(row)
        ir_txt = "-" if r["m"] is None else f"{r['m']:.3f} m"
        aim_txt = "-" if resid is None else f"{resid:+.1f} deg"
        log(f"[IR] box {i}: ray {math.degrees(wrap_pi(ang - body0[2])):+5.1f} deg, mono {d_mono:.2f} m, "
            f"IR {ir_txt} ({r['n']}/{r['n_total']}), aim {aim_txt} after {iters} turn(s)"
            + (", LOST" if lost else "") + ("" if consistent in (None, True) else ", INCONSISTENT"))
    if p.restore_heading and body0 is not None:
        turn_in_place(body_pose, chassis, body0[2], math.radians(3.0), p.turn_timeout_s, cancel)
    return out


def describe(mount, p):
    return dict(mount=asdict(mount), aim=asdict(p), apothem_m=APOTHEM_M)


# ====================================================================
# Command line (acquisition/ir_live.py)
# ====================================================================

def add_ir_arguments(p):
    m, a = IRMount(), AimParams()
    p.add_argument("--ir-index", type=int, default=m.index,
                   help="position of the ranging IR sensor in the SDK list, 0-3 (tools/ir_check.py)")
    p.add_argument("--ir-forward-m", type=float, default=m.forward_m, help="IR sensor ahead of the body origin [m]")
    p.add_argument("--ir-right-m", type=float, default=m.right_m, help="IR sensor right of the body origin [m]")
    p.add_argument("--ir-yaw-deg", type=float, default=m.yaw_deg, help="IR axis from the body heading, + left")
    p.add_argument("--ir-beam-deg", type=float, default=m.beam_deg, help="full cone of the IR (simulation only)")
    p.add_argument("--ir-freq", type=int, default=20, help="IR subscription rate [Hz]: 1, 5, 10, 20, 50")
    p.add_argument("--ir-samples", type=int, default=a.n_ir, help="IR samples per box (median)")
    p.add_argument("--aim-tol-deg", type=float, default=a.aim_tol_deg)
    p.add_argument("--aim-max-fine", type=int, default=a.max_fine)
    p.add_argument("--aim-match-deg", type=float, default=a.match_deg)
    p.add_argument("--aim-timeout-s", type=float, default=a.turn_timeout_s)
    p.add_argument("--ir-consistency-m", type=float, default=a.consistency_m)
    p.add_argument("--yolo-weights", default="results/yolo_seg/box_seg/weights/best.pt")
    p.add_argument("--yolo-device", default="auto")
    p.add_argument("--camera-pitch-deg", type=float, default=None,
                   help="downward pitch of the camera (default: yolo_gnn/camera.py PITCH_DEG of the robot, 11.5)")


def mount_from_args(a):
    return IRMount(index=a.ir_index, forward_m=a.ir_forward_m, right_m=a.ir_right_m, yaw_deg=a.ir_yaw_deg,
                   beam_deg=a.ir_beam_deg)


def aim_from_args(a):
    return AimParams(aim_tol_deg=a.aim_tol_deg, max_fine=a.aim_max_fine, match_deg=a.aim_match_deg,
                     turn_timeout_s=a.aim_timeout_s, n_ir=a.ir_samples, consistency_m=a.ir_consistency_m)


def camera_pitch(a):
    from yolo_gnn.camera import PITCH_DEG
    return a.camera_pitch_deg if a.camera_pitch_deg is not None else PITCH_DEG.get(a.robot_id, 11.5)
