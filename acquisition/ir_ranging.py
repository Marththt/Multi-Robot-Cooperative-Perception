"""Range from the infrared (ToF) sensor, bearing from the camera: the GIMBAL points the IR at every box.

The camera and one ToF sensor sit on the gimbal of the RoboMaster EP, so pointing the gimbal points both; the
chassis never moves. The ToF measures along its own axis only (a narrow cone), so it ranges one box at a time.

Geometry (GimbalMount; body = the OptiTrack rigid body of the chassis, x forward, y left):
  - the gimbal yaw axis stands gimbal_forward_m ahead of the body origin;
  - the IR sits ir_arm_m ahead of that axis, in the direction the gimbal points, at height ir_height_m; its axis
    is the gimbal direction (plus ir_yaw_deg / ir_pitch_deg if it is mounted slightly turned);
  - the camera sits cam_arm_m ahead of the axis, at cam_height_m, optical axis pitched like the gimbal plus
    cam_pitch_deg.
  With the gimbal at yaw psi_g (left +) and pitch theta (down +), the IR beam leaves from height h = ir_height_m
  and a slant reading d hits at horizontal distance d cos(theta) and height h - d sin(theta). The IR at 30 cm is at
  the height of the box tops (30 cm): a level beam grazes them, so the gimbal is pitched down to hit the face at
  aim_height_m (mid-face by default); a reading is a box only if the hit height is inside the face.

At a stop (measure_targets):
  1. camera + YOLO (yolo_gnn/segment.py, yolo_gnn/hybrid.frame_rays): one ray per segmented box, a world direction
     from the lens and a rough monocular range. No marker is needed.
  2. for every ray: gimbal yaw towards the box (the IR axis passes through the gimbal axis, so there is no
     parallax in yaw) and gimbal pitch down to aim_height_m on the face (the monocular range gives the distance).
  3. the camera checks the aim: the same box is found again and the gimbal is corrected (at most max_fine times).
  4. IR reading (median of n samples), turned into a horizontal range with the gimbal pitch read back.
  5. box centre: the point of the camera ray whose horizontal distance from the IR is the range + the apothem
     (the IR hits the face); checked against the monocular range (consistent) and against the face height.
  6. the gimbal goes back to where it was.

The same code runs without hardware (SimRays, FakeIR, FakeGimbal): acquisition/ir_live.py
--sim.
"""

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from vf.control.goto import wrap_pi
from yolo_gnn.camera import APOTHEM_M, BOX_HEIGHT_M


@dataclass
class GimbalMount:
    ir_index: int = 2               # position of the IR in the SDK distance list (tools/ir_check.py; 2 on G11)
    gimbal_forward_m: float = 0.0   # gimbal yaw axis ahead of the rigid body origin
    ir_arm_m: float = 0.10          # IR ahead of the gimbal axis, along the gimbal direction
    ir_height_m: float = 0.30       # IR above the floor
    ir_yaw_deg: float = 0.0         # IR axis turned from the gimbal direction, + left
    ir_pitch_deg: float = 0.0       # IR axis pitched from the gimbal pitch, + down
    cam_arm_m: float = 0.10         # camera ahead of the gimbal axis
    cam_height_m: float = 0.30      # lens above the floor
    cam_pitch_deg: float = 0.0      # optical axis pitched from the gimbal pitch, + down
    aim_height_m: float = 0.15      # where on the face the IR is aimed (box faces are BOX_HEIGHT_M high)
    beam_deg: float = 20.0          # full cone of the IR (only the simulated IR and the expected box use it)


@dataclass
class AimParams:
    aim_tol_deg: float = 1.0        # IR axis this close to the box direction: aimed
    max_fine: int = 2               # camera-checked corrections after the first pointing
    match_deg: float = 6.0          # the same box in the next frame: ray within this angle of the expected one
    n_ir: int = 10                  # IR samples per box (median)
    consistency_m: float = 0.5      # IR range and monocular range further apart: flagged
    min_sep_deg: float = 3.0        # rays of one frame closer than this are one box
    settle_s: float = 0.2           # wait after pointing, before reading


def unit(a):
    return np.array([math.cos(a), math.sin(a)])


def gimbal_axis(body, m):
    return np.array([body[0], body[1]]) + m.gimbal_forward_m * unit(body[2])


def ir_pose(body, gyaw, gpitch, m):
    """IR origin (x, y), beam heading (world yaw), beam pitch (down +) for the body pose and gimbal angles."""
    head = wrap_pi(body[2] + gyaw)
    o = gimbal_axis(body, m) + m.ir_arm_m * unit(head)
    return o, wrap_pi(head + math.radians(m.ir_yaw_deg)), gpitch + math.radians(m.ir_pitch_deg)


def camera_pose(body, gyaw, gpitch, m):
    """Lens (x, y, world yaw) and optical pitch (down +)."""
    head = wrap_pi(body[2] + gyaw)
    o = gimbal_axis(body, m) + m.cam_arm_m * unit(head)
    return (float(o[0]), float(o[1]), head), gpitch + math.radians(m.cam_pitch_deg)


def aim_angles(body, target_xy, m, horiz_m=None):
    """Gimbal (yaw, pitch) that puts the IR beam on the box at target_xy (its centre), hitting the face at
    aim_height_m. horiz_m: horizontal distance IR -> face if known (default from target_xy)."""
    a = gimbal_axis(body, m)
    v = np.asarray(target_xy, float) - a
    gyaw = wrap_pi(math.atan2(v[1], v[0]) - body[2] - math.radians(m.ir_yaw_deg))
    if horiz_m is None:
        horiz_m = max(0.05, float(np.hypot(*v)) - m.ir_arm_m - APOTHEM_M)
    gpitch = math.atan2(m.ir_height_m - m.aim_height_m, horiz_m) - math.radians(m.ir_pitch_deg)
    return gyaw, gpitch


def slant_to_floor(d, pitch, height):
    """(horizontal distance, height) where a slant reading d hits, for a beam at `pitch` (down +) from `height`."""
    return d * math.cos(pitch), height - d * math.sin(pitch)


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
    out = []
    for ang, d in sorted(rays, key=lambda r: r[1]):
        if all(abs(wrap_pi(ang - a)) >= min_sep_rad for a, _ in out):
            out.append((ang, d))
    return out


def nearest_ray(rays, angle, max_rad):
    best = min(rays, key=lambda r: abs(wrap_pi(r[0] - angle)), default=None)
    return best if best is not None and abs(wrap_pi(best[0] - angle)) <= max_rad else None


def expected(origin, heading, pitch, height, boxes, beam_deg):
    """What the IR should read, from the true boxes ({'id', 'x', 'y'}): the nearest box whose footprint enters the
    cone (in yaw) and whose face the beam hits between the floor and the top. Returns exp_box, exp_horiz_m
    (horizontal distance to the face), exp_slant_m (what the IR should read), exp_hit_height_m, off_deg (box
    centre off the beam axis, + left); exp_box None when no box is hit (exp_floor_m: where the beam meets the
    floor, when pitched down)."""
    half = math.radians(beam_deg) / 2
    best = None
    for b in boxes:
        v = np.array([b["x"], b["y"]]) - origin
        d = float(np.hypot(*v))
        off = wrap_pi(math.atan2(v[1], v[0]) - heading)
        horiz = d - APOTHEM_M
        hit = height - horiz * math.tan(pitch)
        if abs(off) <= half + math.atan2(APOTHEM_M, max(d, 1e-3)) and 0.0 <= hit <= BOX_HEIGHT_M:
            if best is None or horiz < best[1]:
                best = (b["id"], horiz, hit, off)
    floor = height / math.sin(pitch) if pitch > 1e-3 else None
    if best is None:
        return dict(exp_box=None, exp_horiz_m=None, exp_slant_m=None, exp_hit_height_m=None, off_deg=None,
                    exp_floor_m=floor)
    return dict(exp_box=best[0], exp_horiz_m=best[1], exp_slant_m=best[1] / math.cos(pitch),
                exp_hit_height_m=best[2], off_deg=math.degrees(best[3]), exp_floor_m=floor)


# ====================================================================
# Ray sources: real camera + YOLO, simulated
# ====================================================================

class YoloRays:
    """rays() -> (lens pose (x, y, yaw), [(world angle, monocular range), ...], image) from the latest frame, with
    the gimbal angles read at that moment. The monocular range of frame_rays uses the lens height: it is set
    here on the imported module (yolo_gnn/hybrid.py itself is not changed)."""

    def __init__(self, camera, segmenter, camera_matrix, dist_coeffs, body_pose_fn, gimbal, mount, wait_s=0.15):
        from yolo_gnn import hybrid
        hybrid.CAMERA_HEIGHT_M = mount.cam_height_m
        self.camera, self.seg = camera, segmenter
        self.K = np.asarray(camera_matrix, float)
        self.D = np.asarray(dist_coeffs, float).ravel()
        self.body_pose_fn, self.gimbal, self.mount, self.wait_s = body_pose_fn, gimbal, mount, wait_s
        self.p = hybrid.Params()

    def rays(self):
        from yolo_gnn.hybrid import frame_rays
        time.sleep(self.wait_s)
        img, _ = self.camera.read()
        body, ang = self.body_pose_fn(), self.gimbal.angles()
        if img is None or body is None or ang is None:
            return None, [], img
        lens, pitch = camera_pose(body, ang[0], ang[1], self.mount)
        _, _, labels = self.seg(img)
        return lens, [(a, d) for a, d, _ in frame_rays(labels, self.K, self.D, lens, pitch, self.p)], img


class SimRays:
    """Simulated camera on the gimbal: every box centre in the field of view and range, not hidden behind a
    nearer box, with a bearing error and a monocular range error."""

    def __init__(self, boxes_xy, body_pose_fn, gimbal, mount, rng, hfov_deg=88.0, range_m=(0.3, 3.0),
                 sd_bearing_deg=0.5, sd_range_rel=0.15):
        self.boxes = np.asarray(boxes_xy, float).reshape(-1, 2)
        self.body_pose_fn, self.gimbal, self.mount, self.rng = body_pose_fn, gimbal, mount, rng
        self.half, self.range_m = math.radians(hfov_deg) / 2, range_m
        self.sd_b, self.sd_r = math.radians(sd_bearing_deg), sd_range_rel

    def rays(self):
        gy, gp = self.gimbal.angles()
        lens, _ = camera_pose(self.body_pose_fn(), gy, gp, self.mount)
        v = self.boxes - lens[:2]
        d = np.hypot(v[:, 0], v[:, 1])
        ang = np.arctan2(v[:, 1], v[:, 0])
        out = []
        for j in np.argsort(d):
            if not (self.range_m[0] <= d[j] <= self.range_m[1]) or abs(wrap_pi(ang[j] - lens[2])) > self.half:
                continue
            hidden = any(d[k] < d[j] and abs(wrap_pi(ang[k] - ang[j])) < math.atan2(APOTHEM_M, d[k])
                         for k in range(len(d)) if k != j)
            if not hidden:
                out.append((wrap_pi(ang[j] + self.rng.normal(0, self.sd_b)),
                            float(d[j] * (1 + self.rng.normal(0, self.sd_r)))))
        return lens, out, None


class FakeIR:
    """Simulated ToF on the gimbal: the expected() box face at the slant distance, or the floor, with noise;
    nothing (not valid) when the beam meets neither within 3 m."""

    def __init__(self, boxes_xy, body_pose_fn, gimbal, mount, rng, sd_m=0.01, sd_rel=0.01):
        self.boxes = [dict(id=11 + k, x=float(x), y=float(y)) for k, (x, y) in enumerate(boxes_xy)]
        self.body_pose_fn, self.gimbal, self.mount, self.rng = body_pose_fn, gimbal, mount, rng
        self.sd_m, self.sd_rel = sd_m, sd_rel
        self.first_raw = "(simulated IR)"

    def reading(self):
        gy, gp = self.gimbal.angles()
        o, head, pitch = ir_pose(self.body_pose_fn(), gy, gp, self.mount)
        e = expected(o, head, pitch, self.mount.ir_height_m, self.boxes, self.mount.beam_deg)
        r = e["exp_slant_m"] if e["exp_box"] is not None else e["exp_floor_m"]
        return r if r is not None and r <= 3.0 else None

    def latest(self):
        r = self.reading()
        return time.time(), [r or 0.0] * 4

    def valid(self, m):
        return m is not None and 0.1 <= m <= 10.0

    def measure(self, index, n=10, timeout_s=None):
        r = self.reading()
        if r is None:
            return dict(m=None, sd_m=None, n=0, n_total=n, valid=False, samples=[0.0] * n)
        s = [float(r + self.rng.normal(0, self.sd_m + self.sd_rel * r)) for _ in range(n)]
        return dict(m=float(np.median(s)), sd_m=float(np.std(s)), n=n, n_total=n, valid=True, samples=s)

    def stop(self):
        pass


class FakeGimbal:
    """Same interface as robomaster_sdk.gimbal.Gimbal, no hardware: the angles are set at once."""

    def __init__(self, yaw=0.0, pitch=0.0):
        self.yaw, self.pitch = yaw, pitch
        self.first_raw = "(fake gimbal)"

    def start(self):
        return self

    def angles(self):
        mv = getattr(self, "moving", None)
        if mv is not None:                                   # start_move: constant speed towards the target
            t0, y0, y1, w = mv
            step = w * (time.time() - t0)
            if step >= abs(y1 - y0):
                self.yaw, self.moving = y1, None
            else:
                self.yaw = y0 + math.copysign(step, y1 - y0)
        dv = getattr(self, "driving", None)
        if dv is not None:                                   # drive: constant speed until drive(0) or point()
            t0, w = dv
            now = time.time()
            self.yaw += w * (now - t0)
            self.driving = (now, w)
        return self.yaw, self.pitch

    def start_move(self, yaw, pitch, speed_deg_s=10.0):
        self.pitch = pitch
        self.moving = (time.time(), self.yaw, yaw, math.radians(speed_deg_s))

    def drive(self, yaw_speed_deg_s, pitch_speed_deg_s=0.0):
        self.angles()
        self.driving = (time.time(), math.radians(yaw_speed_deg_s)) if yaw_speed_deg_s else None

    def point(self, yaw, pitch, **kw):
        self.moving = self.driving = None
        self.yaw = yaw
        self.pitch = max(math.radians(-30.0), min(math.radians(25.0), pitch))
        return 0.0, 0.0

    def recenter(self, timeout_s=None):
        self.yaw = self.pitch = 0.0

    def close(self, recenter=True):
        if recenter:
            self.recenter()


# ====================================================================
# Aim with the gimbal and measure
# ====================================================================

def range_row(r, body, gimbal, m):
    """IR reading -> pose of the beam, horizontal range and hit height (the gimbal angles are read back)."""
    ang = gimbal.angles()
    if body is None or ang is None:
        return dict(ir_origin=None, ir_heading_rad=None, ir_pitch_rad=None, ir_horiz_m=None, ir_hit_height_m=None,
                    gimbal_yaw_deg=None, gimbal_pitch_deg=None)
    o, head, pitch = ir_pose(body, ang[0], ang[1], m)
    horiz = hit = None
    if r["m"] is not None:
        horiz, hit = slant_to_floor(r["m"], pitch, m.ir_height_m)
    return dict(ir_origin=[float(o[0]), float(o[1])], ir_heading_rad=head, ir_pitch_rad=pitch, ir_horiz_m=horiz,
                ir_hit_height_m=hit, gimbal_yaw_deg=math.degrees(ang[0]), gimbal_pitch_deg=math.degrees(ang[1]))


def measure_targets(rays_fn, ir, body_pose, gimbal, m, p=AimParams(), cancel=None, log=print, on_frame=None):
    """Points the IR (with the gimbal) at every box the camera sees and measures it. One dict per box."""
    body = body_pose()
    start = gimbal.angles()
    if body is None or start is None:
        log("[IR] no pose or no gimbal angles: nothing measured")
        return []
    cam0, rays0, img = rays_fn()
    if on_frame is not None and img is not None:
        on_frame("start", img)
    if cam0 is None:
        log("[IR] no camera pose: nothing measured")
        return []
    rays0 = dedupe(rays0, math.radians(p.min_sep_deg))
    rays0.sort(key=lambda r: wrap_pi(r[0] - body[2]))
    out = []
    for i, (ang0, d0) in enumerate(rays0):
        if cancel is not None and cancel():
            break
        target = np.asarray(cam0[:2]) + d0 * unit(ang0)
        cam, ang, d_mono, iters, lost, resid = cam0, ang0, d0, 0, False, None
        for it in range(1 + p.max_fine):
            iters = it + 1
            gimbal.point(*aim_angles(body, target, m))
            cam_now, rays_now, img = rays_fn()
            if on_frame is not None and img is not None:
                on_frame(f"t{i}_a{it}", img)
            if cam_now is None:
                lost = True
                break
            expect = math.atan2(target[1] - cam_now[1], target[0] - cam_now[0])
            hit = nearest_ray(rays_now, expect, math.radians(p.match_deg))
            if hit is None:
                lost = True
                break
            cam, (ang, d_mono) = cam_now, hit
            target = np.asarray(cam[:2]) + d_mono * unit(ang)
            g = gimbal.angles()
            resid = math.degrees(wrap_pi(aim_angles(body, target, m)[0] - g[0])) if g else None
            if resid is not None and abs(resid) <= p.aim_tol_deg:
                break
        time.sleep(p.settle_s)
        body = body_pose() or body
        r = ir.measure(m.ir_index, n=p.n_ir)
        rr = range_row(r, body, gimbal, m)
        est, how, consistent, on_face = None, None, None, None
        if rr["ir_horiz_m"] is not None:
            o = np.asarray(rr["ir_origin"])
            rng = rr["ir_horiz_m"] + APOTHEM_M
            est = point_at_range(cam[:2], ang, o, rng)
            how = "camera ray"
            if est is None:
                est, how = o + rng * unit(rr["ir_heading_rad"]), "IR axis"
            consistent = abs(rng - float(np.hypot(*(target - o)))) <= p.consistency_m
            on_face = 0.0 <= rr["ir_hit_height_m"] <= BOX_HEIGHT_M
        row = dict(target=i, ray0_angle_rad=ang0, mono0_m=d0, ray_angle_rad=ang, mono_m=d_mono,
                   camera_pose0=list(cam0), camera_pose=list(cam), body_pose=list(body), aim_iters=iters,
                   aim_lost=lost, aim_residual_deg=resid, ir_m=r["m"], ir_sd_m=r["sd_m"], ir_n=r["n"],
                   ir_n_total=r["n_total"], ir_valid=r["valid"], ir_samples=r["samples"], consistent=consistent,
                   on_face=on_face, est_from=how, position=None if est is None else [float(v) for v in est],
                   distance=None, bearing_rad=None, **rr)
        if est is not None:
            v = est - np.asarray(cam0[:2])
            row["distance"] = float(np.hypot(*v))
            row["bearing_rad"] = float(wrap_pi(cam0[2] - math.atan2(v[1], v[0])))
        out.append(row)
        ir_txt = "-" if r["m"] is None else f"{r['m']:.3f} m slant, {rr['ir_horiz_m']:.3f} m flat"
        log(f"[IR] box {i}: ray {math.degrees(wrap_pi(ang - body[2])):+5.1f} deg, mono {d_mono:.2f} m, IR {ir_txt} "
            f"({r['n']}/{r['n_total']})" + (", LOST" if lost else "")
            + ("" if consistent in (None, True) else ", INCONSISTENT")
            + ("" if on_face in (None, True) else ", NOT ON THE FACE"))
    gimbal.point(*start)
    return out


def describe(m, p):
    return dict(mount=asdict(m), aim=asdict(p), apothem_m=APOTHEM_M, box_height_m=BOX_HEIGHT_M)


# ====================================================================
# Command line (acquisition/ir_live.py)
# ====================================================================

def add_ir_arguments(ap):
    m, a = GimbalMount(), AimParams()
    ap.add_argument("--ir-index", type=int, default=m.ir_index,
                    help="position of the IR in the SDK distance list, 0-3 (tools/ir_check.py)")
    ap.add_argument("--gimbal-forward-m", type=float, default=m.gimbal_forward_m,
                    help="gimbal yaw axis ahead of the rigid body origin [m]")
    ap.add_argument("--ir-arm-m", type=float, default=m.ir_arm_m,
                    help="IR ahead of the gimbal yaw axis, along the gimbal direction [m]")
    ap.add_argument("--ir-height-m", type=float, default=m.ir_height_m, help="IR above the floor [m]")
    ap.add_argument("--ir-yaw-deg", type=float, default=m.ir_yaw_deg)
    ap.add_argument("--ir-pitch-deg", type=float, default=m.ir_pitch_deg)
    ap.add_argument("--cam-arm-m", type=float, default=m.cam_arm_m)
    ap.add_argument("--cam-height-m", type=float, default=m.cam_height_m)
    ap.add_argument("--cam-pitch-deg", type=float, default=m.cam_pitch_deg)
    ap.add_argument("--aim-height-m", type=float, default=m.aim_height_m,
                    help="height on the box face the IR is aimed at [m] (faces are 0.30 m high)")
    ap.add_argument("--ir-beam-deg", type=float, default=m.beam_deg, help="full cone of the IR, for the expected box")
    ap.add_argument("--ir-freq", type=int, default=20, help="IR subscription rate [Hz]: 1, 5, 10, 20, 50")
    ap.add_argument("--ir-samples", type=int, default=a.n_ir, help="IR samples per reading (median)")
    ap.add_argument("--aim-tol-deg", type=float, default=a.aim_tol_deg)
    ap.add_argument("--aim-max-fine", type=int, default=a.max_fine)
    ap.add_argument("--aim-match-deg", type=float, default=a.match_deg)
    ap.add_argument("--ir-consistency-m", type=float, default=a.consistency_m)
    ap.add_argument("--settle-s", type=float, default=a.settle_s, help="wait after pointing, before reading")
    ap.add_argument("--gimbal-yaw-sign", type=float, default=None, help="override robomaster_sdk.gimbal.YAW_SIGN")
    ap.add_argument("--gimbal-pitch-sign", type=float, default=None, help="override robomaster_sdk.gimbal.PITCH_SIGN")
    ap.add_argument("--yolo-weights", default="results/yolo_seg/box_seg/weights/best.pt")
    ap.add_argument("--yolo-device", default="auto")


def mount_from_args(a):
    return GimbalMount(ir_index=a.ir_index, gimbal_forward_m=a.gimbal_forward_m, ir_arm_m=a.ir_arm_m,
                       ir_height_m=a.ir_height_m, ir_yaw_deg=a.ir_yaw_deg, ir_pitch_deg=a.ir_pitch_deg,
                       cam_arm_m=a.cam_arm_m, cam_height_m=a.cam_height_m, cam_pitch_deg=a.cam_pitch_deg,
                       aim_height_m=a.aim_height_m, beam_deg=a.ir_beam_deg)


def aim_from_args(a):
    return AimParams(aim_tol_deg=a.aim_tol_deg, max_fine=a.aim_max_fine, match_deg=a.aim_match_deg,
                     n_ir=a.ir_samples, consistency_m=a.ir_consistency_m, settle_s=a.settle_s)
