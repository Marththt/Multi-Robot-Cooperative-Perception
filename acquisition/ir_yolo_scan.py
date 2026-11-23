"""One robot finding the boxes with YOLO and ranging each with the IR on its gimbal, for the live demo.

As acquisition/ir_scan.py, but the boxes to range come from the camera, not from OptiTrack: in its turn the robot
points the gimbal at each --look-yaw-deg (default 0: straight ahead of the chassis), takes a frame, YOLO
(yolo_gnn.segment, the weights of model C) gives one ray per box it sees (direction and a rough monocular
distance), and for every ray the gimbal points the IR at it, checks the aim on a new frame (up to --aim-max-fine
corrections) and reads the IR (acquisition/ir_ranging.measure_targets). Each reading becomes a state for
demo/live.py exactly as in ir_scan.py:

  pose        the IR origin and the direction of the beam at that reading
  detections  one: class "box", distance = horizontal distance to the face hit by the beam, bearing 0 (along the
              beam), no viewing angle, position = where the beam hits

so the demo runs MHT and the graph network on it unchanged (--sensor-model config/sensor_model_ir.json).
OptiTrack is still used for the pose of the robot, for the aim calibration at start (a box it tracks) and, in
the log only, to say which true box each reading belongs to (true_box, true_horiz_m: the analysis, not the method).

A reading is dropped (a state with no detection) when the aim was lost (the box is no longer in the frame), when
the IR gives no valid range, when the beam meets the floor or passes over the boxes, or, unless
--keep-inconsistent, when the IR range and the monocular range of YOLO differ by more than --ir-consistency-m
(the beam slipped past the box, or YOLO saw something that is not a box at that distance).

One turn = one look and every box YOLO found in it (a few seconds per box): acquisition/ir_live_robots.py
--scanner yolo gives the turns a longer timeout. --fan works as in ir_scan.py (once every --fan-every looks).

YOLO needs ultralytics and torch in this environment, and the camera intrinsics: --camera-calib (default
config/checkerboard_calib_P0.json, same camera model on every robot).

    python -m acquisition.ir_yolo_scan --robot-id G11 --rb 8 --state-out data/live_ir_yolo/G11.json --verbose
    python -m acquisition.ir_yolo_scan --robot-id S1 --sim --state-out data/live_ir_sim/S1.json --verbose
"""

import math
import os
import signal
import time

import numpy as np

from acquisition.ir_live import STOP, Rig, estimate_offset, make_parser, resolve
from acquisition.ir_ranging import APOTHEM_M, aim_from_args, measure_targets
from acquisition.ir_scan import CONNECTED_MSG, Turns, fan, park, publish, state
from vf.control.goto import wrap_pi

DEFAULT_CAMERA_CALIB = "config/checkerboard_calib_P0.json"


def true_box(boxes, xy):
    """(id, centre) of the OptiTrack box nearest a point, for the log (None if none within 0.4 m)."""
    best = None
    for b in boxes:
        d = math.hypot(b["x"] - xy[0], b["y"] - xy[1])
        if d <= 0.4 and (best is None or d < best[0]):
            best = (d, b)
    return (None, None) if best is None else (best[1]["id"], (best[1]["x"], best[1]["y"]))


def reading_state(a, g, boxes):
    """The live state of one YOLO + IR reading (acquisition/ir_scan.py format)."""
    if g.get("ir_origin") is None or g.get("ir_heading_rad") is None:
        return None
    o, head = g["ir_origin"], g["ir_heading_rad"]
    pose = (o[0], o[1], head)
    horiz, hit = g.get("ir_horiz_m"), g.get("ir_hit_height_m")
    why = None
    if g.get("aim_lost"):
        why = "aim lost"
    elif g.get("ir_m") is None or horiz is None:
        why = "no valid IR"
    elif not 0.0 <= hit <= 0.6:
        why = "beam on the floor or above the boxes"
    elif g.get("consistent") is False and not a.keep_inconsistent:
        why = "IR and monocular range disagree"
    dets = []
    if why is None:
        dets.append(dict(class_name="box", confidence=1.0, distance=horiz, bearing_rad=0.0, viewing_angle_rad=None,
                         position=[o[0] + horiz * math.cos(head), o[1] + horiz * math.sin(head)], bbox=None))
    # the analysis only: the true box near the point the beam hit (centre = hit + one apothem along the beam)
    hit_xy = None if horiz is None else (o[0] + (horiz + APOTHEM_M) * math.cos(head),
                                         o[1] + (horiz + APOTHEM_M) * math.sin(head))
    yolo_xy = (g["camera_pose"][0] + g["mono_m"] * math.cos(g["ray_angle_rad"]),
               g["camera_pose"][1] + g["mono_m"] * math.sin(g["ray_angle_rad"]))
    tid, tc = true_box(boxes, hit_xy or yolo_xy)
    true_h = None if tc is None else math.hypot(tc[0] - o[0], tc[1] - o[1]) - APOTHEM_M
    return state(a, pose, dets, dict(
        dropped=why, yolo_target=g["target"], yolo_ray_deg=math.degrees(g["ray_angle_rad"]), yolo_mono_m=g["mono_m"],
        aim_iters=g["aim_iters"], aim_residual_deg=g["aim_residual_deg"], ir_raw_m=None, ir_slant_m=g.get("ir_m"),
        ir_hit_height_m=hit, ir_valid_n=g.get("ir_n"), consistent=g.get("consistent"), true_box=tid,
        aimed_box=tid, true_horiz_m=true_h, ir_slot=a.ir_slot, yaw_offset_deg=a.yaw_offset_now,
        gimbal=[g.get("gimbal_yaw_deg"), g.get("gimbal_pitch_deg")]))


def look(rig, a, rays, look_yaw):
    """One look: the gimbal at look_yaw (relative to the chassis), YOLO, then every box ranged with the IR."""
    rig.point(math.radians(look_yaw), math.radians(a.look_pitch_deg))
    msgs = []
    got = measure_targets(rays.rays, rig.ir, rig.body_pose, rig.gimbal, rig.mount, aim_from_args(a),
                          cancel=lambda: STOP.is_set() or os.path.exists(a.stop_file), log=msgs.append)
    if a.verbose:
        for m in msgs:
            print(m, flush=True)
    return got


def scan(rig, a):
    rays = rig.start_camera()
    turns = Turns(a)
    n_look, last_fan = 0, None
    park(rig, a)
    while not STOP.is_set() and not os.path.exists(a.stop_file):
        if not turns.wait():
            break
        rounds = n_look // len(a.look_yaw_deg)                   # complete rounds of looks so far
        if (a.fan and n_look > 0 and n_look % len(a.look_yaw_deg) == 0 and rounds % a.fan_every == 0
                and last_fan != rounds):
            n = fan(rig, a)
            last_fan = rounds
            if a.verbose:
                print(f"fan: {n} IR samples", flush=True)
            park(rig, a)
            turns.done()
            continue
        look_yaw = a.look_yaw_deg[n_look % len(a.look_yaw_deg)]
        n_look += 1
        a.yaw_offset_now, a.ir_slot = math.degrees(rig.yaw_off), rig.mount.ir_index
        t0 = time.time()
        got = look(rig, a, rays, look_yaw)
        boxes, body = rig.boxes(), rig.body_pose()
        kept = 0
        for g in got:
            st = reading_state(a, g, boxes)
            if st is None:
                continue
            kept += bool(st["detections"])
            publish(a, st, boxes=boxes, body=body)
            if a.verbose:
                d = st["detections"][0]["distance"] if st["detections"] else None
                err = None if d is None or st["true_horiz_m"] is None else d - st["true_horiz_m"]
                print(f"YOLO box {g['target']} (ray {math.degrees(wrap_pi(g['ray_angle_rad'] - body[2])) if body else 0:+.1f} "
                      f"deg): " + (f"dropped, {st['dropped']}" if d is None else f"{d:.3f} m")
                      + ("" if err is None else f" (true box {st['true_box']}, error {err * 100:+.1f} cm)"), flush=True)
        if not got and body is not None:
            ang = rig.gimbal.angles() or (0.0, 0.0)
            publish(a, state(a, (body[0], body[1], wrap_pi(body[2] + ang[0])), [], dict(note="YOLO saw no box")),
                    boxes=boxes, body=body)
        if a.verbose:
            print(f"look {look_yaw:+.0f} deg: YOLO {len(got)} box(es), {kept} ranged, {time.time() - t0:.1f} s",
                  flush=True)
        park(rig, a)
        turns.done()


def main():
    ap = make_parser(with_mode=False, description=__doc__)
    ap.add_argument("--state-out", required=True, help="live state file of this robot (demo/live.py --live-dir)")
    ap.add_argument("--stop-file", default="", help="the scan stops when this file exists")
    ap.add_argument("--verbose", action="store_true", help="one line per box")
    ap.add_argument("--look-yaw-deg", type=float, nargs="+", default=[0.0],
                    help="gimbal yaw of the looks, relative to the chassis (+ left), one per turn in turn: e.g. "
                         "-40 0 40 to cover more than the field of view of the camera")
    ap.add_argument("--look-pitch-deg", type=float, default=5.0, help="gimbal pitch of the looks (+ down)")
    ap.add_argument("--keep-inconsistent", action="store_true",
                    help="keep the readings whose IR range and YOLO monocular range disagree")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the aim calibration at start")
    ap.add_argument("--turns", action="store_true",
                    help="range only when <live dir>/turn.json gives this robot the turn (acquisition/ir_live_robots.py)")
    ap.add_argument("--park-up-deg", type=float, default=20.0,
                    help="with --turns: between its turns the gimbal points the beam this much upwards")
    ap.add_argument("--fan", action="store_true", help="a fan of the gimbal for the free space (acquisition/ir_scan.py)")
    ap.add_argument("--fan-every", type=int, default=1, help="one fan every this many rounds of looks")
    ap.add_argument("--fan-deg", type=float, default=40.0)
    ap.add_argument("--fan-pitch-deg", type=float, default=4.0)
    ap.add_argument("--fan-speed-dps", type=float, default=10.0)
    a = resolve(ap.parse_args(), ap)
    if not a.sim and a.camera_calib is None:
        a.camera_calib = DEFAULT_CAMERA_CALIB
    a.stop_file = a.stop_file or os.path.join(os.path.dirname(a.state_out) or ".", "STOP")
    a.log_dir = os.path.join(os.path.dirname(a.state_out) or ".", "log")
    a.fan_dir = 1
    a.yaw_offset_now = a.yaw_offset_deg
    os.makedirs(os.path.dirname(a.state_out) or ".", exist_ok=True)
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    rig = Rig(a)
    if not a.no_calibrate:
        estimate_offset(rig)
    print(CONNECTED_MSG, flush=True)
    try:
        scan(rig, a)
    finally:
        rig.close()
        print("stopped", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
