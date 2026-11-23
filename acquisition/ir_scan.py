"""One robot ranging the boxes with the IR on its gimbal, continuously, for the live demo.

The gimbal points the IR at every box OptiTrack tracks within --max-range (sweeping from right to left and back),
reads it (median of --ir-samples, with the IR range correction of the robot, acquisition/ir_live.IR_FIT) and,
after each box, rewrites the live state of the robot (--state-out) in the format of acquisition/static_trial.py
--live, so that demo/live.py runs MHT and the graph network on it unchanged:

  pose        the IR origin and the direction of the beam at that reading ("camera pose")
  detections  one: class "box", distance = horizontal distance to the face hit by the beam, bearing 0 (along the
              beam), viewing angle 0, position = where the beam hits

demo/live.py pools the detections of the last --window-s seconds of every robot, each with the pose of its own
state, so it must be started with a window longer than one sweep (a few seconds per box) and with a sensor
calibration that does not correct the IR ranges (config/sensor_calib_ir.json): see acquisition/ir_live_robots.py.

The aiming uses the OptiTrack positions of the boxes (the camera plays no part yet): the robot ranges every box,
including one hidden behind another (then the IR reads the nearer one). A reading that is not valid writes a state
with no detection.

    python -m acquisition.ir_scan --robot-id G1 --rb 8 --state-out data/live_ir/G1.json
"""

import math
import os
import signal
import time


from acquisition.ir_live import STOP, Rig, make_parser, resolve
from acquisition.ir_ranging import aim_angles
from demo.methods import write_json
from vf.control.goto import wrap_pi

CONNECTED_MSG = "IR scan running"          # acquisition/ir_live_robots.py waits for this before the next robot


def state(a, pose, detections, extra):
    return dict(robot_id=a.robot_id, t=time.time(), pose=[float(v) for v in pose], detections=detections, p_d=1.0,
                source="ir", **extra)


def scan(rig, a):
    n_sweep = 0
    while not STOP.is_set() and not os.path.exists(a.stop_file):
        body, boxes = rig.body_pose(), rig.boxes()
        if body is None:
            time.sleep(0.2)
            continue
        boxes = [b for b in boxes if math.hypot(b["x"] - body[0], b["y"] - body[1]) <= a.max_range]
        if not boxes:
            ang = rig.gimbal.angles() or (0.0, 0.0)
            write_json(a.state_out, state(a, (body[0], body[1], wrap_pi(body[2] + ang[0])), [],
                                          dict(note="no box within range")))
            time.sleep(0.5)
            continue
        boxes.sort(key=lambda b: wrap_pi(math.atan2(b["y"] - body[1], b["x"] - body[0]) - body[2]))
        if n_sweep % 2:
            boxes.reverse()                                   # back and forth: the gimbal never makes a long jump
        n_sweep += 1
        for b in boxes:
            if STOP.is_set() or os.path.exists(a.stop_file):
                break
            body = rig.body_pose() or body
            rig.point(*aim_angles(body, (b["x"], b["y"]), rig.mount))
            r = rig.sample(a.ir_samples)
            if r.get("ir_x") is None:
                continue
            pose = (r["ir_x"], r["ir_y"], math.radians(r["ir_heading_deg"]))
            dets = []
            if r["ir_m"] is not None and r.get("ir_horiz_m") is not None and 0.0 <= r["ir_hit_height_m"] <= 0.6:
                d = r["ir_horiz_m"]
                dets.append(dict(class_name="box", confidence=1.0, distance=d, bearing_rad=0.0, viewing_angle_rad=0.0,
                                 position=[pose[0] + d * math.cos(pose[2]), pose[1] + d * math.sin(pose[2])],
                                 bbox=None))
            truth = rig.target_truth(r, b)
            write_json(a.state_out, state(a, pose, dets, dict(
                aimed_box=b["id"], ir_raw_m=r.get("ir_raw_m"), ir_slant_m=r["ir_m"], ir_hit_height_m=r.get("ir_hit_height_m"),
                true_horiz_m=truth.get("target_horiz_m"), gimbal=[r.get("gimbal_yaw_deg"), r.get("gimbal_pitch_deg")])))
            if a.verbose:
                err = None if not dets or truth.get("target_horiz_m") is None else dets[0]["distance"] - truth["target_horiz_m"]
                print(f"box {b['id']}: " + ("no valid IR" if not dets else
                      f"{dets[0]['distance']:.3f} m" + ("" if err is None else f" (error {err * 100:+.1f} cm)")),
                      flush=True)


def main():
    ap = make_parser(with_mode=False, description=__doc__)
    ap.add_argument("--state-out", required=True, help="live state file of this robot (demo/live.py --live-dir)")
    ap.add_argument("--max-range", type=float, default=3.5, help="only boxes closer than this [m]")
    ap.add_argument("--stop-file", default="", help="the scan stops when this file exists")
    ap.add_argument("--verbose", action="store_true", help="one line per box")
    a = resolve(ap.parse_args(), ap)
    a.stop_file = a.stop_file or os.path.join(os.path.dirname(a.state_out) or ".", "STOP")
    os.makedirs(os.path.dirname(a.state_out) or ".", exist_ok=True)
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    rig = Rig(a)
    print(CONNECTED_MSG, flush=True)
    try:
        scan(rig, a)
    finally:
        rig.close()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
