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

With --turns (as acquisition/ir_live_robots.py starts it) the robots take turns, one box each: the IR cannot be
switched off from the SDK, so while another robot ranges, the gimbal points the beam upwards (--park-up-deg) and
no two beams meet on a box. Every state is also appended to <live dir>/log/<robot>.jsonl (tools/ir_view.py).

The aiming uses the OptiTrack positions of the boxes (the camera plays no part yet): the robot ranges every box,
including one hidden behind another (then the IR reads the nearer one). A reading that is not valid writes a state
with no detection.

    python -m acquisition.ir_scan --robot-id G11 --rb 8 --state-out data/live_ir/G11.json
"""

import json
import math
import os
import signal
import time

from acquisition.ir_live import STOP, Rig, make_parser, resolve
from acquisition.ir_ranging import aim_angles
from demo.methods import read_json, write_json
from vf.control.goto import wrap_pi

CONNECTED_MSG = "IR scan running"          # acquisition/ir_live_robots.py waits for this before the next robot


def state(a, pose, detections, extra):
    return dict(robot_id=a.robot_id, t=time.time(), pose=[float(v) for v in pose], detections=detections, p_d=1.0,
                source="ir", **extra)


def publish(a, st, boxes=None, body=None):
    """The live state for demo/live.py, and the same state (with the robot pose and the true boxes) appended to
    <live dir>/log/<robot>.jsonl for tools/ir_view.py and the analysis."""
    write_json(a.state_out, st)
    os.makedirs(a.log_dir, exist_ok=True)
    with open(os.path.join(a.log_dir, f"{a.robot_id}.jsonl"), "a") as f:
        f.write(json.dumps(dict(st, body=None if body is None else [float(v) for v in body], boxes=boxes)) + "\n")


class Turns:
    """Turn taking set by acquisition/ir_live_robots.py: <dir>/turn.json {"robot", "seq"} says whose turn it is;
    the robot ranges ONE box and answers in <dir>/<robot>.done.json {"seq"}. Without --turns every robot ranges
    when it likes (beams of several robots can then meet on one box)."""

    def __init__(self, a):
        self.a, self.on = a, a.turns
        self.dir = os.path.dirname(a.state_out) or "."
        self.seq = None

    def wait(self):
        """Blocks until it is this robot's turn (True), or the run stops (False)."""
        if not self.on:
            return True
        while not STOP.is_set() and not os.path.exists(self.a.stop_file):
            t = read_json(os.path.join(self.dir, "turn.json"))
            if t and t.get("robot") == self.a.robot_id and t.get("seq") != self.seq:
                self.seq = t["seq"]
                return True
            time.sleep(0.02)
        return False

    def done(self):
        if self.on:
            write_json(os.path.join(self.dir, f"{self.a.robot_id}.done.json"), dict(seq=self.seq, t=time.time()))


def sweep_order(rig, a, n_sweep):
    body, boxes = rig.body_pose(), rig.boxes()
    if body is None:
        return body, []
    boxes = [b for b in boxes if math.hypot(b["x"] - body[0], b["y"] - body[1]) <= a.max_range]
    boxes.sort(key=lambda b: wrap_pi(math.atan2(b["y"] - body[1], b["x"] - body[0]) - body[2]))
    if n_sweep % 2:
        boxes.reverse()                                       # back and forth: the gimbal never makes a long jump
    return body, boxes


def park(rig, a):
    """Beam away from the boxes (towards the ceiling) while another robot ranges: the IR cannot be switched off."""
    if a.turns:
        ang = rig.gimbal.angles() or (0.0, 0.0)
        rig.gimbal.point(ang[0], math.radians(-a.park_up_deg))


def range_box(rig, a, b, body):
    rig.point(*aim_angles(body, (b["x"], b["y"]), rig.mount))
    r = rig.sample(a.ir_samples)
    if r.get("ir_x") is None:
        return None
    pose = (r["ir_x"], r["ir_y"], math.radians(r["ir_heading_deg"]))
    dets = []
    if r["ir_m"] is not None and r.get("ir_horiz_m") is not None and 0.0 <= r["ir_hit_height_m"] <= 0.6:
        d = r["ir_horiz_m"]
        dets.append(dict(class_name="box", confidence=1.0, distance=d, bearing_rad=0.0, viewing_angle_rad=0.0,
                         position=[pose[0] + d * math.cos(pose[2]), pose[1] + d * math.sin(pose[2])], bbox=None))
    truth = rig.target_truth(r, b)
    st = state(a, pose, dets, dict(
        aimed_box=b["id"], ir_raw_m=r.get("ir_raw_m"), ir_slant_m=r["ir_m"], ir_hit_height_m=r.get("ir_hit_height_m"),
        ir_valid_n=r["ir_n"], expected_box=r.get("exp_box"), true_horiz_m=truth.get("target_horiz_m"),
        gimbal=[r.get("gimbal_yaw_deg"), r.get("gimbal_pitch_deg")]))
    if a.verbose:
        err = None if not dets or truth.get("target_horiz_m") is None else dets[0]["distance"] - truth["target_horiz_m"]
        print(f"box {b['id']}: " + ("no valid IR" if not dets else f"{dets[0]['distance']:.3f} m"
                                    + ("" if err is None else f" (error {err * 100:+.1f} cm)")), flush=True)
    return st


def scan(rig, a):
    turns = Turns(a)
    n_sweep, queue, body = 0, [], None
    park(rig, a)
    while not STOP.is_set() and not os.path.exists(a.stop_file):
        if not turns.wait():
            break
        if not queue:
            body, queue = sweep_order(rig, a, n_sweep)
            n_sweep += 1
        if body is None or not queue:
            if body is not None:
                ang = rig.gimbal.angles() or (0.0, 0.0)
                publish(a, state(a, (body[0], body[1], wrap_pi(body[2] + ang[0])), [], dict(note="no box within range")),
                        boxes=rig.boxes(), body=body)
            turns.done()
            time.sleep(0.05 if a.turns else 0.5)
            continue
        b = queue.pop(0)
        body = rig.body_pose() or body
        st = range_box(rig, a, b, body)
        park(rig, a)
        if st is not None:
            publish(a, st, boxes=rig.boxes(), body=body)
        turns.done()


def main():
    ap = make_parser(with_mode=False, description=__doc__)
    ap.add_argument("--state-out", required=True, help="live state file of this robot (demo/live.py --live-dir)")
    ap.add_argument("--max-range", type=float, default=3.5, help="only boxes closer than this [m]")
    ap.add_argument("--stop-file", default="", help="the scan stops when this file exists")
    ap.add_argument("--verbose", action="store_true", help="one line per box")
    ap.add_argument("--turns", action="store_true",
                    help="range only when <live dir>/turn.json gives this robot the turn (acquisition/ir_live_robots.py)")
    ap.add_argument("--park-up-deg", type=float, default=20.0,
                    help="with --turns: between its turns the gimbal points the beam this much upwards")
    a = resolve(ap.parse_args(), ap)
    a.stop_file = a.stop_file or os.path.join(os.path.dirname(a.state_out) or ".", "STOP")
    a.log_dir = os.path.join(os.path.dirname(a.state_out) or ".", "log")
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
