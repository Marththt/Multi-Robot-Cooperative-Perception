"""One robot of the fleet in the lab: drives, fans or aims its gimbal, and reports what its IR and camera see.

Started by fleet/lab.py (one process per robot, one after the other: the SDK finds a robot through a port only
one process at a time can use). It talks to the brain through files in --dir:

  goals/<robot>.json      written by the brain (atomic), read here every 0.1 s:
                          {"seq", "t", "target": [x, y] or null, "yaw": rad or null, "v_max": m/s or null,
                           "gimbal": {"mode": "fan"} or {"mode": "aim", "task": id, "xy": [x, y]}}
  reports/<robot>.jsonl   appended here, one JSON per line, read by the brain:
                          {"type": "status", "t", "pose": [x, y, yaw], "gy": gimbal yaw [rad], "mode", "task"}
                          {"type": "ir", "t", "o": [x, y], "h": heading, "d": horizontal range [m] or null}
                                d null = the beam went beyond --ir-max-m (free space up to there)
                          {"type": "cam", "t", "lens": [x, y, yaw], "rays": [[world angle, monocular range], ...]}
                          {"type": "aim", "t", "task", "o", "h", "d"}   the aimed reading (median of --aim-samples)
  STOP                    the node stops the chassis, stops and re-centres the gimbal, and exits

Every loop (20 Hz):
  chassis   the controller of the captures (vf/control/goto.compute_command, same signs and gains) towards the
            goal target and yaw, at most --v-max; nothing without a goal, with --no-drive, or without an
            OptiTrack pose (the SDK watchdog stops the robot when no command arrives). Safety, OptiTrack only
            (never used by the perception): a rigid body (box, robot) within --bumper-m of the robot in the
            direction it moves stops the translation; so does leaving the arena (--regions) by more than 0.1 m.
  gimbal    fan: speed control (Gimbal.drive, never an SDK action) back and forth between -fan and +fan around
            the chassis heading, beam pitched --fan-pitch-deg down; aim: the chassis stops, the gimbal points the
            IR at the box (acquisition/ir_ranging.aim_angles, mid-face), waits --settle-s, reads --aim-samples
            samples, reports their median, and the fan goes on.
  IR        the newest sample of the IR (with the gimbal angles and the pose at that moment) as a beam: range
            corrected (acquisition/ir_live.IR_FIT), turned into a horizontal range with the beam pitch; a beam
            that meets the floor or reads beyond --ir-max-m reports free space only.
  camera    a thread: frame -> YOLO -> one ray per box (acquisition/ir_ranging.YoloRays), --no-camera to leave it out.

The gimbal aiming offset of the robot (acquisition/ir_live.YAW_OFFSET_DEG) is applied to the gimbal angles, not
to the chassis heading; --calibrate measures it at start with a yaw sweep around one box (about 40 s).

Without hardware (--sim --sim-world <world.json> --sim-pose X Y YAW_DEG): the chassis integrates its commands
(vf/control/goto.FakeRobot), the gimbal is acquisition/ir_ranging.FakeGimbal, the IR and the camera see the scene
of the world file (fleet/sensing.ir_beam and look).

    python -m fleet.robot_node --robot-id G11 --rb 8 --dir data/fleet_lab
    python -m fleet.robot_node --robot-id S1 --sim --sim-world data/fleet_lab/world.json --sim-pose -2.2 -0.6 0 \\
        --dir data/fleet_lab
"""

import json
import math
import os
import signal
import threading
import time

import numpy as np

from acquisition.ir_live import STOP, make_parser, resolve
from acquisition.ir_ranging import aim_angles, camera_pose, ir_pose, mount_from_args, slant_to_floor
from vf.control.goto import SIGN_X, SIGN_Y, SIGN_Z, compute_command, wrap_pi

CONNECTED_MSG = "fleet node running"       # fleet/lab.py waits for this before it starts the next robot
RATE_HZ = 20.0
DEFAULT_CAMERA_CALIB = "config/checkerboard_calib_P0.json"


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Reports:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def put(self, **row):
        row.setdefault("t", time.time())
        with self.lock:
            self.f.write(json.dumps(row) + "\n")

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------------------------- hardware
class RealRobot:
    """The robot, its gimbal, IR and camera, OptiTrack (acquisition/ir_live.Rig, unchanged), plus its chassis."""

    def __init__(self, a):
        from acquisition.ir_live import Rig
        self.a = a
        self.rig = Rig(a)
        self.mount = self.rig.mount
        self.gimbal, self.ir, self.chassis = self.rig.gimbal, self.rig.ir, self.rig.robot.chassis
        self.yolo = None
        if not a.no_camera:
            self.yolo = self.rig.start_camera()
        self.last_ir_t = None

    def body(self):
        """Chassis pose (x, y, yaw) from OptiTrack, WITHOUT the gimbal aiming offset."""
        q = self.rig.hub.get_pose(self.a.rb)
        return None if q is None else (float(q[0]), float(q[1]), float(q[2]))

    def gimbal_angles(self):
        """(yaw, pitch) of the gimbal relative to the chassis as the IR and the camera really point (the aiming
        offset added), or None."""
        ang = self.gimbal.angles()
        return None if ang is None else (wrap_pi(ang[0] + self.rig.yaw_off), ang[1])

    def point(self, yaw, pitch):
        self.gimbal.point(wrap_pi(yaw - self.rig.yaw_off), pitch)

    def ir_sample(self):
        """(time, corrected slant range or None) of the newest IR sample not given yet, or None."""
        s = self.ir.latest()
        if s is None or s[0] == self.last_ir_t:
            return None
        self.last_ir_t = s[0]
        v = s[1][self.mount.ir_index] if self.mount.ir_index < len(s[1]) else 0.0
        if not self.ir.valid(v):
            return s[0], None
        k, q = getattr(self.ir, "k", 1.0), getattr(self.ir, "q", 0.0)
        return s[0], (v - q) / k

    def ir_measure(self, n):
        return self.ir.measure(self.mount.ir_index, n=n)["m"]

    def others(self):
        """(x, y) of every other rigid body OptiTrack tracks (boxes, robots): for the bumper only."""
        out = []
        for rb in self.rig.hub.tracked():
            if rb == self.a.rb:
                continue
            q = self.rig.hub.get_pose(rb)
            if q is not None:
                out.append((float(q[0]), float(q[1])))
        return out

    def rays(self):
        lens, rays, _ = self.yolo.rays()
        return lens, rays

    def drive(self, vx, vy, wz):
        self.chassis.drive_speed(x=SIGN_X * vx, y=SIGN_Y * vy, z=SIGN_Z * math.degrees(wz), timeout=0.3)

    def close(self):
        try:
            self.drive(0.0, 0.0, 0.0)
        except Exception:
            pass
        self.rig.close()


class SimRobot:
    """Fakes with the same interface: chassis vf/control/goto.FakeRobot, acquisition/ir_ranging.FakeGimbal, the
    IR and the camera on the scene of a fleet world file."""

    def __init__(self, a):
        from acquisition.ir_ranging import FakeGimbal
        from fleet.sensing import CameraModel, GimbalIR
        from fleet.world import Grid, World
        from vf.control.goto import FakeRobot
        self.a = a
        self.mount = mount_from_args(a)
        self.world = World.load(a.sim_world)
        self.grid = Grid(self.world.arena, 0.05)
        self.ids = self.world.ids(self.grid)
        x, y, yaw = a.sim_pose
        self.chassis = FakeRobot(x, y, math.radians(yaw))
        self.gimbal = FakeGimbal()
        self.rng = np.random.default_rng(a.seed)
        self.irm = GimbalIR(r_max_m=a.ir_max_m)
        self.cam = CameraModel(sd_range_m=(0.02, 0.15), fp_rate=0.1, p_d_near=0.85, p_d_far=0.57)
        self.yolo = None if a.no_camera else True
        self.last_ir_t = 0.0

    def body(self):
        return self.chassis.pose()

    def gimbal_angles(self):
        return self.gimbal.angles()

    def point(self, yaw, pitch):
        self.gimbal.point(yaw, pitch)

    def _beam(self):
        from fleet.sensing import ir_beam
        body, (gy, gp) = self.body(), self.gimbal.angles()
        o, h, _ = ir_pose(body, gy, gp, self.mount)
        d, _ = ir_beam(self.grid, self.ids, o, h, self.irm.r_max_m + 0.5, [], self.rng, self.irm)
        return None if d is None else d / math.cos(gp)          # the sensor reads a slant range

    def ir_sample(self):
        now = time.time()
        if now - self.last_ir_t < 0.8 / RATE_HZ:
            return None
        self.last_ir_t = now
        return now, self._beam()

    def ir_measure(self, n):
        v = [self._beam() for _ in range(n)]
        v = [x for x in v if x is not None]
        return float(np.median(v)) if len(v) * 2 > n else None

    def others(self):
        return []

    def rays(self):
        from fleet.sensing import look
        time.sleep(0.15)
        body, (gy, gp) = self.body(), self.gimbal.angles()
        lens, _ = camera_pose(body, gy, gp, self.mount)
        fr = look(self.cam, self.world, self.grid, self.ids, self.a.robot_id, (body[0], body[1], lens[2]), [],
                  self.rng, depth=False)
        return lens, [(wrap_pi(lens[2] + d["bearing"]), d["range"]) for d in fr.detections]

    def drive(self, vx, vy, wz):
        self.chassis.drive_speed(x=SIGN_X * vx, y=SIGN_Y * vy, z=SIGN_Z * math.degrees(wz))

    def close(self):
        self.drive(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------- node
class Node:
    def __init__(self, a, hw):
        self.a, self.hw = a, hw
        self.rep = Reports(os.path.join(a.dir, "reports", f"{a.robot_id}.jsonl"))
        self.goal_path = os.path.join(a.dir, "goals", f"{a.robot_id}.json")
        self.stop_path = os.path.join(a.dir, "STOP")
        self.goal, self.goal_seq = None, None
        self.fan_dir, self.fan_sent = 1.0, 0.0
        self.mode, self.task, self.done_tasks = "fan", None, set()
        self.arena = None
        if a.regions and not a.sim:
            from vf.fusion.sensor_model import arena_from_regions
            self.arena = arena_from_regions(a.regions)
        self.bump_said = 0.0

    def stopped(self):
        return STOP.is_set() or os.path.exists(self.stop_path)

    def read_goal(self):
        g = read_json(self.goal_path)
        if g and g.get("seq") != self.goal_seq:
            self.goal, self.goal_seq = g, g.get("seq")

    # ---------------------------------------------------------------- chassis
    def drive_step(self, body):
        g = self.goal
        if body is None or g is None or g.get("target") is None or self.a.no_drive or \
                time.time() - float(g.get("t", 0)) > 3.0:          # a brain that stopped writing: stand still
            self.hw.drive(0.0, 0.0, 0.0)
            return
        v_max = min(self.a.v_max, g.get("v_max") or self.a.v_max)
        vx, vy, wz, d, _ = compute_command(body, g["target"], g.get("yaw"), v_max)
        if d < 0.03:
            vx = vy = 0.0
        if vx or vy:
            c, s = math.cos(body[2]), math.sin(body[2])
            wx, wy = c * vx - s * vy, s * vx + c * vy                  # world direction of the motion
            why = self.bumper(body, wx, wy)
            if why:
                vx = vy = 0.0
                if time.time() - self.bump_said > 2.0:
                    print(f"bumper: {why}, translation stopped", flush=True)
                    self.bump_said = time.time()
        self.hw.drive(vx, vy, wz)

    def bumper(self, body, wx, wy):
        for ox, oy in self.hw.others():
            dx, dy = ox - body[0], oy - body[1]
            d = math.hypot(dx, dy)
            if d < self.a.bumper_m and dx * wx + dy * wy > 0:
                return f"something tracked {d:.2f} m ahead"
        if self.arena is not None:
            x0, x1, y0, y1 = self.arena
            nx, ny = body[0] + wx * 0.5, body[1] + wy * 0.5
            if not (x0 - 0.1 <= nx <= x1 + 0.1 and y0 - 0.1 <= ny <= y1 + 0.1):
                return "arena border"
        return None

    # ---------------------------------------------------------------- gimbal
    def fan_step(self, ang):
        lim = math.radians(self.a.fan_deg)
        if ang is not None and ((ang[0] > lim and self.fan_dir > 0) or (ang[0] < -lim and self.fan_dir < 0)):
            self.fan_dir = -self.fan_dir
            self.fan_sent = 0.0
        if time.time() - self.fan_sent > 0.2:                   # re-sent: a lost message must not stop the fan
            self.hw.gimbal.drive(self.fan_dir * self.a.fan_dps)
            self.fan_sent = time.time()

    def aim(self, task, xy):
        """Chassis stopped, gimbal on the box, settle, aimed reading, report; then the fan goes on."""
        self.mode, self.task = "aim", task
        self.hw.drive(0.0, 0.0, 0.0)
        for _ in range(3):
            self.hw.gimbal.drive(0.0)
            time.sleep(0.03)
        body = self.hw.body()
        if body is None:
            return
        yaw, pitch = aim_angles(body, xy, self.hw.mount)
        self.status(body)
        self.hw.point(yaw, pitch)
        time.sleep(self.a.settle_s)
        body, ang = self.hw.body(), self.hw.gimbal_angles()
        d = self.hw.ir_measure(self.a.aim_samples)
        row = dict(type="aim", task=task, o=None, h=None, d=None)
        if body is not None and ang is not None:
            o, h, p = ir_pose(body, ang[0], ang[1], self.hw.mount)
            row.update(o=[float(o[0]), float(o[1])], h=float(h))
            if d is not None:
                horiz, height = slant_to_floor(d, p, self.hw.mount.ir_height_m)
                row["d"] = float(horiz) if height > 0.02 else None
        self.rep.put(**row)
        if self.a.verbose:
            print(f"aim task {task}: {row['d']}", flush=True)
        self.done_tasks.add(task)
        self.hw.point(yaw, math.radians(self.a.fan_pitch_deg))
        self.mode, self.task, self.fan_sent = "fan", None, 0.0

    # ---------------------------------------------------------------- reports
    def ir_step(self, body, ang):
        s = self.hw.ir_sample()
        if s is None or body is None or ang is None:
            return
        _, d = s
        o, h, p = ir_pose(body, ang[0], ang[1], self.hw.mount)
        if d is None:
            return                                   # no echo: unknown (too close, or nothing), not free space
        horiz, height = slant_to_floor(d, p, self.hw.mount.ir_height_m)
        if height <= 0.02 or horiz > self.a.ir_max_m:
            horiz = None                             # floor, or beyond the range used: free up to ir_max_m
        self.rep.put(type="ir", o=[float(o[0]), float(o[1])], h=float(h), d=horiz)

    def status(self, body):
        ang = self.hw.gimbal_angles()
        self.rep.put(type="status", pose=None if body is None else [float(v) for v in body],
                     gy=None if ang is None else float(ang[0]), mode=self.mode, task=self.task)

    def camera_loop(self):
        while not self.stopped():
            try:
                lens, rays = self.hw.rays()
            except Exception as e:
                print(f"camera: {e}", flush=True)
                time.sleep(0.5)
                continue
            if lens is not None:
                self.rep.put(type="cam", lens=[float(v) for v in lens],
                             rays=[[float(a), float(d)] for a, d in rays])

    # ---------------------------------------------------------------- main loop
    def run(self):
        self.hw.point(0.0, math.radians(self.a.fan_pitch_deg))
        if self.hw.yolo is not None:
            threading.Thread(target=self.camera_loop, daemon=True).start()
        print(CONNECTED_MSG, flush=True)
        period, next_goal, next_status = 1.0 / RATE_HZ, 0.0, 0.0
        while not self.stopped():
            t0 = time.time()
            if t0 >= next_goal:
                self.read_goal()
                next_goal = t0 + 0.1
            g = (self.goal or {}).get("gimbal") or {"mode": "fan"}
            if g.get("mode") == "aim" and g.get("task") not in self.done_tasks:
                self.aim(g["task"], g["xy"])
                continue
            body, ang = self.hw.body(), self.hw.gimbal_angles()
            self.drive_step(body)
            self.fan_step(ang)
            self.ir_step(body, ang)
            if t0 >= next_status:
                self.status(body)
                next_status = t0 + 0.1
            time.sleep(max(0.0, period - (time.time() - t0)))

    def close(self):
        try:
            self.hw.drive(0.0, 0.0, 0.0)
            for _ in range(3):
                self.hw.gimbal.drive(0.0)
                time.sleep(0.05)
        finally:
            self.hw.close()
            self.rep.close()


def main():
    ap = make_parser(with_mode=False, description=__doc__)
    ap.add_argument("--dir", default="data/fleet_lab", help="folder shared with fleet/lab.py")
    ap.add_argument("--no-drive", action="store_true", help="never move the chassis (sensing and planning only)")
    ap.add_argument("--no-camera", action="store_true", help="no camera / YOLO (map from the IR only)")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the gimbal aiming offset at start (a yaw sweep around one box, ~40 s)")
    ap.add_argument("--v-max", type=float, default=0.15, help="speed limit [m/s]")
    ap.add_argument("--bumper-m", type=float, default=0.38,
                    help="a tracked rigid body closer than this (centre to centre) in the direction of motion stops "
                         "the translation")
    ap.add_argument("--fan-deg", type=float, default=40.0)
    ap.add_argument("--fan-dps", type=float, default=30.0)
    ap.add_argument("--fan-pitch-deg", type=float, default=4.0,
                    help="beam pitched down this much while fanning (meets the floor at 0.30 / tan: 4.3 m at 4 deg)")
    ap.add_argument("--ir-max-m", type=float, default=2.5, help="horizontal range of the IR used for the map")
    ap.add_argument("--aim-samples", type=int, default=5)
    ap.add_argument("--sim-world", default=None, help="--sim: the scene (fleet/world.World JSON)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.sim and not a.sim_world:
        ap.error("--sim needs --sim-world")
    if not a.sim and a.camera_calib is None and not a.no_camera:
        a.camera_calib = DEFAULT_CAMERA_CALIB
    a = resolve(a, ap)
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    hw = SimRobot(a) if a.sim else RealRobot(a)
    if a.calibrate and not a.sim:
        from acquisition.ir_live import estimate_offset
        estimate_offset(hw.rig)
    node = Node(a, hw)
    try:
        node.run()
    finally:
        node.close()
        print("stopped", flush=True)


if __name__ == "__main__":
    main()
