"""Closed-loop motion of a RoboMaster EP to a planar pose, with OptiTrack.

Proportional controller tested on the robots, with its limits and safety
stops. For the autonomous survey (acquisition/survey.py):

  - go_to() takes the pose as a callable and returns whether the goal was
    reached, so a caller can chain many goals;
  - an optional `others` callable gives the positions of the other robots:
    if one is closer than `stop_distance_m` and lies in the direction of
    motion, the robot stands still (the runtime clock keeps running), so
    two robots never drive into each other (moving away is always allowed);
  - FakeRobot integrates the commands, to test everything without robot
    and without OptiTrack.

Command signs (SIGN_X, SIGN_Y, SIGN_Z): SIGN_X and SIGN_Z measured on the
robots; SIGN_Y = -1 measured with tools/check_motion.py
(with +1 a "move left" command moved the robot to the right).
"""

import math
import time

POS_TOL_M = 0.05
YAW_TOL_RAD = math.radians(5.0)

KP_LIN = 1.2
KP_YAW = 2.0
V_MAX = 0.5                # m/s
W_MAX_DEG = 90.0           # deg/s

RATE_HZ = 20.0
CMD_TIMEOUT_S = 0.3        # robot watchdog: no new command -> it stops
MAX_RUNTIME_S = 30.0       # per goal
MAX_LOST_FRAMES = 10       # consecutive missing poses -> abort the goal
DIVERGENCE_M = 0.3         # farther than this from the closest point reached -> abort
YAW_DIVERGENCE_DEG = 30.0  # heading error this much larger than the best reached -> abort

SIGN_X = +1.0
SIGN_Y = -1.0   # +1 sent the robot RIGHT: measured with tools/check_motion.py
SIGN_Z = -1.0


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def compute_command(pose, goal_xy, goal_yaw, v_max=V_MAX):
    """Current pose (x, y, yaw) + goal -> velocities in the ROBOT frame.
    Returns (vx, vy, wz, dist, yaw_err): vx forward, vy left [m/s], wz [rad/s]."""
    x, y, yaw = pose
    ex, ey = goal_xy[0] - x, goal_xy[1] - y
    dist = math.hypot(ex, ey)
    c, s = math.cos(yaw), math.sin(yaw)
    vx = KP_LIN * (c * ex + s * ey)
    vy = KP_LIN * (-s * ex + c * ey)
    speed = math.hypot(vx, vy)
    if speed > v_max:            # saturation that keeps the direction
        vx, vy = vx * v_max / speed, vy * v_max / speed
    yaw_err, wz = 0.0, 0.0
    if goal_yaw is not None:
        yaw_err = wrap_pi(goal_yaw - yaw)
        w_max = math.radians(W_MAX_DEG)
        wz = max(-w_max, min(w_max, KP_YAW * yaw_err))
    return vx, vy, wz, dist, yaw_err


def reached(dist, yaw_err, goal_yaw, pos_tol=POS_TOL_M, yaw_tol=YAW_TOL_RAD):
    if dist > pos_tol:
        return False
    return goal_yaw is None or abs(yaw_err) <= yaw_tol


def blocked(pose, goal_xy, others_xy, stop_distance_m):
    """True if another robot is closer than stop_distance_m AND lies ahead,
    i.e. in the half-plane the robot is moving towards. A robot that is
    near but behind does not block: moving away from it is always allowed,
    so two robots that ended up close can still separate."""
    ex, ey = goal_xy[0] - pose[0], goal_xy[1] - pose[1]
    for ox, oy in others_xy:
        dx, dy = ox - pose[0], oy - pose[1]
        if math.hypot(dx, dy) < stop_distance_m and dx * ex + dy * ey > 0:
            return True
    return False


def stop(chassis):
    if chassis is None:
        return
    try:
        chassis.drive_speed(x=0, y=0, z=0, timeout=1)
    except Exception:
        pass


def go_to(pose_fn, chassis, goal_xy, goal_yaw=None, others=None,
          stop_distance_m=0.6, max_runtime_s=MAX_RUNTIME_S, v_max=V_MAX,
          fence=None, pos_tol=POS_TOL_M, cancel=None, log=print):
    """Drives the robot to goal_xy (and goal_yaw, None = any heading).

    pose_fn() -> (x, y, yaw) of the robot in the OptiTrack frame, or None.
    chassis: robomaster chassis (drive_speed), or a FakeRobot.
    others() -> [(x, y), ...] positions of the other robots (optional).
    fence(pose) -> False when the robot is where it must not be (e.g. too
    close to the arena border): the robot is stopped at once (optional).
    cancel() -> True to stop at once (e.g. Ctrl+C from the launcher).
    Returns True if the goal was reached, False on timeout, lost tracking,
    fence, or when the robot moves away from the goal instead of closing in
    (a wrong yaw or command sign: the robot is stopped at once)
    (the robot is stopped in every case)."""
    period = 1.0 / RATE_HZ
    t0 = time.time()
    lost = 0
    best = float("inf")
    best_yaw = float("inf")
    last = dict(dist=None, yaw_err=0.0)
    waited = 0
    try:
        while True:
            loop_start = time.time()
            if cancel is not None and cancel():
                log("[STOP] stop requested")
                return False
            if loop_start - t0 > max_runtime_s:
                why = ""
                if last["dist"] is not None:
                    why = (f" (still {last['dist']:.2f} m and {abs(math.degrees(last['yaw_err'])):.0f} deg "
                           f"from the goal")
                    why += (f"; {waited * period:.0f} s waiting for another robot in the way)" if waited
                            else ")")
                log("[STOP] goal timeout" + why)
                return False
            pose = pose_fn()
            if pose is None:
                lost += 1
                if lost >= MAX_LOST_FRAMES:
                    log("[STOP] tracking lost")
                    return False
                time.sleep(period)    # no command: the watchdog stops the robot
                continue
            lost = 0
            if fence is not None and not fence(pose):
                log("[STOP] outside the allowed area")
                return False
            vx, vy, wz, dist, yaw_err = compute_command(pose, goal_xy, goal_yaw, v_max)
            if dist <= pos_tol:
                vx = vy = 0.0          # close enough: only turn
            if reached(dist, yaw_err, goal_yaw, pos_tol=pos_tol):
                return True
            last.update(dist=dist, yaw_err=yaw_err)
            best = min(best, dist)
            if dist > best + DIVERGENCE_M:
                log("[STOP] moving AWAY from the goal: wrong yaw or command signs "
                    "(run tools/check_motion.py)")
                return False
            if goal_yaw is not None:
                best_yaw = min(best_yaw, abs(yaw_err))
                if abs(yaw_err) > best_yaw + math.radians(YAW_DIVERGENCE_DEG):
                    log("[STOP] turning the WRONG way: rotation sign or yaw of this robot "
                        "(run tools/check_motion.py)")
                    return False
            if others is not None and blocked(pose, goal_xy, others(), stop_distance_m):
                vx = vy = 0.0          # wait for the other robot to pass; still turn on the spot
                waited += 1
            chassis.drive_speed(x=SIGN_X * vx, y=SIGN_Y * vy, z=SIGN_Z * math.degrees(wz),
                                timeout=CMD_TIMEOUT_S)
            left = period - (time.time() - loop_start)
            if left > 0:
                time.sleep(left)
    finally:
        stop(chassis)


class FakeRobot:
    """Integrates the drive_speed commands (with the same signs as the real
    robot): pose() / drive_speed(), to test the motion without hardware.
    speedup > 1 makes simulated time run faster than the wall clock; keep
    it below 1 / (KP_YAW / RATE_HZ) = 10, otherwise the discrete
    controller overshoots the heading and never settles."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, speedup=1.0):
        self.x, self.y, self.yaw = x, y, yaw
        self._cmd = (0.0, 0.0, 0.0)
        self._t = time.time()
        self.speedup = speedup

    def _advance(self):
        now = time.time()
        dt = (now - self._t) * self.speedup
        self._t = now
        vx, vy, wz = self._cmd
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw = wrap_pi(self.yaw + wz * dt)

    def pose(self):
        self._advance()
        return (self.x, self.y, self.yaw)

    def drive_speed(self, x=0.0, y=0.0, z=0.0, timeout=None):
        self._advance()
        self._cmd = (x / SIGN_X, y / SIGN_Y, math.radians(z / SIGN_Z))