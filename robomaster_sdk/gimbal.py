"""Gimbal of the RoboMaster EP, for pointing the camera and the IR sensor mounted on it.

What the DJI SDK does (robomaster/gimbal.py, robomaster/robot.py of robomaster 0.1.1.68):
  - gimbal.moveto(pitch, yaw, ...)  absolute target, yaw relative to the CHASSIS, pitch relative to the GROUND
                                    (coordinate mode YCPN); pitch in [-25, 30] deg, yaw in [-250, 250] deg
  - gimbal.move(pitch, yaw, ...)    relative to the current angles, each in [-55, 55] deg
  - gimbal.sub_angle(freq, cb)      cb((pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle)): the first
                                    two relative to the chassis, the last two to the power-on attitude [deg]
  - robot.set_robot_mode(mode)      "free": gimbal and chassis independent (needed here, so that pointing the
                                    gimbal never turns the chassis); "chassis_lead": the gimbal follows the chassis

Convention of this module: yaw positive to the LEFT (counter-clockwise seen from above, as the OptiTrack yaw),
pitch positive DOWN (as the camera pitch of yolo_gnn/camera.py), radians. YAW_SIGN / PITCH_SIGN turn it into the
SDK's; they are not documented by DJI and are checked on the robot with tools/gimbal_check.py.

The Robot class (robot.py) is not changed: from_robot() takes the gimbal of the DJI robot object it wraps.

    from robomaster_sdk import Robot
    from robomaster_sdk.gimbal import Gimbal
    with Robot("P0") as r:
        g = Gimbal.from_robot(r).start()
        g.point(math.radians(20), math.radians(5))      # 20 deg left, 5 deg down
        print(g.angles())
        g.close()
"""

import math
import threading
import time

YAW_SIGN = -1.0      # SDK yaw = YAW_SIGN * yaw of this module [deg]   (to check: tools/gimbal_check.py)
PITCH_SIGN = -1.0    # SDK pitch = PITCH_SIGN * pitch of this module [deg]
PITCH_LIMITS_DEG = (-25.0, 30.0)    # SDK pitch range of moveto
YAW_LIMITS_DEG = (-250.0, 250.0)


class Gimbal:
    def __init__(self, gimbal, ep=None, freq=20, yaw_sign=YAW_SIGN, pitch_sign=PITCH_SIGN, free_mode=True,
                 restore_mode="chassis_lead"):
        self.g, self.ep, self.freq = gimbal, ep, freq
        self.yaw_sign, self.pitch_sign = yaw_sign, pitch_sign
        self.free_mode, self.restore_mode = free_mode, restore_mode
        self.lock = threading.Lock()
        self.last = None            # (time, pitch, yaw, pitch_ground, yaw_ground) as the SDK gives them [deg]
        self.first_raw = None
        self.running = False

    @classmethod
    def from_robot(cls, robot, **kw):
        ep = getattr(robot, "_ep", robot)
        if ep is None or not hasattr(ep, "gimbal"):
            raise RuntimeError("robot not connected, or no gimbal")
        return cls(ep.gimbal, ep=ep, **kw)

    def _on_angle(self, info):
        if self.first_raw is None:
            self.first_raw = repr(info)
        try:
            p, y, pg, yg = (float(v) for v in info)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.last = (time.time(), p, y, pg, yg)

    def start(self):
        if self.free_mode and self.ep is not None:
            self.ep.set_robot_mode(mode="free")
        self.g.sub_angle(freq=self.freq, callback=self._on_angle)
        self.running = True
        t0 = time.time()
        while self.last is None and time.time() - t0 < 2.0:
            time.sleep(0.05)
        return self

    def raw(self):
        with self.lock:
            return self.last

    def angles(self):
        """(yaw, pitch) [rad] relative to the chassis, this module's convention, or None."""
        r = self.raw()
        if r is None:
            return None
        return math.radians(self.yaw_sign * r[2]), math.radians(self.pitch_sign * r[1])

    def point(self, yaw, pitch, speed_deg_s=90.0, tol_deg=0.5, timeout_s=4.0, corrections=2):
        """Points the gimbal to (yaw, pitch) [rad, this module's convention] relative to the chassis: moveto,
        then up to `corrections` relative moves on the residual read back from sub_angle. Returns the residual
        (yaw, pitch) [deg], or None without angle readings."""
        sy = max(YAW_LIMITS_DEG[0], min(YAW_LIMITS_DEG[1], self.yaw_sign * math.degrees(yaw)))
        sp = max(PITCH_LIMITS_DEG[0], min(PITCH_LIMITS_DEG[1], self.pitch_sign * math.degrees(pitch)))
        self.g.moveto(pitch=sp, yaw=sy, pitch_speed=speed_deg_s, yaw_speed=speed_deg_s).wait_for_completed(timeout_s)
        res = None
        for _ in range(corrections + 1):
            time.sleep(0.15)
            r = self.raw()
            if r is None:
                return None
            ey, ep = sy - r[2], sp - r[1]
            res = (self.yaw_sign * ey, self.pitch_sign * ep)
            if abs(ey) <= tol_deg and abs(ep) <= tol_deg:
                break
            self.g.move(pitch=max(-55, min(55, ep)), yaw=max(-55, min(55, ey)), pitch_speed=speed_deg_s / 3,
                        yaw_speed=speed_deg_s / 3).wait_for_completed(timeout_s)
        return res

    def start_move(self, yaw, pitch, speed_deg_s=10.0):
        """Starts a move to (yaw, pitch) [rad, this module's convention] at speed_deg_s and returns at once (the
        angles are read back with angles() while it moves: a sweep that samples on the way)."""
        sy = max(YAW_LIMITS_DEG[0], min(YAW_LIMITS_DEG[1], self.yaw_sign * math.degrees(yaw)))
        sp = max(PITCH_LIMITS_DEG[0], min(PITCH_LIMITS_DEG[1], self.pitch_sign * math.degrees(pitch)))
        return self.g.moveto(pitch=sp, yaw=sy, pitch_speed=speed_deg_s, yaw_speed=speed_deg_s)

    def recenter(self, timeout_s=4.0):
        self.g.recenter().wait_for_completed(timeout_s)

    def close(self, recenter=True):
        try:
            if recenter:
                self.recenter()
        finally:
            if self.running:
                self.g.unsub_angle()
                self.running = False
            if self.free_mode and self.restore_mode and self.ep is not None:
                self.ep.set_robot_mode(mode=self.restore_mode)

