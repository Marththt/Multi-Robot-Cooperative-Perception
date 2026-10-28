"""Robot pose providers: on-board odometry, camera offset and frame correction.

OdometryLocalizer uses the position estimate of the RoboMaster SDK (wheels + IMU, chassis.sub_position),
relative to the power-on point, as an alternative to OptiTrack (optitrack.py, the reference used in
the experiments). Zones are (x, y) regions in metres, defined once per deployment in a config file
(config/optitrack_regions.json). Odometry drifts over time (wheel slip, no external correction): for
long runs, bring the robot back to a known point to reset it.

The signature of chassis.sub_position may differ between SDK versions (here: callback with an
(x, y, z) tuple, cs=0 = relative to the power-on point); check it against the installed SDK.
"""

import json
import logging
import math
import threading
from typing import List, NamedTuple, Tuple

log = logging.getLogger(__name__)


class ZoneRegion(NamedTuple):
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def load_regions(json_path) -> List[ZoneRegion]:
    with open(json_path) as f:
        data = json.load(f)
    return [ZoneRegion(name=r["name"], x_min=r["x_min"], x_max=r["x_max"],
                       y_min=r["y_min"], y_max=r["y_max"])
            for r in data["regions"]]


class OdometryLocalizer:
    """Tracks the full POSE (x, y, yaw) relative to the starting point via
    on-board odometry (wheels for x,y + IMU for yaw), and converts it into a
    zone name according to a map of rectangular regions (see
    ZoneRegion/load_regions). The yaw is used by object_position.py to
    estimate WHERE a detected object is, not just in which zone -- see the
    object_position.py module for why."""

    def __init__(self, robot, regions: List[ZoneRegion], freq=5,
                 unknown_zone="unknown"):
        self.robot = robot
        self.regions = regions
        self.unknown_zone = unknown_zone
        self._lock = threading.Lock()
        self._x, self._y, self._yaw_rad = 0.0, 0.0, 0.0
        robot.chassis.sub_position(cs=0, freq=freq, callback=self._on_position)
        # The RoboMaster SDK typically reports attitude in DEGREES
        # (yaw, pitch, roll) -- VERIFY this on your version before trusting
        # it: if your SDK gives radians, remove math.radians below.
        robot.chassis.sub_attitude(freq=freq, callback=self._on_attitude)

    def _on_position(self, info):
        x, y = info[0], info[1]
        with self._lock:
            self._x, self._y = x, y

    def _on_attitude(self, info):
        yaw_deg = info[0]
        with self._lock:
            self._yaw_rad = math.radians(yaw_deg)

    def position(self) -> Tuple[float, float]:
        with self._lock:
            return self._x, self._y

    def pose(self) -> Tuple[float, float, float]:
        """(x, y, yaw_rad) -- used by object_position.py to estimate the
        global position of a detected object."""
        with self._lock:
            return self._x, self._y, self._yaw_rad

    def current_zone(self) -> str:
        x, y = self.position()
        for region in self.regions:
            if (region.x_min <= x < region.x_max
                    and region.y_min <= y < region.y_max):
                return region.name
        return self.unknown_zone

    def close(self):
        try:
            self.robot.chassis.unsub_position()
        except Exception as e:
            log.debug("unsub_position failed: %s", e)
        try:
            self.robot.chassis.unsub_attitude()
        except Exception as e:
            log.debug("unsub_attitude failed: %s", e)


class CameraOffsetPose:
    """Wraps a pose provider (OdometryLocalizer or OptiTrackPose, same
    pose()/position()/current_zone()/close() interface) and SHIFTS the pose
    from the tracked point (the OptiTrack rigid body, or the odometry
    origin) to the TRUE position of the camera, before
    object_position.global_position() uses it to compute bearing/distance
    of a detected object.

    Why it is needed: OptiTrack (and odometry) report the pose of the
    tracked point -- typically the centre of the rigid body -- NOT the
    optical centre of the camera. If the camera is mounted far from that
    point (e.g. on top of an arm, as in this project), every computation
    that uses the raw pose as origin for bearing+distance inherits a
    constant position bias equal to the physical camera<->tracked-point
    offset. None of the calibrations done so far (focal length, FOV, ArUco
    parameters, OptiTrack axis reset) touch this offset: they correct the
    MEASUREMENT relative to the reference point, not the reference point
    itself.

    offset_forward_m / offset_right_m are measured BY HAND (a ruler, from
    the tracked point -- e.g. the rigid body centre -- to the lens), in the
    ROBOT's reference frame (not the arena's): how far the camera is ahead
    of the tracked point, and how far it is to the right (negative = to the
    left). With both at 0.0 (default) this class behaves EXACTLY like the
    provider it wraps -- zero risk of breaking anyone who does not have (or
    has not yet measured) the offset.

    Rotation: same convention as object_position.global_position()
    (global_angle = yaw - bearing, positive bearing = to the right, yaw
    increasing to the left) -- local "forward" maps to (cos(yaw), sin(yaw))
    in global coordinates, local "right" maps to (cos(yaw-90deg),
    sin(yaw-90deg)) = (sin(yaw), -cos(yaw)). (Previously this used
    yaw+90deg, i.e. the LEFT: the same sign error that was in
    global_position, fixed together with it.)

        dx = offset_forward_m*cos(yaw) + offset_right_m*sin(yaw)
        dy = offset_forward_m*sin(yaw) - offset_right_m*cos(yaw)

    The yaw is not corrected: the camera is assumed to point in the same
    direction as the chassis (no independent arm pan is modelled here).

    Assumption to keep in mind: the offset is constant only if the arm
    carrying the camera stays in a FIXED, repeatable pose between one
    capture and the next. If the arm moves (variable pan/tilt), a constant
    offset is no longer enough -- the arm's forward kinematics would be
    needed, which is not covered here."""

    def __init__(self, pose_provider, offset_forward_m: float = 0.0,
                 offset_right_m: float = 0.0):
        self.pose_provider = pose_provider
        self.offset_forward_m = offset_forward_m
        self.offset_right_m = offset_right_m

    def pose(self):
        """(x, y, yaw_rad) of the CAMERA (not of the tracked point), or None
        if the underlying provider does not have a valid pose (yet, or any
        more) -- NEVER invent a pose when tracking is missing."""
        raw_pose = self.pose_provider.pose()
        if raw_pose is None:
            return None
        x, y, yaw = raw_pose
        if self.offset_forward_m == 0.0 and self.offset_right_m == 0.0:
            return x, y, yaw
        dx = (self.offset_forward_m * math.cos(yaw)
              + self.offset_right_m * math.sin(yaw))
        dy = (self.offset_forward_m * math.sin(yaw)
              - self.offset_right_m * math.cos(yaw))
        return x + dx, y + dy, yaw

    def position(self):
        """(x, y) of the camera, or None -- delegates to pose() instead of
        the provider's raw position(), so the offset is applied here too and
        not only on the pose() path."""
        pose = self.pose()
        if pose is None:
            return None
        return pose[0], pose[1]

    def current_zone(self) -> str:
        """Delegated to the underlying provider WITHOUT correction: the zone
        describes where the chassis/robot is, a different concept from
        "where I detected the object" -- it makes no sense to shift it by an
        offset meant for the camera."""
        return self.pose_provider.current_zone()

    def close(self):
        self.pose_provider.close()


class FrameCorrectedPose:
    """Wraps a pose provider (typically OptiTrackPose) and applies a
    ROTATION + TRANSLATION correction to the whole pose (x, y, yaw) -- not a
    small local offset like CameraOffsetPose, but a recalibration of the
    whole frame in which bearing+distance is computed for THIS robot, so
    that the estimated global position matches the true one (see
    tools/calibrate_optitrack.py for how it is measured and why it is
    needed).

    Status (September 2026): verified with tools/read_yaw.py that the
    OptiTrack yaw is identical across robots when they are physically
    parallel (0.22 rad for all of them), and the camera is parallel to the
    rigid body. The large per-robot rotations computed in the past (-99,
    +62, -84 degrees) had been fitted ON TOP OF a bearing sign error in
    object_position.global_position() (yaw + bearing instead of
    yaw - bearing, now fixed): those correction files must be thrown away.
    With the correct sign this class should be needed at most for a small
    residual; if needed, it must be refitted from scratch with
    tools/calibrate_optitrack.py. A correct correction stays valid WHEREVER
    you move the robot: it is applied to the current pose at every frame,
    not tied to a position.

    Unlike CameraOffsetPose (a small physical camera<->rigid body shift,
    position only), here the error concerns the whole robot pose (position
    AND orientation). It must be corrected BEFORE any local offset (see the
    wrapping order in static_trial.py: FrameCorrectedPose first,
    CameraOffsetPose after -- the camera offset uses the yaw to compute
    forward/right in the robot frame, so it must see the pose already
    recalibrated to the true frame, otherwise the camera offset would be
    applied in the wrong direction).

    rotation_rad, dx_m, dy_m are the parameters computed by
    tools/calibrate_optitrack.py (least-squares fit from true-position/
    computed-position pairs of the same tracked object, with the true
    position read from an independent reference rigid body, e.g. a tracker
    on the object itself). With all three at 0.0 (default) this class
    behaves EXACTLY like the provider it wraps -- zero risk for anyone who
    has not done this calibration yet.

    Transformation applied (same rotation convention used throughout the
    project, consistent with standard math.cos/math.sin):
        x' = x*cos(rotation) - y*sin(rotation) + dx_m
        y' = x*sin(rotation) + y*cos(rotation) + dy_m
        yaw' = yaw + rotation"""

    def __init__(self, pose_provider, rotation_rad: float = 0.0,
                 dx_m: float = 0.0, dy_m: float = 0.0):
        self.pose_provider = pose_provider
        self.rotation_rad = rotation_rad
        self.dx_m = dx_m
        self.dy_m = dy_m

    def pose(self):
        """(x, y, yaw_rad) in the corrected frame, or None if the underlying
        provider does not have a valid pose (yet, or any more) -- NEVER
        invent a pose when tracking is missing."""
        raw_pose = self.pose_provider.pose()
        if raw_pose is None:
            return None
        x, y, yaw = raw_pose
        if (self.rotation_rad == 0.0 and self.dx_m == 0.0
                and self.dy_m == 0.0):
            return x, y, yaw
        cos_r = math.cos(self.rotation_rad)
        sin_r = math.sin(self.rotation_rad)
        x_corrected = x * cos_r - y * sin_r + self.dx_m
        y_corrected = x * sin_r + y * cos_r + self.dy_m
        yaw_corrected = yaw + self.rotation_rad
        return x_corrected, y_corrected, yaw_corrected

    def position(self):
        """Corrected (x, y), or None -- delegates to pose() instead of the
        provider's raw position(), so the correction is applied here too."""
        pose = self.pose()
        if pose is None:
            return None
        return pose[0], pose[1]

    def current_zone(self) -> str:
        """Unlike CameraOffsetPose, HERE the zone must be recomputed on the
        CORRECTED position: the error this class corrects is in the frame
        itself (where the robot 'believes' it is relative to the room), so
        leaving the zone uncorrected would use the wrong position to decide
        which zone it is in as well. Reuses the underlying provider's
        regions (if it exposes them, like OptiTrackPose/OdometryLocalizer);
        if it does not, delegates without correcting (old provider
        behaviour, unchanged)."""
        regions = getattr(self.pose_provider, "regions", None)
        unknown_zone = getattr(self.pose_provider, "unknown_zone",
                               "unknown")
        if not regions:
            return self.pose_provider.current_zone()
        pos = self.position()
        if pos is None:
            return unknown_zone
        x, y = pos
        for region in regions:
            if (region.x_min <= x < region.x_max
                    and region.y_min <= y < region.y_max):
                return region.name
        return unknown_zone

    def close(self):
        self.pose_provider.close()
