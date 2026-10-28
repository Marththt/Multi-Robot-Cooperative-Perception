"""Estimate the GLOBAL position (not just the zone) of a detected object,
and group nearby detections as the same physical instance -- the answer
to "how do they tell apart two identical objects in different places".

How it works, in short
----------------------
With a single RGB camera (no depth/stereo on the RoboMaster EP) distance
cannot be measured directly. It is estimated from how large the object
appears in the image (pinhole model / similar triangles), ASSUMING the
real size of the object is known (REAL_HEIGHT_M below -- measure it on
your real objects, do not trust the placeholders).
Then it is combined: robot pose (x, y, yaw -- from localization.py) +
direction of the object in the image (bearing) + estimated distance ->
global position (x, y) of the object, in the same reference frame as
the odometry.

Honest limitation, not hidden
-----------------------------
The distance estimation error GROWS with the real distance: a far object
covers few pixels, so an error of a few pixels in the bounding box
(normal detection noise) turns into a much larger distance error than
for a close object. Do not trust this estimate beyond 2-3 metres without
having validated it empirically on your camera/objects (e.g. place an
object at a known distance, compare with what this module estimates,
adjust the constants below if needed).

Then the grouping: detections (even from different robots, even at
different times) whose estimated position falls within --threshold-m are
treated as THE SAME physical instance; beyond that threshold they are
different instances. It is a simple greedy clustering (it depends on the
order in which detections arrive) -- enough to start with; if in
practice it produces unstable clusters, the next step is a real
clustering (e.g. scikit-learn DBSCAN, which does not depend on order).
"""

import math
from typing import List, Optional, Tuple

# Placeholder -- TO BE VERIFIED against your real camera, or better
# calibrated (tools/calibrate_camera.py, which gives a measured value
# instead of this datasheet estimate). Diagonal FOV 120 degrees = the one
# declared by DJI for the RoboMaster EP (dji.com/robomaster-ep/specs,
# September 2026; the datasheet does not specify diagonal/horizontal,
# assumed diagonal here as the most common convention in camera
# datasheets). Split into horizontal/vertical correctly via the pinhole
# model: tan(FOV_diagonal/2) = sqrt(tan(FOV_h/2)^2 + tan(FOV_v/2)^2) --
# NOT a proportional split of the degrees (the 96/54 of the previous
# version was exactly 16:9, i.e. a simple rescaling by the aspect ratio,
# the wrong formula for a lens). Assumes a 16:9 stream (standard
# convention for the 360p/540p/720p labels of camera.py, not verified on
# your camera) and a rectilinear projection (a truly wide-angle lens like
# this one probably has non-negligible distortion that the pinhole model
# ignores -- another reason why this stays a placeholder, not certified
# data).
_DECLARED_DIAGONAL_FOV_RAD = math.radians(120)
_ESTIMATED_STREAM_ASPECT = 16 / 9
_tan_half_diagonal = math.tan(_DECLARED_DIAGONAL_FOV_RAD / 2)
_tan_half_vertical = _tan_half_diagonal / math.sqrt(_ESTIMATED_STREAM_ASPECT ** 2 + 1)
VERTICAL_FOV_RAD = 2 * math.atan(_tan_half_vertical)
HORIZONTAL_FOV_RAD = 2 * math.atan(_ESTIMATED_STREAM_ASPECT * _tan_half_vertical)

# Typical real height in metres -- TO BE TUNED on YOUR real objects
# (measure them with a ruler: these are only reasonable placeholders).
REAL_HEIGHT_M = {
    "bottle": 0.25,
    "cup": 0.12,
}


def estimate_distance(bbox_height_px: float, image_height_px: float,
                      class_name: str) -> Optional[float]:
    """Estimated distance from the robot to the object (metres), from
    similar triangles. None if the class has no known real height or the
    bbox is degenerate."""
    real_height = REAL_HEIGHT_M.get(class_name)
    if real_height is None or bbox_height_px <= 0:
        return None
    focal_px = (image_height_px / 2) / math.tan(VERTICAL_FOV_RAD / 2)
    return (focal_px * real_height) / bbox_height_px


def estimate_bearing(bbox_center_x_px: float, image_width_px: float) -> float:
    """Angle relative to the camera optical axis (radians, positive to
    the right), from where the bbox centre falls horizontally in the
    image."""
    focal_px = (image_width_px / 2) / math.tan(HORIZONTAL_FOV_RAD / 2)
    offset_px = bbox_center_x_px - image_width_px / 2
    return math.atan2(offset_px, focal_px)


def global_position(robot_pose: Tuple[float, float, float],
                    relative_bearing: float, distance: float) -> Tuple[float, float]:
    """robot_pose = (x, y, yaw_rad) -- see localization.py. Returns
    (x, y) of the object in the same reference frame as the odometry
    (relative to the robot's starting point)."""
    robot_x, robot_y, robot_yaw = robot_pose
    # MINUS, not plus: relative_bearing is positive to the RIGHT (OpenCV
    # convention of aruco_detector.detect_with_pose and of
    # estimate_bearing above), while the yaw of the global frame grows to
    # the LEFT (counter-clockwise). With "+" an object seen on the right
    # ended up mirrored on the left: x stayed correct (cos is even) but y
    # was flipped. Verified on the side-by-side P0/P1/P2 test, same box:
    # spread among the three points went from 2.01m (with +) to 0.22m
    # (with -).
    global_angle = robot_yaw - relative_bearing
    x = robot_x + distance * math.cos(global_angle)
    y = robot_y + distance * math.sin(global_angle)
    return (x, y)


def estimate_object_position(bbox: Tuple[int, int, int, int],
                             image_size: Tuple[int, int],
                             robot_pose: Tuple[float, float, float],
                             class_name: str) -> Optional[Tuple[float, float]]:
    """Single entry point. bbox=(x, y, w, h) in pixels
    (x, y = top-left corner). image_size=(width, height)
    in pixels. None if it cannot be estimated (class without a known real
    height, or robot pose not available -- pass None as robot_pose in
    that case)."""
    if robot_pose is None:
        return None
    x, y, w, h = bbox
    img_width, img_height = image_size
    distance = estimate_distance(h, img_height, class_name)
    if distance is None:
        return None
    center_x = x + w / 2
    bearing = estimate_bearing(center_x, img_width)
    return global_position(robot_pose, bearing, distance)


def cluster_by_position(positions: List[Tuple[float, float]],
                        threshold_m: float = 0.75) -> List[int]:
    """positions: list of (x, y) (no None -- filter them out first).
    Returns a list of cluster indices, same order/length as positions:
    two detections with the same index are treated as the same physical
    instance."""
    centroids: List[List[float]] = []  # [sum_x, sum_y, count]
    assignments = []

    for x, y in positions:
        best_idx, best_dist = None, threshold_m
        for idx, (sx, sy, n) in enumerate(centroids):
            d = math.hypot(x - sx / n, y - sy / n)
            if d <= best_dist:
                best_idx, best_dist = idx, d
        if best_idx is None:
            centroids.append([x, y, 1])
            assignments.append(len(centroids) - 1)
        else:
            sx, sy, n = centroids[best_idx]
            centroids[best_idx] = [sx + x, sy + y, n + 1]
            assignments.append(best_idx)

    return assignments
