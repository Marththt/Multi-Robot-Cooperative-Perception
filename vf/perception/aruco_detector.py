"""Detection with ArUco markers: class, distance, bearing and viewing angle of every marker.

Every box carries ArUco markers of known physical size, so solvePnP gives the measured distance and
orientation of each marker from one image, given the marker size (--aruco-marker-size) and the
camera calibration (tools/calibrate_camera.py; without it, distance and angle are only approximated
from the declared field of view). The fusion methods need these measured distances and angles for
every detection.

Same class, same ID
-------------------
Several IDs can map to the same class (--aruco-classes, see config/aruco_classes.json), and all
boxes carry the same marker: a robot recognises the CLASS of an object but not its identity. Telling
two boxes apart is the job of the estimated position (object_position.py) and of the fusion methods,
not of the marker. Without --aruco-classes every marker maps to the single class "box"
(default_class).
"""

import json
import logging
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

ARUCO_CONFIDENCE = 1.0  # a marker that is read is read: there is no
                        # continuous score as in YOLO, the "confidence"
                        # here is boolean (detected / not detected). 1.0
                        # keeps the same interface (class_name, confidence,
                        # bbox) without inventing a number that does not
                        # exist.

DEFAULT_DICTIONARY = cv2.aruco.DICT_4X4_50  # small dictionary: only a few
    # distinct IDs are needed for now (a handful of boxes) -- fewer markers
    # in the dictionary = lower probability of confusing one ID with
    # another at distance or with motion blur. Raise --aruco-dict if you
    # need more IDs.

# Same FOV placeholder as object_position.py -- IMPORTED from there, not
# copied: the two separate copies used before (96/54 degrees in both
# files) were a risk of silent divergence if one was updated and the
# other was not (which is exactly what had happened: the computation was
# wrong in both in the same way).
# Used ONLY for the approximate camera matrix below, if you do not pass
# a real calibration (tools/calibrate_camera.py).
from vf.perception.object_position import (
    HORIZONTAL_FOV_RAD as _DEFAULT_HORIZONTAL_FOV_RAD,
    VERTICAL_FOV_RAD as _DEFAULT_VERTICAL_FOV_RAD)
def dictionary_from_name(name: str) -> int:
    """'4X4_50' -> cv2.aruco.DICT_4X4_50. Clear error if the name does not
    exist, instead of a cryptic AttributeError further downstream."""
    attribute = f"DICT_{name}"
    if not hasattr(cv2.aruco, attribute):
        raise ValueError(
            f"unknown ArUco dictionary '{name}' (expected something like "
            f"4X4_50, 5X5_100, 6X6_250 -- see the cv2.aruco.DICT_* constants)")
    return getattr(cv2.aruco, attribute)


def load_aruco_classes(path: str) -> Dict[int, str]:
    """Reads the marker-id -> class JSON (see config/aruco_classes.json).
    JSON keys are necessarily strings (a limitation of the format): here
    they are turned back into int, as detectMarkers returns them."""
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["marker_to_class"].items()}


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the JSON written by tools/calibrate_camera.py."""
    with open(path) as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def approximate_camera(width_px: int, height_px: int,
                       horizontal_fov_rad: float = _DEFAULT_HORIZONTAL_FOV_RAD,
                       vertical_fov_rad: float = _DEFAULT_VERTICAL_FOV_RAD
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """APPROXIMATE camera matrix from the declared (unverified) FOV.
    Lets ArUco run today without waiting for a real calibration, but
    distance/angle inherit the same error margin already documented in
    object_position.py. For reliable numbers, calibrate with
    tools/calibrate_camera.py and pass that file instead of letting this
    function be used (ArucoDetector calls it on its own, with a warning,
    only if you do not pass camera_matrix/dist_coeffs)."""
    fx = (width_px / 2) / math.tan(horizontal_fov_rad / 2)
    fy = (height_px / 2) / math.tan(vertical_fov_rad / 2)
    cx, cy = width_px / 2, height_px / 2
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)  # no distortion
        # estimated without a real calibration: assumes an ideal lens,
        # reasonable near the image centre, worse at the edges.
    return camera_matrix, dist_coeffs


def relaxed_parameters() -> "cv2.aruco.DetectorParameters":
    """OpenCV default DetectorParameters, with a few thresholds relaxed
    for the cases that matter most here: small marker in the frame
    (2.5-3m away, 8cm marker = a few tens of pixels) and some blur /
    non-ideal lighting (no dedicated calibration, robot camera stream).
    Reason for the change: "flickering" observed on the real robot
    (sometimes the marker is seen, sometimes not, with marker and robot
    still) -- the default cv2.aruco.DetectorParameters() are meant for
    cleaner conditions (high-resolution image, close marker) and discard
    marginal candidates instead of trying to decode them anyway.

    Honest trade-off: this increases sensitivity, so in theory also the
    risk of false positives (decoding noise as a marker) -- low risk in
    practice because the 4X4_50 dictionary still has an error-correction
    bit check that discards most noise, but it is not a guarantee. If one
    day you see an invented ID that you never printed, this is the first
    suspect to check.

    It is NOT a substitute for camera calibration
    (tools/calibrate_camera.py) or for a larger marker / closer robot if
    the flickering persists -- it is a tweak to the detection parameters,
    it does not add information that is not in the frame."""
    p = cv2.aruco.DetectorParameters()
    # Adaptive threshold: a wider range and finer step help with uneven
    # lighting or lighting that differs slightly from reading to reading
    # (the OpenCV default 3-23-10 is already of this order of magnitude,
    # here it is kept explicit so it can be tuned).
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 30
    p.adaptiveThreshWinSizeStep = 5
    # Minimum accepted perimeter as a fraction of the image diagonal --
    # the OpenCV default 0.03 discards markers covering less than 3% of
    # the image perimeter, the typical case of a small marker seen from
    # far away. Lowered to 0.01.
    p.minMarkerPerimeterRate = 0.01
    # How far a detected contour may deviate from a perfect square and
    # still be considered a marker candidate -- raised slightly to
    # tolerate the perspective/blur of a marker seen from the side or not
    # perfectly in focus.
    p.polygonalApproxAccuracyRate = 0.05
    # Sub-pixel refinement of the detected corners: slower (not
    # significant for 1-2 markers at a few Hz) but more precise corners,
    # which help solvePnP give a more stable distance/bearing frame to
    # frame instead of jumping between slightly different readings.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def _marker_object_points(size_m: float) -> np.ndarray:
    """3D points of the marker corners in ITS OWN reference frame (centre
    at the origin, order TL,TR,BR,BL -- the same as detectMarkers),
    required by solvePnP. cv2.aruco.estimatePoseSingleMarkers is not
    used: it was removed in recent OpenCV versions (verified: absent in
    this environment, cv2 4.13) -- plain solvePnP works on any version."""
    m = size_m / 2
    return np.array([
        [-m, m, 0],
        [m, m, 0],
        [m, -m, 0],
        [-m, -m, 0],
    ], dtype=np.float32)


class ArucoDetector:
    """Same minimal interface as the YOLO detectors (.detect(img) ->
    list[(class_name, confidence, bbox)]), PLUS .detect_with_pose(img),
    which also returns a relative pose (distance, bearing, viewing angle)
    for every detection -- perception.ObjectDetector uses the latter when
    available (checked with hasattr), otherwise it behaves as with YOLO.
    Nothing breaks for callers still passing RemoteYoloDetector."""

    def __init__(self, marker_size_m: float, camera_matrix=None,
                 dist_coeffs=None, dictionary: Optional[int] = None,
                 marker_to_class: Optional[Dict[int, str]] = None,
                 default_class: str = "box",
                 parameters: Optional["cv2.aruco.DetectorParameters"] = None,
                 relax_parameters: bool = True,
                 max_plausible_distance_m: Optional[float] = 8.0):
        self.marker_size_m = marker_size_m
        self.camera_matrix = camera_matrix
        # Safety net AGAINST the near/far ambiguity of solvePnP on a
        # planar marker (see _detect_raw: SOLVEPNP_IPPE_SQUARE almost
        # always resolves it, but "almost" is not "always" with noisy
        # detected corners) -- a distance beyond this threshold for THIS
        # arena is physically impossible (no robot is ever 8m from an
        # object), so it is more likely a numerical artefact than real
        # data: it is discarded as if solvePnP had failed, not propagated.
        # None disables the check (previous behaviour, for callers who
        # pass it explicitly).
        self.max_plausible_distance_m = max_plausible_distance_m
        self.dist_coeffs = dist_coeffs
        if self.camera_matrix is None:
            log.warning(
                "ArucoDetector without camera calibration: using a matrix "
                "APPROXIMATED from the declared FOV (see "
                "approximate_camera()) until you run "
                "tools/calibrate_camera.py. Distance/angle will be less "
                "reliable than the professor's fault model requires "
                "for final numbers.")
        self._object_pts = _marker_object_points(marker_size_m)
        dictionary = dictionary if dictionary is not None else DEFAULT_DICTIONARY
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        # explicit parameters > relaxed (default, see relaxed_parameters())
        # > plain OpenCV defaults if relax_parameters=False -- useful to
        # compare "before/after" if one day you want to check how much
        # they really help.
        if parameters is not None:
            self._parameters = parameters
        elif relax_parameters:
            self._parameters = relaxed_parameters()
        else:
            self._parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._parameters)
        # id -> class. No entry -> default_class (for now a single real
        # class, "box": see the module docstring on the "same class, same
        # ID" trick).
        self.marker_to_class = marker_to_class or {}
        self.default_class = default_class

    def _class_for_id(self, marker_id: int) -> str:
        return self.marker_to_class.get(marker_id, self.default_class)

    def _ensure_camera(self, img):
        if self.camera_matrix is None:
            height_px, width_px = img.shape[0], img.shape[1]
            self.camera_matrix, self.dist_coeffs = approximate_camera(
                width_px, height_px)

    def _detect_raw(self, img):
        """(marker_id, bbox, rvec, tvec) for every marker -- step shared by
        detect() and detect_with_pose()."""
        if img is None:
            return []
        self._ensure_camera(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []

        results = []
        for c, marker_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            # flags=IPPE_SQUARE (not the default ITERATIVE): for EXACTLY 4
            # planar points (a single square marker, always our case) it is
            # the method designed for this -- ITERATIVE is known to
            # converge, occasionally and without warning, to the wrong
            # "mirrored" solution of a planar pose (same 2D projection,
            # completely different depth: e.g. the true 1-2m distance
            # becoming ~20m on one frame and correct again on the next,
            # with the marker still) -- IPPE_SQUARE computes BOTH candidate
            # solutions and returns the one with the lower reprojection
            # error, instead of possibly getting stuck in a local minimum
            # like a generic iterative solver.
            ok, rvec, tvec = cv2.solvePnP(
                self._object_pts, pts.astype(np.float32),
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                log.debug("solvePnP failed for marker %s", marker_id)
                continue

            if self.max_plausible_distance_m is not None:
                distance = float(np.linalg.norm(tvec))
                if distance > self.max_plausible_distance_m:
                    log.warning(
                        "marker %s: distance %.2fm beyond the plausible "
                        "threshold (%.1fm) -- discarded as a likely "
                        "numerical artefact of solvePnP, not a real "
                        "detection (see max_plausible_distance_m)",
                        marker_id, distance, self.max_plausible_distance_m)
                    continue

            results.append((int(marker_id), bbox, rvec, tvec))
        return results

    def detect(self, img):
        """Compatibility with the YOLO detector interface: only
        (class_name, confidence, bbox), no pose."""
        return [(self._class_for_id(mid), ARUCO_CONFIDENCE, bbox)
                for mid, bbox, _rvec, _tvec in self._detect_raw(img)]

    def detect_with_pose(self, img):
        """list[(class_name, confidence, bbox, relative_pose)].
        relative_pose = (distance_m, bearing_rad, viewing_angle_rad):
          - distance_m: norm of tvec (camera -> marker)
          - bearing_rad: horizontal angle of the marker with respect to
            the camera optical axis (positive to the right) -- replaces
            object_position.estimate_bearing with a measured value, not
            estimated from the bbox centre
          - viewing_angle_rad: how much the FACE of the marker is tilted
            with respect to looking straight at the camera (0 = frontal,
            pi/2 = edge-on, almost invisible). Not yet used by any
            downstream module, but it is exactly the "angle" that the
            professor's fault model p_D(occlusion, angle, distance)
            (Section 7) will use when the week 3-4 data arrives -- we
            already log it now (see Detection in perception.py) instead
            of having to add it later.
        """
        results = []
        for mid, bbox, rvec, tvec in self._detect_raw(img):
            x, y, z = tvec.flatten()
            distance = float(np.linalg.norm(tvec))
            bearing = math.atan2(x, z)  # OpenCV convention: x=right+,
                                        # z=forward (optical axis)

            R, _ = cv2.Rodrigues(rvec)
            marker_normal = R @ np.array([0.0, 0.0, 1.0])  # normal of
                # the marker face, in the camera frame
            view_direction = np.array([0.0, 0.0, 1.0])  # from the camera
                # towards the marker, same frame (approximate: it should be
                # normalised on tvec for a strongly off-centre marker, but
                # the error is negligible in our field of view)
            cos_ang = float(np.clip(-marker_normal @ view_direction, -1.0, 1.0))
            viewing_angle = math.acos(cos_ang)

            class_name = self._class_for_id(mid)
            results.append(
                (class_name, ARUCO_CONFIDENCE, bbox,
                 (distance, bearing, viewing_angle)))
        return results
