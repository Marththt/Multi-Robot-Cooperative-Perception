"""Detection with OpenCV + classifier, and fault injection
(Topic 4: robustness of multi-robot perception).

What this module does, and what it does NOT do
----------------------------------------------
It combines candidates.py (finds candidate regions) and classifier.py
(classifies them) into a single reading loop, and applies the optional
simulated systematic fault to the result. This module does NOT decide
how to move the robot (patrol.py) and does NOT fuse the reports of
several robots (baselines 0-3, one level up): here only the raw
observation of ONE robot is produced, with or without fault.

History of the detection (for whoever reads this module from scratch)
---------------------------------------------------------------------
Official SDK markers (`vision.sub_detect_info`) -> candidates.py +
classifier.py (home-made, now superseded, left on disk only as a
reference) -> zero-shot YOLO (yolo_detector.py, no training, but only a
bbox: distance was estimated with a heuristic) -> NOW ArUco markers
(aruco_detector.py), at the professor's explicit request: the fusion
baselines need MEASURED distance/angle for every detection, not just
estimated from the bbox -- see the aruco_detector.py docstring for the
full reasoning. YOLO remains available (patrol.py --detector yolo) for
objects without markers. In any case the interface towards patrol.py and
detection_recorder.py does not change: whatever detector is used, a
Detection comes out with the same shape (with distance/viewing_angle_rad
populated only if the detector really measures them, see the dataclass
below).

Physical fault vs simulated fault
---------------------------------
The professor's document distinguishes "physical" faults (occlusion,
lighting, angle: obtained for real by placing an obstacle or changing
the lighting in the lab, and with a real classifier these faults ALREADY
happen on their own -- a half-occluded or badly lit object is classified
worse without any dedicated code) from a "simulated systematic" fault (a
robot designated faulty with probability p, whose detections are dropped
or mislabelled in software, to get reproducible experiments). Only the
latter is simulated here.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from vf.perception.object_position import estimate_object_position, global_position

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arena zones
# ---------------------------------------------------------------------------
# HISTORICAL/no longer used: the real zone now comes from localization.py
# or optitrack.py (configurable regions, see config/optitrack_regions.json),
# not from a fixed list here. Left only as a reference -- no module
# imports it any more (verified).
ZONES = ["north", "east", "south", "west", "center"]

# ---------------------------------------------------------------------------
# Recognised classes
# ---------------------------------------------------------------------------
# Defined HERE, not in the detection module: the detection can change
# (zero-shot yolo_detector.py, now aruco_detector.py at the professor's
# request, maybe something else tomorrow) without the simulated fault
# below needing to know which one is in use right now.
# "box": the current physical objects (white polystyrene boxes, each with
# an ArUco marker attached -- see aruco_detector.py and
# config/aruco_classes.json). "bottle"/"cup" remain for when we go back
# to YOLO on objects without markers (see yolo_detector.py for why
# "cards" is not there yet either).
CLASSES = ["box", "bottle", "cup"]

# ---------------------------------------------------------------------------
# Simulated systematic fault
# ---------------------------------------------------------------------------
DEFAULT_DROPOUT_PROB = 0.3
DEFAULT_MISLABEL_PROB = 0.2


@dataclass
class Detection:
    """A single detection, already passed (or not) through the simulated
    fault."""
    t: float
    robot_id: str
    class_name: Optional[str]
    zone: str
    confidence: float
    fault_injected: bool
    dropped: bool  # True if the fault made the detection disappear
    bbox: Optional[Tuple[int, int, int, int]] = None  # debug only
    # Estimated GLOBAL position (x, y), see object_position.py -- None if
    # it cannot be computed (robot pose not available, or a "nothing
    # detected" entry that has no bbox to estimate it from). It is used to
    # tell apart two objects of the same class at different points of the
    # room, which the zone alone cannot do (see the historical
    # fondi_log_reali.py).
    position: Optional[Tuple[float, float]] = None
    # distance/viewing_angle: ONLY when the detector really MEASURES them
    # (aruco_detector.ArucoDetector via solvePnP) -- not the heuristic
    # bbox estimate used to compute `position` with YOLO. Keeping them
    # separate from the `position` field matters: they are exactly the
    # two quantities that the professor's fault model p_D(occlusion,
    # angle, distance) (Section 7) will have to use for the week 3-4 data
    # collection -- None with YOLO, so as not to pass off an estimate as a
    # measurement.
    distance: Optional[float] = None
    viewing_angle_rad: Optional[float] = None
    # bearing of the marker (positive to the right of the optical axis)
    # and the camera pose (x, y, yaw) of the frame it was seen in: with
    # distance they are the raw observation the fusion turns into a world
    # position (localization_methods.measurement_from_observation)
    bearing_rad: Optional[float] = None
    robot_pose: Optional[Tuple[float, float, float]] = None

    def to_json(self):
        return json.dumps(asdict(self))


class FaultInjector:
    """Reproducible systematic fault for ONE robot designated as faulty.

    A fixed seed per experiment (not per detection) makes the results
    repeatable: same seed, same sequence of drops/errors, even when
    re-running the experiment later to compare different aggregation
    methods on the SAME faults.
    """

    def __init__(self, robot_id, faulty_robot_ids, dropout_prob=None,
                 mislabel_prob=None, seed=None):
        self.robot_id = robot_id
        self.active = robot_id in set(faulty_robot_ids)
        self.dropout_prob = (DEFAULT_DROPOUT_PROB if dropout_prob is None
                             else dropout_prob)
        self.mislabel_prob = (DEFAULT_MISLABEL_PROB if mislabel_prob is None
                              else mislabel_prob)
        base = seed if seed is not None else 0
        # Do NOT use hash((base, robot_id)): the hash of a tuple containing
        # a string depends on PYTHONHASHSEED, randomised at every process
        # start -- "same seed" would give a DIFFERENT sequence every time,
        # the opposite of what is needed to compare the baselines on the
        # same faults. random.Random() seeded with a string instead uses a
        # stable hashing (SHA-512) independent of PYTHONHASHSEED: it is the
        # only correct way to seed with robot_id.
        self._rng = random.Random(f"{base}-{robot_id}")

    def apply(self, class_name, confidence):
        """Returns (class_name, confidence, fault_injected, dropped) after
        any corruption. If not active, passes everything through
        unchanged."""
        if not self.active:
            return class_name, confidence, False, False

        if self._rng.random() < self.dropout_prob:
            return None, 0.0, True, True

        if self._rng.random() < self.mislabel_prob:
            others = [c for c in CLASSES if c != class_name]
            if others:
                return self._rng.choice(others), confidence, True, False

        return class_name, confidence, False, False


class ObjectDetector:
    """One reading loop: capture, detect, apply the simulated fault.

    `detector` is any object with a method
    `.detect(img) -> list[(class_name, confidence, bbox)]` -- e.g. the
    zero-shot YOLO detector (yolo_detector.py), in future possibly a
    fine-tuned YOLO, without this class or patrol.py having to change.

    `camera` is any object with a NON-blocking `.read() -> (frame, age)`
    method, returning (None, None) if there is nothing yet -- the
    interface of YOUR Camera class (robomaster_sdk.camera.Camera): a
    background reader caching the last frame, not a direct
    `robot.camera.read_cv2_image()`. Reason: the same one already written
    in the docstring of your camera.py -- read_cv2_image() blocks until
    it has a frame, and calling it inside a 10 Hz loop ties the sample
    rate to the video rate instead of the patrol clock. The previous
    version of this module called read_cv2_image() directly (never tried
    on the real robot, so never measured as a problem -- but it is the
    same bug your code already documents as solved elsewhere, fixed here
    before discovering it on the robot)."""

    def __init__(self, camera, robot_id, current_zone_fn, detector,
                 fault_injector=None, pose_fn=None, max_age=None):
        self.camera = camera
        self.robot_id = robot_id
        self.current_zone_fn = current_zone_fn
        self.detector = detector
        self.fault_injector = fault_injector
        # pose_fn: callable -> (x, y, yaw_rad) or None (see
        # localization.OdometryLocalizer.pose). If None, Detection.position
        # always stays None: things keep working at zone level only
        # (previous behaviour), no error.
        self.pose_fn = pose_fn
        # max_age: seconds beyond which a frame is considered too old
        # (consistent with MAX_FRAME_AGE in camera.py, used there for the
        # manipulation dataset). None (default) = no check: there is no
        # tuned value for this use case yet.
        self.max_age = max_age

    def read(self):
        """Returns a list of Detection. Empty if there is no frame yet
        (camera just started, stream dropped) or if the available one is
        too old (max_age) -- MISSING data, not "I looked and there was
        nothing": the two must be kept distinct. If there is a frame but
        no detection, it still returns ONE Detection with class_name=None:
        without it, events.jsonl cannot distinguish "the robot never
        passed here" from "it passed and saw nothing" -- and without that
        distinction neither recall nor the false positive rate can be
        computed on real data (see metriche.py/
        valuta_esperimento_reale.py: both also count negative cases, not
        just detections).

        If `detector` also exposes .detect_with_pose(img) (today
        aruco_detector.ArucoDetector), it is used instead of
        .detect(img): distance/bearing MEASURED for every detection
        instead of estimated from the bbox height
        (object_position.estimate_object_position, used only as a
        fallback for a detector without pose -- e.g. YOLO). See the
        aruco_detector.py docstring for why."""
        zone = self.current_zone_fn()
        t = time.time()
        reports = []

        img, age = self.camera.read()
        if img is None:
            log.debug("no frame available yet")
            return reports
        if self.max_age is not None and age is not None and age > self.max_age:
            log.debug("frame %.2fs old (> %.2fs), discarded",
                      age, self.max_age)
            return reports

        if hasattr(self.detector, "detect_with_pose"):
            predictions = self.detector.detect_with_pose(img)
        else:
            predictions = [(class_name, confidence, bbox, None) for class_name,
                           confidence, bbox in self.detector.detect(img)]

        if not predictions:
            reports.append(Detection(
                t=t, robot_id=self.robot_id, class_name=None, zone=zone,
                confidence=0.0, fault_injected=False, dropped=False,
                bbox=None))
            return reports

        img_height, img_width = img.shape[0], img.shape[1]
        pose = self.pose_fn() if self.pose_fn is not None else None

        for class_name, confidence, bbox, relative_pose in predictions:
            # The position is estimated BEFORE the simulated fault: a
            # systematic fault corrupts what the robot BELIEVES it saw
            # (class/confidence), not the geometry -- a mislabel is still
            # "something seen in that direction/distance", even if the
            # reported class is wrong.
            if relative_pose is not None:
                distance, bearing, viewing_angle = relative_pose
                position = (global_position(pose, bearing, distance)
                            if pose is not None else None)
            else:
                distance, bearing, viewing_angle = None, None, None
                position = estimate_object_position(
                    bbox, (img_width, img_height), pose, class_name)

            if self.fault_injector is not None:
                class_name, confidence, fault, dropped = (
                    self.fault_injector.apply(class_name, confidence))
            else:
                fault, dropped = False, False

            if dropped:
                reports.append(Detection(
                    t=t, robot_id=self.robot_id, class_name=None, zone=zone,
                    confidence=0.0, fault_injected=True, dropped=True,
                    bbox=bbox, position=None))
                continue

            reports.append(Detection(
                t=t, robot_id=self.robot_id, class_name=class_name, zone=zone,
                confidence=confidence, fault_injected=fault,
                dropped=False, bbox=bbox, position=position,
                distance=distance, viewing_angle_rad=viewing_angle,
                bearing_rad=bearing,
                robot_pose=tuple(pose) if pose is not None else None))

        return reports