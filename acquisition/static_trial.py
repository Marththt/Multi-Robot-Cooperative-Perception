"""Capture ONE static trial for one robot: one observation per distinct
physical OBJECT in view (class, confidence, distance, bearing, position)
-- not a patrol.

This is the "static trial" design from the professor's document (Section
3.2: "exactly one observation is counted per robot per static trial, a
single snapshot, not one per video frame") -- different from the
continuous-patrol architecture of patrol.py, which was built before this
detail and is NOT the priority for now: moving is not needed to validate
the fusion algorithms, running them on a stationary trial is. patrol.py
stays ready for later; this script neither touches nor replaces it.

If the scene contains MORE than one object (e.g. 2 boxes seen together),
"one observation per robot per trial" is read per OBJECT, not per trial:
each robot may write several rows in the same trial (one per distinct
physical instance it sees), grouped by estimated position -- see
capture_trial() below. merge_static_trials.py then re-pairs them across
robots, again by position, before running the fusion.

Why a separate file, and not patrol.py without the motion
------------------------------------------------------------------------
A static trial has no "traversed zones" and no path: robots and objects
stay still, and a human decides WHEN to take the snapshot (run this
script once per robot per trial). Reusing PATROL_ROUTE / the 10Hz loop of
Patrol.run() only to ignore its motion would be more code to understand
for a simpler task. What IS reused (identical, already tested):
ObjectDetector, the detector (ArUco or YOLO), the pose provider
(OptiTrack or odometry) -- same logic, same modules as patrol.py, just
without driving the chassis.

One trial = one physical arrangement of objects in the arena, stationary.
--n-readings closely spaced readings per robot in THAT trial (default 5),
and only ONE is kept (see _pick_representative() below: with ArUco, whose
confidence is always 1.0, the reading whose distance is closest to the
group median is chosen -- this reduces the noise of a single unlucky
frame, residual motion blur or a reflection, so the choice is actually
meaningful rather than an arbitrary tie-break), staying faithful to the
document's "one observation per robot per trial": the final choice that
ends up in the log is ONE, not N.

Usage (one process per robot, like patrol.py -- see there for why YOLO
needs separate processes)
--------------------------------------------------------------------------
    python -m acquisition.static_trial --robot-id T0 --trial-id trial001 \\
        --out trials/ --regions config/optitrack_regions.json \\
        --localization optitrack --optitrack-rb-id 0 \\
        --detector aruco --aruco-marker-size 0.08

    python -m acquisition.static_trial --robot-id T1 --trial-id trial001 \\
        --out trials/ --regions config/optitrack_regions.json \\
        --localization optitrack --optitrack-rb-id 1 \\
        --detector aruco --aruco-marker-size 0.08

Appends ONE JSON row to <out>/<robot-id>.jsonl (one file per robot:
several trials in the same folder accumulate, each row labelled with its
--trial-id). merge_static_trials.py then reads ALL robot files in a
folder, groups them by trial-id and runs one or more aggregation
baselines on top.

Live preview, BEFORE trusting a trial
------------------------------------------------
If a trial returns class_name=None for a robot, a log cannot tell whether
the camera failed to start or the marker simply was not well framed.
--live opens a window with the REAL camera video, with detected markers
drawn on top in real time (green outline + class/distance), instead of
capturing a trial:

    python -m acquisition.static_trial --robot-id P0 --detector aruco \\
        --aruco-marker-size 0.08 --live

With --live, --trial-id/--out/--regions/--localization are not needed
(the camera and the detector are all this question requires). Press 'q'
in the window to quit. Same robot connection as always, same command you
would use for a real trial: just add/remove --live, --trial-id/--out.

Adding --state-out (plus --regions and the localization flags) also
writes this robot's live state continuously, for dashboard.py:

    python -m acquisition.static_trial --robot-id P0 --live --state-out live/P0.json \\
        --regions config/optitrack_regions.json \\
        --localization optitrack --optitrack-rb-id 8 \\
        --detector aruco --aruco-marker-size 0.08 \\
        --camera-calib config/checkerboard_calib_P0.json
"""

import argparse
import json
import logging
import math
import os
import time

import cv2

from vf.perception.object_position import global_position          # in live_preview
from vf.perception.object_position import cluster_by_position      # in capture_trial
from vf.localization.localization import (load_regions, OdometryLocalizer,
                                          CameraOffsetPose, FrameCorrectedPose)
from vf.localization.mocap_pose import MocapPose             # in main
from vf.localization.optitrack import OptiTrackPose          # in main
from vf.perception.perception import ObjectDetector, FaultInjector  # in main

from vf.perception.aruco_detector import (                         # in main
            ArucoDetector, load_aruco_classes, load_calibration,
            dictionary_from_name)

from vf.perception.yolo_client import RemoteYoloDetector         # in main
log = logging.getLogger(__name__)


def write_live_state(path, robot_id, pose, detections_state, p_d=1.0):
    """ATOMIC write (temporary file + rename) of THIS robot's live state,
    for dashboard.py. Overwritten on every frame -- this is not a growing
    log (like the static trials), just "the last thing this robot saw",
    read continuously by another process. The atomic rename prevents
    dashboard.py from reading a half-written file (rare but possible with
    a file rewritten several times per second by one process while
    another reads it)."""
    data = dict(
        robot_id=robot_id, t=time.time(),
        pose=list(pose) if pose is not None else None,
        detections=detections_state,
        # p_D with which THIS robot is dropping detections (--p-d), shown
        # by the dashboards next to the robot (1.0 = no interference).
        p_d=p_d)
    folder = os.path.dirname(path)
    if folder:
        # unlike the non-live path (main() already calls os.makedirs on
        # --out before writing), here the --state-out folder (e.g.
        # "live/") may not exist yet on first start -- without this,
        # open(tmp, "w") fails with FileNotFoundError on the first frame,
        # before the video window is even shown.
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows only: an antivirus/indexer (Windows Defender and
            # similar often scan newly created/modified files in real
            # time) can hold the file for a fraction of a second exactly
            # when this rename arrives -- transient, and not caused by
            # dashboard.py (which reads read-only; on Windows Python
            # already opens files with read/write/delete sharing, so it
            # should never block a rename by itself). Retry a few times
            # with a short pause instead of bringing down the whole
            # process (and the video window with it) for a single frame.
            time.sleep(0.02 * (attempt + 1))
    # still failing after the retries: do NOT raise -- a missed state
    # update must not interrupt the capture/video preview, which remains
    # this process's primary information (dashboard.py will simply mark
    # this robot as "old"/STALE if it keeps happening long enough).
    # Warning printed sparingly (not on every frame) so as not to flood
    # the terminal.
    write_live_state._n_failed = getattr(write_live_state, "_n_failed", 0) + 1
    n = write_live_state._n_failed
    if n in (1, 10, 100) or n % 1000 == 0:
        print(f"[warning] writing {path} blocked by another "
              f"process {n} time(s) so far (probably antivirus/file "
              f"scanning on Windows) -- skipping this state update, "
              f"the video keeps running. If it persists, try "
              f"excluding the folder from the antivirus.")


def live_preview(camera, detector, robot_id, pose_fn=None, state_out=None,
                 debug=False, p_d=1.0, seed=0):
    """Video window with the detected markers drawn on top, in real time
    -- see the module docstring (--live). 'q' to quit.

    If pose_fn AND state_out are both given (requires --regions and, with
    OptiTrack, --optitrack-rb-id -- see main()), besides showing the video
    it continuously writes THIS robot's state (pose + detections with
    global position, not just bbox) to state_out: this is what
    dashboard.py reads for the live merge across robots. Without
    pose_fn/state_out it behaves exactly as before (video preview only,
    no file written, no OptiTrack dependency) -- fully compatible with
    earlier callers.

    Each marker gets a coloured box and a small numbered circle (same
    colour); the full details (class, distance, bearing, xy) are in a
    numbered legend at the bottom, no longer written on top of each box --
    with 3+ nearby markers the text of each one overlapped its neighbour's
    and became unreadable.

    p_d < 1.0 = simulated interference (Section 7.1 of the professor's
    document, with constant p_D): EVERY detected marker in EVERY frame is
    an independent Bernoulli trial, kept with probability p_d and dropped
    with probability 1-p_d. A dropped marker is a "not seen" (missed
    detection) for this robot: it does not end up in state_out, so the
    dashboard treats it as 0 observations for that class. In the video
    window it stays drawn in GREY with "DROPPED", so you can see what the
    interference removed. Draws are reproducible with the same seed
    (random.Random seeded with a string, stable across runs -- same
    choice as FaultInjector in perception.py).

    debug=True prints to the TERMINAL (no longer overlaid on the image)
    the pixel offset and the bearing-from-pixel for each marker -- the
    pixel->bearing self-consistency check commented below, previously
    drawn on the frame."""
    import random
    from vf.perception.object_position import global_position

    rng = random.Random(f"{seed}-{robot_id}-p_d")
    n_detected_total = 0
    n_dropped_total = 0
    if p_d < 1.0:
        print(f"INTERFERENCE ACTIVE: p_D={p_d:.2f} -- every marker in every "
              f"frame is kept with probability {p_d:.2f} "
              f"(seed={seed})")

    n_frames = 0
    n_no_frame = 0
    print("Waiting for the first frame... video window coming, "
          "'q' in the window to quit.")
    while True:
        img, age = camera.read()
        if img is None:
            n_no_frame += 1
            if n_no_frame % 30 == 1:
                print(f"still NO frame from the camera "
                      f"({n_no_frame} attempts) -- if this "
                      f"number keeps rising, the problem is the "
                      f"camera, not the markers")
            time.sleep(0.05)
            continue
        n_frames += 1
        if n_frames == 1:
            real_height, real_width = img.shape[0], img.shape[1]
            print(f"REAL resolution of the first frame received: "
                  f"{real_width}x{real_height}px -- compare it by "
                  f"eye with 'image_size' in the file passed to "
                  f"--camera-calib (in your two files it is [1280, 720]). If "
                  f"they differ, the calibrated camera_matrix is applied "
                  f"to pixels of a different scale: a concrete suspect for a "
                  f"distance/bearing discrepancy between robots, nothing else "
                  f"is needed to rule it out or confirm it.")

        if hasattr(detector, "detect_with_pose"):
            results = detector.detect_with_pose(img)
        else:
            results = [(c, conf, bbox, None)
                       for c, conf, bbox in detector.detect(img)]

        pose = pose_fn() if pose_fn is not None else None

        display = img.copy()
        img_height, img_width = display.shape[0], display.shape[1]
        camera_matrix = getattr(detector, "camera_matrix", None)
        cx_camera = float(camera_matrix[0][2]) if camera_matrix is not None else img_width / 2.0
        fx_camera = float(camera_matrix[0][0]) if camera_matrix is not None else None
        # Vertical line at the EXACT centre used by the computation (cx of
        # the camera_matrix, not necessarily the pixel centre of the
        # image, should cx one day no longer be assumed centred) -- a
        # visual reference to judge by eye whether a marker is shifted
        # right/left, without measuring anything with a tape.
        cv2.line(display, (int(round(cx_camera)), 0),
                 (int(round(cx_camera)), img_height), (255, 0, 0), 1)

        # One colour per marker (by index, not by class: two nearby boxes
        # stay distinguishable) so the box on the image and its row in the
        # legend at the bottom are recognisable at a glance without
        # reading overlapping text.
        MARKER_PALETTE = [(0, 140, 255), (255, 90, 60), (60, 200, 60),
                          (200, 60, 200), (60, 200, 200), (200, 200, 60)]

        detections_state = []
        legend_rows = []
        for i, (class_name, conf, bbox, relative_pose) in enumerate(results):
            x, y, w, h = bbox
            n_detected_total += 1
            # Independent Bernoulli per marker per frame: kept with
            # probability p_d. With p_d=1.0 no draw at all (the generator
            # is not consumed, behaviour identical to before).
            dropped = p_d < 1.0 and rng.random() >= p_d
            if dropped:
                n_dropped_total += 1
                grey = (128, 128, 128)
                cv2.rectangle(display, (x, y), (x + w, y + h), grey, 2)
                cx_badge, cy_badge = x + 14, y + 14
                cv2.circle(display, (cx_badge, cy_badge), 13, grey, -1)
                cv2.circle(display, (cx_badge, cy_badge), 13, (0, 0, 0), 2)
                cv2.putText(display, str(i + 1), (cx_badge - 6, cy_badge + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                legend_rows.append(
                    (i + 1, grey,
                     f"{class_name} DROPPED (interference p_D={p_d:.2f})"))
                continue
            colour = MARKER_PALETTE[i % len(MARKER_PALETTE)]
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
            position = None
            # PIXEL offset of the bbox centre from the centre used by the
            # computation (cx), and the bearing that THIS offset alone
            # would imply with a simple pinhole model
            # (atan2(offset_px, fx)) -- compare it with b= below (the real
            # one, from solvePnP): if the two numbers are close, the
            # pixel->bearing pipeline is self-consistent and the problem is
            # elsewhere (e.g. in the hand measurement); if they are very
            # different, that is concrete evidence of a bug in the
            # detection->bearing chain, no more guessing needed.
            # This used to be written on the frame (one of the reasons the
            # text of several nearby markers overlapped) -- now only on the
            # terminal, and only with --debug.
            if debug:
                marker_centre_px = x + w / 2.0
                offset_px = marker_centre_px - cx_camera
                msg = f"  marker {i + 1} ({class_name}): offset={offset_px:+.0f}px"
                if fx_camera:
                    bearing_from_pixel_deg = math.degrees(
                        math.atan2(offset_px, fx_camera))
                    msg += f" b_from_pixel={bearing_from_pixel_deg:+.0f}deg"
                print(msg)
            if relative_pose is not None:
                dist, bearing, viewing_angle = relative_pose
                if pose is not None:
                    position = global_position(pose, bearing, dist)
                row = f"{class_name} d={dist:.2f}m b={math.degrees(bearing):+.0f}deg"
                if position is not None:
                    row += f" xy=({position[0]:+.2f},{position[1]:+.2f})"
            else:
                dist = bearing = viewing_angle = None
                row = f"{class_name} (no pose estimated)"
            legend_rows.append((i + 1, colour, row))

            # Only a number on the box -- the rest of the row is in the
            # legend at the bottom.
            cx_badge, cy_badge = x + 14, y + 14
            cv2.circle(display, (cx_badge, cy_badge), 13, colour, -1)
            cv2.circle(display, (cx_badge, cy_badge), 13, (0, 0, 0), 2)
            cv2.putText(display, str(i + 1), (cx_badge - 6, cy_badge + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            if state_out is not None:
                detections_state.append(dict(
                    class_name=class_name, confidence=conf, distance=dist,
                    bearing_rad=bearing, viewing_angle_rad=viewing_angle,
                    position=list(position) if position is not None else None))

        if legend_rows:
            # Semi-transparent panel at the bottom, one row per marker,
            # numbered and coloured like its box -- NEVER overlapping
            # another row, unlike the previous per-box text: readability
            # with several nearby objects no longer depends on how far
            # apart they are in the image.
            row_height = 24
            panel_height = row_height * len(legend_rows) + 16
            y_panel = max(0, img_height - panel_height)
            overlay = display.copy()
            cv2.rectangle(overlay, (0, y_panel), (img_width, img_height),
                          (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
            for row_idx, (number, colour, text) in enumerate(legend_rows):
                y_text = y_panel + 16 + row_idx * row_height
                cv2.circle(display, (18, y_text - 5), 9, colour, -1)
                cv2.putText(display, str(number), (14, y_text - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(display, text, (34, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

        status_text = (f"frames: {n_frames}  markers seen: {len(results)}  "
                       f"cx={cx_camera:.0f}px")
        if p_d < 1.0:
            status_text += f"  p_D={p_d:.2f}"
        cv2.putText(display, status_text,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if state_out is not None:
            write_live_state(state_out, robot_id, pose, detections_state,
                             p_d=p_d)
        cv2.imshow(f"debug ArUco - {robot_id} (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"\nframes received: {n_frames}, attempts without "
          f"a frame: {n_no_frame}")
    if p_d < 1.0 and n_detected_total:
        kept = n_detected_total - n_dropped_total
        print(f"interference: {kept}/{n_detected_total} detections kept "
              f"= empirical p_D {kept / n_detected_total:.3f} "
              f"(set to {p_d:.2f})")


def _pick_representative(readings, debug=False):
    """Among several readings of the SAME physical instance (same robot,
    same trial, already grouped), pick the most representative one for
    the final log -- the distance estimate is what really matters.

    With ArUco the confidence is ALWAYS 1.0 (ARUCO_CONFIDENCE in
    aruco_detector.py: "a marker read is a marker read", no real score).
    "keep the best by confidence" is a tie there: it just picks the first
    among readings with identical confidence, NOT the one with the most
    reliable distance -- the closely spaced --n-readings (residual motion
    blur, corner detection slightly different frame by frame) were not
    reducing noise as the module docstring promises.

    When at least 2 readings have a MEASURED distance (ArUco via solvePnP
    -- not YOLO's bbox-based estimate), the one whose distance is closest
    to the group MEDIAN is chosen instead: this discards a possible single
    anomalous frame (an unfavourable angle, a reflection) without
    inventing a synthetic distance that no frame actually measured -- the
    returned Detection is always one that was REALLY observed. Without a
    measured distance (YOLO, or fewer than 2 readings with a distance) we
    fall back to comparing confidence, which is a real signal there."""
    with_distance = [r for r in readings if r.distance is not None]
    if len(with_distance) >= 2:
        distances = sorted(r.distance for r in with_distance)
        n = len(distances)
        median = (distances[n // 2] if n % 2 == 1
                  else (distances[n // 2 - 1] + distances[n // 2]) / 2)
        chosen = min(with_distance, key=lambda r: abs(r.distance - median))
        if debug:
            print(f"    [debug] representative chosen by median "
                  f"distance: {[round(d, 3) for d in distances]} -> "
                  f"median={median:.3f} -> chosen distance="
                  f"{chosen.distance:.3f}")
        return chosen
    return max(readings, key=lambda r: r.confidence)


def capture_trial(detector, n_readings=5, pause_s=0.1, instance_threshold_m=0.5,
                  debug=False):
    """n_readings closely spaced readings. If the scene contains MORE than
    one object (e.g. 2 boxes in view at the same time), the n_readings see
    ALL of them repeatedly -- this function groups them by estimated
    POSITION (same principle as object_position.py /
    the instance-separation option of fondi_log_reali.py) so the result
    is ONE reading per distinct PHYSICAL OBJECT, not one per raw reading
    (which would count the same object several times).

    Returns a LIST of Detection: normally 1 element (a single object in
    view, or none), but it can have more if the robot sees several
    physical instances together. If no reading saw anything, the list
    contains ONE "nothing seen" Detection (class_name=None) so the trial is
    still recorded for this robot, not silently missing from the file.

    Honest limitation: a detection WITHOUT an estimable position (robot
    pose unavailable at that moment) cannot be assigned to an instance --
    it is dropped rather than risking merging it with the wrong object.
    This only happens if OptiTrack loses tracking in the middle of the
    n_readings: rare, but worth knowing if a trial with several objects
    comes back with fewer objects than expected.

    debug=True prints EVERY raw reading (how many detections in the frame,
    whether each one has an estimable position or not) instead of only
    the final result -- use it when a trial with several objects in view
    (confirmed with --live) comes back with fewer instances than expected:
    it tells whether the problem is upstream (at that precise moment the
    camera saw fewer markers than --live shows, e.g. because of a slightly
    different angle/position) or downstream (the markers were there but
    with no `pose` available, or the estimated positions ended up closer
    than the --instance-threshold-m threshold despite being two distinct
    real objects)."""
    from vf.perception.object_position import cluster_by_position

    all_readings = []
    last_empty = None
    for i in range(n_readings):
        raw_readings = detector.read()
        if debug:
            print(f"  [debug] reading {i + 1}/{n_readings}: "
                  f"{len(raw_readings)} raw detection(s)")
            for r in raw_readings:
                if r.position is not None:
                    pos_str = (f"position=({r.position[0]:.3f}, "
                               f"{r.position[1]:.3f})")
                else:
                    pos_str = "position=NONE (robot pose not available now)"
                print(f"    - class_name={r.class_name} conf={r.confidence:.2f} "
                      f"distance={r.distance} {pos_str}")
        for r in raw_readings:
            if r.class_name is None:
                last_empty = r
            else:
                all_readings.append(r)
        time.sleep(pause_s)

    if debug:
        with_position = sum(1 for r in all_readings if r.position is not None)
        print(f"  [debug] total over the {n_readings} readings: {len(all_readings)} "
              f"non-empty detection(s), {with_position} with estimable "
              f"position (instance-threshold-m={instance_threshold_m})")

    if not all_readings:
        return [last_empty] if last_empty is not None else []

    positioned = [r for r in all_readings if r.position is not None]
    if len(positioned) < 2:
        # either 0 or 1 positionable reading: nothing to group by
        # instance, fall back to the simple behaviour (the best by
        # confidence among those available).
        if debug and len(all_readings) > len(positioned):
            print("  [debug] fewer than 2 detections with a position: no "
                  "per-instance grouping possible, picking the "
                  "representative among ALL raw detections (including "
                  "those without a position)")
        return [_pick_representative(all_readings, debug=debug)]

    cluster_indices = cluster_by_position(
        [r.position for r in positioned], threshold_m=instance_threshold_m)
    n_clusters = max(cluster_indices) + 1
    if debug:
        print(f"  [debug] grouped into {n_clusters} distinct physical "
              f"instance(s)")
    readings_per_cluster = [[] for _ in range(n_clusters)]
    for r, idx in zip(positioned, cluster_indices):
        readings_per_cluster[idx].append(r)
    if debug:
        for i, group in enumerate(readings_per_cluster, start=1):
            print(f"  [debug] instance {i}: {len(group)} reading(s) in the "
                  f"cluster")
    return [_pick_representative(group, debug=debug)
            for group in readings_per_cluster]


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-id", required=True,
                   help="robot name in config.ROBOTS, and robot_id in the log")
    p.add_argument("--sn", default=None)
    p.add_argument("--ip", default=None)
    p.add_argument("--live", action="store_true",
                   help="live video preview with the detected markers "
                        "drawn on top, instead of capturing a trial -- "
                        "see the module docstring. With --live, "
                        "--trial-id/--out/--regions/--localization are "
                        "not needed, UNLESS --state-out is also used")
    p.add_argument("--state-out", default=None,
                   help="ONLY with --live: continuously writes this "
                        "robot's state (pose + detections with global "
                        "position) to this JSON file, overwritten on "
                        "every frame -- read by dashboard.py for the "
                        "live merge across robots. Requires --regions "
                        "and (with OptiTrack) --optitrack-rb-id, because "
                        "the global position needs the robot pose")
    p.add_argument("--trial-id", default=None,
                   help="identifies THIS trial (same label for all "
                        "participating robots) -- ties the rows of the "
                        "different robots together in "
                        "merge_static_trials.py -- required unless "
                        "--live is used")
    p.add_argument("--out", default=None,
                   help="folder shared by the robot processes: one row "
                        "per robot per trial in <out>/<robot-id>.jsonl "
                        "-- required unless --live is used")
    p.add_argument("--regions", default=None,
                   help="see config/optitrack_regions.json -- only needed "
                        "for 'zone' in the log, not the data that matters "
                        "here (the precise position is, see "
                        "--localization) -- required unless --live is used")
    p.add_argument("--localization", choices=["odometry", "optitrack"],
                   default="optitrack",
                   help="OptiTrack by default for static trials: the "
                        "robot is stationary, the pose is needed only "
                        "once, no drift to accumulate as in patrolling "
                        "(see patrol.py)")
    p.add_argument("--optitrack-rb-id", type=int, default=None)
    p.add_argument("--optitrack-server", default="127.0.0.1")
    p.add_argument("--optitrack-local", default="127.0.0.1")
    p.add_argument("--camera-offset-forward-m", type=float, default=0.0,
                   help="how far FORWARD (in the robot's reference frame) "
                        "the camera is relative to the point tracked by "
                        "OptiTrack/odometry (e.g. the rigid body centre) "
                        "-- measure it with a ruler if the camera is "
                        "mounted on an arm and not at the tracked point: "
                        "see CameraOffsetPose in localization.py. Default "
                        "0.0 = no correction (behaviour unchanged)")
    p.add_argument("--camera-offset-right-m", type=float, default=0.0,
                   help="how far to the RIGHT (negative = to the left) the "
                        "camera is relative to the tracked point -- see "
                        "--camera-offset-forward-m")
    p.add_argument("--frame-correction-file", default=None,
                   help="JSON file produced by tools/calibrate_optitrack.py "
                        "({\"rotation_rad\":.., \"dx_m\":.., \"dy_m\":..}) "
                        "-- corrects the misalignment between THIS robot's "
                        "OptiTrack frame and the true shared frame (see "
                        "FrameCorrectedPose in localization.py). Applied "
                        "BEFORE --camera-offset-*. Omitted = no correction "
                        "(behaviour unchanged)")
    p.add_argument("--detector", choices=["aruco", "yolo"], default="aruco")
    p.add_argument("--aruco-marker-size", type=float, default=None)
    p.add_argument("--aruco-dict", default="4X4_50")
    p.add_argument("--aruco-classes", default=None)
    p.add_argument("--camera-calib", default=None)
    p.add_argument("--aruco-max-distance-m", type=float, default=8.0,
                   help="drops an ArUco detection whose reported distance "
                        "exceeds this threshold -- protects against "
                        "solvePnP's near/far ambiguity on a planar marker "
                        "(see ArucoDetector), which can make the distance "
                        "jump to an absurd value (e.g. ~20m) on an "
                        "isolated frame without the marker having moved. "
                        "Raise the value if your arena really is larger "
                        "than 8m; --aruco-max-distance-m -1 disables the "
                        "check")
    p.add_argument("--aruco-default-params", action="store_true",
                   help="use OpenCV's default DetectorParameters instead "
                        "of the relaxed ones (see "
                        "aruco_detector.relaxed_parameters()) -- only to "
                        "compare 'before/after' if you want to check how "
                        "much they really help against a marker that "
                        "'flickers in and out'")
    p.add_argument("--camera-resolution", choices=["360", "540", "720"],
                   default="720",
                   help="video stream resolution (see camera.py: 720p has "
                        "no measured cost on the link compared to 540p, "
                        "and more pixels on the marker help precisely "
                        "against 'flickering in and out' -- camera.py's "
                        "default of 540 is raised to 720 here)")
    p.add_argument("--yolo-host", default="127.0.0.1")
    p.add_argument("--yolo-port", type=int, default=8765)
    p.add_argument("--n-readings", type=int, default=5,
                   help="closely spaced readings per trial, the best one "
                        "is kept -- see capture_trial()")
    p.add_argument("--instance-threshold-m", type=float, default=0.5,
                   help="within how many metres two readings of THIS robot "
                        "are considered the same physical object instead "
                        "of two distinct instances (same default as "
                        "merge_static_trials.py)")
    p.add_argument("--debug", action="store_true",
                   help="print every raw reading during capture (how "
                        "many markers in the frame, whether each has an "
                        "estimable position) -- use it if a trial with "
                        "several objects in view (confirmed with --live) "
                        "comes back with fewer instances than expected, "
                        "see the capture_trial() docstring. With --live it "
                        "additionally prints to the terminal (no longer "
                        "on the video) the pixel offset and the "
                        "bearing-from-pixel for each marker, see "
                        "live_preview()")
    p.add_argument("--faulty", action="store_true",
                   help="designate THIS robot as faulty for this trial "
                        "(simulated systematic fault)")
    p.add_argument("--dropout-prob", type=float, default=None)
    p.add_argument("--mislabel-prob", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-d", type=float, default=1.0,
                   help="ONLY with --live: simulated interference with "
                        "constant detection probability p_D (Section 7.1 "
                        "of the document): every marker in every frame is "
                        "kept with probability p_D, otherwise treated as "
                        "not seen. One value per process = one value per "
                        "robot. Default 1.0 = no interference. Draws are "
                        "reproducible with --seed")
    args = p.parse_args()

    if not 0.0 <= args.p_d <= 1.0:
        p.error("--p-d must be between 0 and 1")
    if args.p_d < 1.0 and not args.live:
        p.error("--p-d only applies with --live (for static trials use "
                "--faulty --dropout-prob)")

    # --state-out is checked BEFORE the "not args.live" block below: if
    # --live is missing, --trial-id is usually missing too (--live and
    # normal trials are alternative modes), so checking
    # trial-id/out/regions first would show the user "--trial-id is
    # required" instead of the much clearer "--state-out only makes sense
    # together with --live" -- misleading for someone who simply forgot
    # --live.
    if args.state_out is not None:
        if not args.live:
            p.error("--state-out only makes sense together with --live")
        if args.regions is None:
            p.error("--state-out requires --regions (needed to compute "
                    "the global position)")
        if args.localization == "optitrack" and args.optitrack_rb_id is None:
            p.error("--state-out requires --optitrack-rb-id with "
                    "--localization optitrack")
    if not args.live:
        if args.trial_id is None:
            p.error("--trial-id is required (unless --live is used)")
        if args.out is None:
            p.error("--out is required (unless --live is used)")
        if args.regions is None:
            p.error("--regions is required (unless --live is used)")

    # In --live mode WITHOUT --state-out we skip regions/OptiTrack: the
    # video preview only serves to check what the camera sees and does
    # not need the robot pose (see live_preview() above). With --state-out,
    # instead, the pose is needed (for the global position), so it must be
    # set up as in the non-live path.
    needs_pose = (not args.live) or (args.state_out is not None)
    regions = None
    pose_provider = None
    if needs_pose:
        regions = load_regions(args.regions)
        if args.localization == "optitrack":
            if args.optitrack_rb_id is None:
                p.error("--optitrack-rb-id is required with "
                        "--localization optitrack")
            from vf.localization.mocap_pose import MocapPose
            from vf.localization.optitrack import OptiTrackPose
            source = MocapPose(args.optitrack_rb_id, args.optitrack_server,
                               args.optitrack_local)
            source.wait_ready()
            pose_provider = OptiTrackPose(
                pose_source=source, regions=regions, close_source=True)

    raw_detector = None
    if args.detector == "aruco":
        if args.aruco_marker_size is None:
            p.error("--aruco-marker-size is required with "
                    "--detector aruco (measure it on the printed marker, "
                    "in metres)")
        from vf.perception.aruco_detector import (
            ArucoDetector, load_aruco_classes, load_calibration,
            dictionary_from_name)
        marker_to_class = (load_aruco_classes(args.aruco_classes)
                           if args.aruco_classes else None)
        if args.camera_calib:
            camera_matrix, dist_coeffs = load_calibration(args.camera_calib)
        else:
            camera_matrix, dist_coeffs = None, None
        max_distance = (None if args.aruco_max_distance_m < 0
                        else args.aruco_max_distance_m)
        raw_detector = ArucoDetector(
            marker_size_m=args.aruco_marker_size,
            camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
            dictionary=dictionary_from_name(args.aruco_dict),
            marker_to_class=marker_to_class,
            relax_parameters=not args.aruco_default_params,
            max_plausible_distance_m=max_distance)
    else:
        from vf.perception.yolo_client import RemoteYoloDetector
        raw_detector = RemoteYoloDetector(host=args.yolo_host,
                                          port=args.yolo_port)

    from robomaster_sdk import Robot
    from robomaster_sdk.camera import Camera as CameraSDK

    with Robot(args.robot_id, sn=args.sn, ip=args.ip) as robot:
        camera = CameraSDK(robot.camera).start(
            resolution=args.camera_resolution)

        localizer = None
        if needs_pose:
            localizer = (pose_provider if pose_provider is not None
                         else OdometryLocalizer(robot, regions))
            # Corrects the misalignment between THIS robot's OptiTrack
            # frame and the true shared frame -- see FrameCorrectedPose in
            # localization.py and tools/calibrate_optitrack.py. Applied
            # BEFORE the camera offset below: the camera offset uses the
            # yaw to compute forward/right in the robot frame, so it must
            # see the pose already re-calibrated to the true frame.
            # Without --frame-correction-file it is a transparent no-op.
            if args.frame_correction_file:
                with open(args.frame_correction_file) as f:
                    correction = json.load(f)
                localizer = FrameCorrectedPose(
                    localizer,
                    rotation_rad=correction["rotation_rad"],
                    dx_m=correction["dx_m"],
                    dy_m=correction["dy_m"])
            # Corrects the offset between the tracked point (OptiTrack
            # rigid body, or odometry origin) and the TRUE camera position
            # -- see CameraOffsetPose in localization.py. With both
            # offsets at 0.0 (default) it is a transparent no-op: no
            # behaviour change for those who do not pass them.
            localizer = CameraOffsetPose(
                localizer,
                offset_forward_m=args.camera_offset_forward_m,
                offset_right_m=args.camera_offset_right_m)

        if args.live:
            try:
                pose_fn = localizer.pose if localizer is not None else None
                live_preview(camera, raw_detector, args.robot_id,
                             pose_fn=pose_fn, state_out=args.state_out,
                             debug=args.debug, p_d=args.p_d,
                             seed=args.seed)
            finally:
                camera.stop()
                if localizer is not None:
                    localizer.close()
            return

        fault_injector = FaultInjector(
            robot_id=args.robot_id,
            faulty_robot_ids=[args.robot_id] if args.faulty else [],
            dropout_prob=args.dropout_prob, mislabel_prob=args.mislabel_prob,
            seed=args.seed)
        detector = ObjectDetector(
            camera=camera, robot_id=args.robot_id,
            current_zone_fn=localizer.current_zone,
            detector=raw_detector, fault_injector=fault_injector,
            pose_fn=localizer.pose)

        # give the camera (non-blocking background reader, see
        # perception.py) time to have a first frame ready.
        time.sleep(0.5)

        try:
            readings = capture_trial(detector, n_readings=args.n_readings,
                                     instance_threshold_m=args.instance_threshold_m,
                                     debug=args.debug)
            # Camera pose at capture time, stored in every row as
            # "robot_pose": localize_trials.py uses it to know each
            # robot's field of view.
            trial_pose = localizer.pose()
        finally:
            camera.stop()
            localizer.close()

    if not readings:
        # should not happen (see capture_trial: even "nothing seen"
        # produces a Detection), but if it does, record it as a "nothing"
        # row instead of silently disappearing from the file.
        readings = [None]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.robot_id}.jsonl")
    rows = []
    for r in readings:
        rows.append(dict(
            trial_id=args.trial_id, t=time.time(), robot_id=args.robot_id,
            class_name=r.class_name if r else None,
            confidence=r.confidence if r else 0.0,
            fault_injected=r.fault_injected if r else False,
            dropped=r.dropped if r else False,
            zone=r.zone if r else "unknown",
            position=list(r.position) if (r and r.position) else None,
            distance=r.distance if r else None,
            bearing_rad=r.bearing_rad if r else None,
            viewing_angle_rad=r.viewing_angle_rad if r else None,
            # pose of the frame the marker was seen in (the one its
            # position was computed with); trial pose for "nothing seen"
            robot_pose=(list(r.robot_pose) if (r and r.robot_pose is not None)
                        else list(trial_pose) if trial_pose is not None else None)))

    # Re-running the SAME --trial-id for the SAME robot (e.g. after
    # adjusting the robot's position, or to redo a capture that looked
    # wrong) REPLACES that trial's previous rows in this file instead of
    # piling up under the old ones. Previously it was pure append:
    # running the same trial_id twice left both the old AND the new rows
    # in the same file, and merge_static_trials.py merged them as if they
    # were different readings of the same trial, giving results that
    # looked like a bug (extra instances or inconsistent positions) but
    # were just stale data never cleaned up. Other trial_ids in the same
    # file are not touched.
    other_rows = []
    n_replaced = 0
    if os.path.exists(path):
        with open(path) as f:
            for existing_row in f:
                existing_row = existing_row.strip()
                if not existing_row:
                    continue
                if json.loads(existing_row).get("trial_id") == args.trial_id:
                    n_replaced += 1
                else:
                    other_rows.append(existing_row)

    with open(path, "w") as f:
        for old_row in other_rows:
            f.write(old_row + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")

    description = "; ".join(
        f"class_name={r['class_name']} conf={r['confidence']:.2f} dist={r['distance']}"
        for r in rows)
    note = (f" (replaced {n_replaced} previous row(s) of the same "
            f"trial)" if n_replaced else "")
    print(f"trial {args.trial_id} ({args.robot_id}): {len(rows)} "
          f"object(s) -- {description}  -> {path}{note}")


if __name__ == "__main__":
    main()