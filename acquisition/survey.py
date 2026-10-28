"""Autonomous survey of a static scene: ONE robot drives by itself through a
list of stops around the boxes, and at every stop it captures what its
camera sees, exactly like static_trial.py.

To start all the robots with one command: launch_all.py (it runs this
script once per robot, at the same time).

Why the robots do not need to be synchronised
---------------------------------------------
The boxes do not move, so what robot i sees from stop k is valid at any
time. Every stop is an independent OBSERVATION POINT (observations +
camera pose). The scans that the localization methods are tested on are
composed afterwards, offline, by pipeline/build_scans.py, choosing which
observation points to put together (also more "virtual robots" than the
real ones).

Stops
-----
Rings around the centre of the box layout (--radii), one stop every
--step-deg degrees, the robot facing the centre (plus --yaw-offsets-deg:
extra captures at the same stop with the camera turned left/right, so the
boxes also fall near the edge of the field of view). Each robot starts
the rings at its own angle (--start-deg, e.g. 0 / 120 / 240 for three
robots) and all go round in the same direction, so they stay apart.
Stops closer than --margin to the arena border are skipped. The centre
of the rings comes from the ground truth of the layout
(tools/record_ground_truth.py). Paths: straight to the next stop when the
line stays at least --keep-out-m from every box, otherwise round the
layout on an arc, in the direction whose waypoints all stay inside the
arena (minus --margin); a stop with no such path is skipped.

Safety
------
Speed limit, per-stop timeout, stop when the OptiTrack pose is lost (the
robot watchdog stops it without commands), geofence (the robot is stopped
as soon as it gets closer than --margin / 2 to the arena border, and the
survey ends), wait while another robot is closer than --stop-distance and
in the way (its rigid body id in --others; moving away from it is always
allowed), Ctrl+C stops the robot. A stop that cannot be reached is logged
and skipped.

Output
------
<out>/<robot-id>.jsonl, one row per detected marker (or one "nothing
seen" row) per capture, same fields as static_trial.py plus layout_id,
stop, radius_m, angle_deg, yaw_offset_deg. trial_id =
"<layout>/<robot>/<stop>/<yaw offset>". The file is only appended to:
if the survey is interrupted, running the same command again skips the
captures already in the file.

Raw data (real robot only), for re-processing and for training a learned
model: every reading of every capture (--n-readings frames) is kept.
  <out>/frames/<robot-id>/<capture>_k<reading>.jpg   every camera frame
  <out>/raw/<robot-id>.jsonl   one row per frame: trial id, round, reading
      index, time, frame age, image file, camera and body pose at that
      frame, and ALL the markers detected in it (class, bbox, distance,
      bearing, viewing angle, position), before any grouping
  <out>/raw/meta_<layout>_<robot-id>.json   camera calibration, marker
      size and dictionary, camera offsets, capture settings and the true
      box positions of the layout
--no-frames skips the images (the raw rows are still written).

    # dry run: print the plan, no robot
    python -m acquisition.survey --robot-id P0 --layout-id L1 --ground-truth config/ground_truth_positions.json --plan-only

    # simulation: fake robot and synthetic observations from the ground truth
    # (field of view, range, occlusions between boxes, Gaussian noise on d and beta)
    python -m acquisition.survey --robot-id P0 --layout-id L1 --ground-truth config/ground_truth_positions.json --sim --out survey/

    # real robot
    python -m acquisition.survey --robot-id P0 --layout-id L1 --ground-truth config/ground_truth_positions.json \\
        --out survey/ --regions config/optitrack_regions.json --optitrack-rb-id 8 --others 7 9 \\
        --start-deg 0 --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P0.json
"""

import argparse
import faulthandler
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time

from vf.control.goto import FakeRobot, go_to, stop, wrap_pi

ARENA_DEFAULT = (-2.690, 2.135, -2.672, 1.519)

# Set when the survey must stop (launch_all.py creates the file "abort" in
# the sync folder on Ctrl+C): every loop checks it, the robot is stopped and
# the connection closed before the process exits.
STOP = threading.Event()


def watch_abort_file(path):
    """Round mode: sets STOP as soon as `path` exists (checked every 0.2 s)."""
    def run():
        while not STOP.is_set():
            if os.path.exists(path):
                STOP.set()
                break
            time.sleep(0.2)
    threading.Thread(target=run, daemon=True).start()



class PhaseWatch:
    """What this robot is doing now, for the team and for debugging.

    phase(text) records the current step (driving, capturing, saving, ...)
    and writes it to <sync dir>/status_<robot>.json, so a robot waiting for
    the team can say what the missing robot is doing. If a working step
    (not a wait for the team) lasts longer than stuck_s, the stack of every
    thread is written to <sync dir>/stuck_<robot>.txt (launch_all.py prints
    it) and, when possible, printed: it shows exactly where the process is
    stuck. The file is written by faulthandler from C, so it works even when
    Python itself is frozen or the output of this process is blocked."""

    def __init__(self, robot_id, folder, stuck_s):
        self.robot_id, self.folder, self.stuck_s = robot_id, folder, stuck_s
        self.text, self.t, self.idle, self.dumped = "starting", time.time(), True, False
        self.dump = None
        if folder:
            os.makedirs(folder, exist_ok=True)
        if folder and stuck_s > 0:
            try:
                self.dump = open(os.path.join(folder, f"stuck_{robot_id}.txt"), "a")
            except OSError:
                self.dump = None
        threading.Thread(target=self._run, daemon=True).start()

    def set(self, text, idle=False):
        self.text, self.t, self.idle, self.dumped = text, time.time(), idle, False
        if self.dump is not None:
            try:
                if idle:
                    faulthandler.cancel_dump_traceback_later()
                else:
                    self.dump.write(f"--- {time.strftime('%H:%M:%S')} {text}\n")
                    self.dump.flush()
                    faulthandler.dump_traceback_later(self.stuck_s, repeat=True, file=self.dump)
            except (OSError, ValueError):
                pass
        if self.folder:
            try:
                with open(os.path.join(self.folder, f"status_{self.robot_id}.json"), "w") as f:
                    json.dump(dict(phase=text, t=self.t, idle=idle), f)
            except OSError:
                pass

    def _run(self):
        while True:
            time.sleep(1.0)
            if self.idle or self.dumped or self.stuck_s <= 0:
                continue
            busy = time.time() - self.t
            if busy > self.stuck_s:
                self.dumped = True
                print(f"STUCK: '{self.text}' for {busy:.0f} s; where every thread of this process is now "
                      f"(send this to debug):", flush=True)
                try:
                    sys.stdout.flush()
                    faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
                    sys.stdout.flush()
                except Exception as e:
                    print(f"(stack not available: {e})", flush=True)


WATCH = None


def phase(text, idle=False):
    """Current step of this robot (see PhaseWatch); no-op without a team."""
    if WATCH is not None:
        WATCH.set(text, idle)


def read_phase(folder, robot_id):
    """(phase, seconds in it) of another robot, or None."""
    try:
        with open(os.path.join(folder, f"status_{robot_id}.json")) as f:
            q = json.load(f)
        return q["phase"], time.time() - q["t"]
    except (OSError, ValueError, KeyError):
        return None


class FrameWriter:
    """Saves the camera frames in a background thread: a slow disk (e.g. a
    network drive) never holds up the robot and the team."""

    def __init__(self):
        self.q = queue.Queue()
        self.failed = 0
        threading.Thread(target=self._run, daemon=True).start()

    def put(self, path, img, quality):
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        self.q.put((path, buf.tobytes()))
        return True

    def _run(self):
        while True:
            path, data = self.q.get()
            try:
                with open(path, "wb") as f:
                    f.write(data)
            except OSError as e:
                self.failed += 1
                print(f"frame not saved: {path} ({e})", flush=True)
            finally:
                self.q.task_done()

    def pending(self):
        return self.q.unfinished_tasks

    def flush(self, timeout_s=60.0):
        t0 = time.time()
        while self.q.unfinished_tasks and time.time() - t0 < timeout_s:
            time.sleep(0.1)
        if self.q.unfinished_tasks:
            print(f"{self.q.unfinished_tasks} frame(s) still not written after {timeout_s:.0f} s", flush=True)


FRAMES = None


def _ctrl_break(signum, frame):
    raise KeyboardInterrupt


if hasattr(signal, "SIGBREAK"):          # Windows: Ctrl+Break from the launcher -> same cleanup as Ctrl+C
    signal.signal(signal.SIGBREAK, _ctrl_break)


def _with_timeout(fn, seconds, what):
    """Runs fn() but gives up waiting after `seconds` (a stuck SDK call
    must not prevent the other cleanup steps)."""
    err = []
    th = threading.Thread(target=lambda: _call(fn, err), daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        print(f"{what}: no answer after {seconds:.0f} s, going on", flush=True)
    elif err:
        print(f"{what}: {err[0]}", flush=True)


def _call(fn, err):
    try:
        fn()
    except Exception as e:
        err.append(e)
CONNECTED_MSG = "robot connected"


# ====================================================================
# Plan
# ====================================================================

def layout_from_ground_truth(path, layout_id):
    with open(path) as f:
        data = json.load(f)
    if layout_id not in data:
        sys.exit(f"layout {layout_id!r} not in {path} (record it with tools/record_ground_truth.py)")
    boxes = data[layout_id]
    if not boxes:
        sys.exit(f"layout {layout_id!r} has no boxes")
    cx = sum(b["x"] for b in boxes) / len(boxes)
    cy = sum(b["y"] for b in boxes) / len(boxes)
    reach = max(math.hypot(b["x"] - cx, b["y"] - cy) for b in boxes)
    return boxes, (cx, cy), reach


def plan_stops(center, radii, step_deg, start_deg, arena, margin, yaw_offsets_deg):
    """List of captures: dict(stop, radius_m, angle_deg, x, y, yaw, yaw_offset_deg).
    Rings in the given order, alternating direction, so that consecutive
    stops are close; at every stop one capture per yaw offset."""
    x0, x1, y0, y1 = arena
    n = max(1, int(round(360.0 / step_deg)))
    plan, k = [], 0
    for ring, r in enumerate(radii):
        angles = [start_deg + i * 360.0 / n for i in range(n)]
        if ring % 2:
            angles = angles[::-1]
        for a in angles:
            x = center[0] + r * math.cos(math.radians(a))
            y = center[1] + r * math.sin(math.radians(a))
            if not (x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin):
                continue
            face = math.atan2(center[1] - y, center[0] - x)
            for off in yaw_offsets_deg:
                plan.append(dict(stop=k, radius_m=r, angle_deg=round(a % 360.0, 1), x=x, y=y,
                                 yaw=wrap_pi(face + math.radians(off)), yaw_offset_deg=off))
            k += 1
    return plan


def split_groups(plan, n_groups, center):
    """Splits the stops of `plan` among n_groups robots: every robot gets a
    contiguous SECTOR of angles around the layout, so the robots work far
    from each other. The sectors start after the widest empty angle (e.g.
    the side cut by the arena border), so no sector has to go round it.
    Inside a sector: ring by ring, alternating direction. Returns one list
    of captures per robot (all the yaw offsets of a stop together)."""
    by_stop = {}
    for c in plan:
        by_stop.setdefault(c["stop"], []).append(c)
    stops = sorted(by_stop, key=lambda k: by_stop[k][0]["angle_deg"])
    angles = sorted({by_stop[k][0]["angle_deg"] for k in stops})
    if len(angles) > 1:
        gaps = [((angles[(i + 1) % len(angles)] - angles[i]) % 360.0, i) for i in range(len(angles))]
        start = angles[(max(gaps)[1] + 1) % len(angles)]
    else:
        start = angles[0]
    rel = {k: (by_stop[k][0]["angle_deg"] - start) % 360.0 for k in stops}
    ordered_angles = sorted({rel[k] for k in stops})
    # cut the angles so that every sector gets about the same number of stops
    count = {ang: sum(1 for k in stops if rel[k] == ang) for ang in ordered_angles}
    owner, cum, total = {}, 0, len(stops)
    for ang in ordered_angles:
        owner[ang] = min(n_groups - 1, int((cum + count[ang] / 2.0) * n_groups / total))
        cum += count[ang]
    groups = []
    for g in range(n_groups):
        ks = [k for k in stops if owner[rel[k]] == g]
        radii = sorted({by_stop[k][0]["radius_m"] for k in ks})
        seq = []
        for i, r in enumerate(radii):
            ring = sorted((k for k in ks if by_stop[k][0]["radius_m"] == r), key=lambda k: rel[k],
                          reverse=bool(i % 2))
            for k in ring:
                seq += by_stop[k]
        groups.append(seq)
    return groups


def border_distance(p, arena):
    x0, x1, y0, y1 = arena
    return min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1])


def inside(p, arena, margin):
    x0, x1, y0, y1 = arena
    return x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin


def segment_clear(a, b, boxes, clearance_m):
    """True if the segment a-b stays at least clearance_m from every box."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    for box in boxes:
        px, py = box["x"] - ax, box["y"] - ay
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, (px * dx + py * dy) / L2))
        if math.hypot(px - t * dx, py - t * dy) < clearance_m:
            return False
    return True


def robots_clear(a, b, robots, clearance_m):
    """True if the segment a-b does not bring the robot closer than
    clearance_m to any of the (parked) robots in `robots` [(x, y), ...].
    Moving away from a robot that is already close is allowed: the leg is
    rejected only if it gets closer than clearance_m AND closer than it
    already is at its start."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    for rx, ry in robots:
        px, py = rx - ax, ry - ay
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, (px * dx + py * dy) / L2))
        d = math.hypot(px - t * dx, py - t * dy)
        if d < clearance_m and d < math.hypot(px, py) - 1e-6:
            return False
    return True


def route(cur_xy, goal_xy, center, boxes, clearance_m, arena, margin, step_deg=15.0,
          robots=(), robot_clearance_m=0.7):
    """Waypoints from cur to goal that never pass closer than clearance_m to
    a box and never leave the arena (minus margin).

    Straight line when it is clear of the boxes (the arena is a rectangle:
    a segment between two points inside it stays inside). Otherwise round
    the layout on an arc about its centre, trying both directions and
    taking the shorter one whose waypoints are all inside the arena and
    whose legs are all clear of the boxes. A robot that starts too close
    to a box first moves straight away from the layout centre. `robots`
    are the positions of the other (standing) robots: no leg may pass
    closer than robot_clearance_m to them; the arc is also tried further
    out to go round them. None if no path works."""
    def clear(a, b):
        return segment_clear(a, b, boxes, clearance_m) and robots_clear(a, b, robots, robot_clearance_m)

    if clear(cur_xy, goal_xy):
        return [tuple(goal_xy)]
    cx, cy = center
    reach = max(math.hypot(b["x"] - cx, b["y"] - cy) for b in boxes)
    if not segment_clear(cur_xy, cur_xy, boxes, clearance_m):
        # already too close to a box: first move straight away from the
        # layout centre, out of the clearance, then plan from there
        ux, uy = cur_xy[0] - cx, cur_xy[1] - cy
        n = math.hypot(ux, uy) or 1.0
        out = (cx + ux / n * (reach + clearance_m + 0.05), cy + uy / n * (reach + clearance_m + 0.05))
        if not inside(out, arena, margin) or not segment_clear(out, out, boxes, clearance_m):
            return None
        if not robots_clear(cur_xy, out, robots, robot_clearance_m):
            return None
        rest = route(out, goal_xy, center, boxes, clearance_m, arena, margin, step_deg, robots, robot_clearance_m)
        return None if rest is None else [out] + rest
    ra = math.hypot(cur_xy[0] - cx, cur_xy[1] - cy)
    rb = math.hypot(goal_xy[0] - cx, goal_xy[1] - cy)
    # arc as close to the boxes as the clearance allows (+5 cm of slack),
    # but not inside the smaller of the two ring radii
    r = max(reach + clearance_m + 0.05, min(ra, rb))
    a0 = math.atan2(cur_xy[1] - cy, cur_xy[0] - cx)
    a1 = math.atan2(goal_xy[1] - cy, goal_xy[0] - cx)
    short = wrap_pi(a1 - a0)
    options = []
    arc_radii = [r] + ([r + robot_clearance_m] if robots else [])
    for r_arc in arc_radii:
        for da in (short, short - math.copysign(2 * math.pi, short) if short else 2 * math.pi):
            n = max(1, int(math.ceil(abs(math.degrees(da)) / step_deg)))
            # intermediate points on the arc, the ends at the start and goal
            # radius, never inside reach + clearance
            pts = []
            for i in range(n + 1):
                f = i / n
                rr = max(r_arc if 0 < i < n else (ra if i == 0 else rb), reach + clearance_m)
                pts.append((cx + rr * math.cos(a0 + da * f), cy + rr * math.sin(a0 + da * f)))
            path = [tuple(cur_xy)] + pts + [tuple(goal_xy)]
            if not all(inside(q, arena, margin) for q in pts[1:-1]):  # not the start: the robot is there
                continue
            if not all(clear(path[k], path[k + 1]) for k in range(len(path) - 1)):
                continue
            length = sum(math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                         for k in range(len(path) - 1))
            options.append((length, pts[1:] + [tuple(goal_xy)]))
    if not options:
        return None
    return min(options)[1]


# ====================================================================
# Simulated sensing (--sim)
# ====================================================================

def occluded(target, boxes, cam_xy, box_side_m):
    """True if another box stands between the camera and `target`: it is
    closer to the camera and its centre is less than one box side from the
    camera-target line (the two boxes overlap in the image)."""
    tx, ty = target["x"] - cam_xy[0], target["y"] - cam_xy[1]
    d = math.hypot(tx, ty)
    for o in boxes:
        if o is target:
            continue
        ox, oy = o["x"] - cam_xy[0], o["y"] - cam_xy[1]
        along = (ox * tx + oy * ty) / d
        if 0 < along < d and abs(ox * ty - oy * tx) / d < box_side_m:
            return True
    return False


def simulated_rows(boxes, pose, hfov_deg=88.0, r_min=0.15, r_max=3.0,
                   sigma_d=0.03, sigma_b=math.radians(1.0), box_side_m=0.20, rng=random):
    """Synthetic observations of the ground-truth boxes from a camera pose:
    every box in the field of view and not hidden behind another box
    (occluded), distance and bearing with Gaussian noise."""
    x, y, psi = pose
    rows = []
    for b in boxes:
        dx, dy = b["x"] - x, b["y"] - y
        d = math.hypot(dx, dy)
        beta = wrap_pi(psi - math.atan2(dy, dx))
        if not (r_min <= d <= r_max and abs(beta) <= math.radians(hfov_deg) / 2):
            continue
        if box_side_m > 0 and occluded(b, boxes, (x, y), box_side_m):
            continue
        rows.append(dict(class_name=b["class_name"], confidence=1.0,
                         distance=d + rng.gauss(0, sigma_d), bearing_rad=beta + rng.gauss(0, sigma_b),
                         viewing_angle_rad=None))
    return rows


# ====================================================================
# Main
# ====================================================================

def draw_plan(path, arena, margin, boxes, center, plan, legs):
    """Top view of the plan: arena, margin, boxes, stops (numbered, with the
    camera direction) and the planned paths between them."""
    # matplotlib Figure without pyplot: the backend of the program that
    # calls this (e.g. the live window of launch_all.py) is not touched
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    x0, x1, y0, y1 = arena
    fig = Figure(figsize=(7, 7 * (y1 - y0) / (x1 - x0)))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="k"))
    ax.add_patch(Rectangle((x0 + margin, y0 + margin), x1 - x0 - 2 * margin, y1 - y0 - 2 * margin,
                               fill=False, color="0.6", ls="--"))
    for b in boxes:
        ax.add_patch(Rectangle((b["x"] - 0.1, b["y"] - 0.1), 0.2, 0.2, color="tab:orange"))
    ax.plot(*center, "k+")
    for pts in legs:
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color="tab:blue", lw=0.8)
    seen = set()
    for c in plan:
        ax.annotate("", xy=(c["x"] + 0.3 * math.cos(c["yaw"]), c["y"] + 0.3 * math.sin(c["yaw"])),
                    xytext=(c["x"], c["y"]), arrowprops=dict(arrowstyle="->", color="tab:green"))
        if c["stop"] not in seen:
            seen.add(c["stop"])
            ax.plot(c["x"], c["y"], "o", color="tab:green")
            ax.text(c["x"] + 0.05, c["y"] + 0.05, str(c["stop"]), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.1, x1 + 0.1)
    ax.set_ylim(y0 - 0.1, y1 + 0.1)
    ax.set_xlabel("x [m] (OptiTrack planar frame)")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def done_trials(path, rounds_only=False):
    """trial_ids already in the file (to resume). rounds_only: only the
    captures made in round mode (rows with a "round"), so data of an older
    survey in the same file is not mistaken for this one."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if not rounds_only or r.get("round") is not None:
                        done.add(r["trial_id"])
    return done


def _jsonable(v):
    """json.dumps fallback for numpy values."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    raise TypeError(f"not JSON serializable: {type(v).__name__}")


class RecordingCamera:
    """Wraps the camera: read() is passed through, and the frame it returns
    is kept together with the camera and body pose at that moment."""

    def __init__(self, camera, camera_pose, body_pose):
        self.camera = camera
        self.camera_pose = camera_pose
        self.body_pose = body_pose
        self.last = None

    def read(self):
        img, age = self.camera.read()
        self.last = dict(t=time.time(), img=img, age=age,
                         camera_pose=self.camera_pose(), body_pose=self.body_pose())
        return img, age


class RecordingDetector:
    """Wraps ObjectDetector for capture_trial(): every read() is kept as one
    frame record (image, poses, all its detections), in reading order."""

    def __init__(self, detector, cam_rec):
        self.detector = detector
        self.cam_rec = cam_rec
        self.frames = []

    def read(self):
        self.cam_rec.last = None
        dets = self.detector.read()
        frame = dict(self.cam_rec.last or dict(t=time.time(), img=None, age=None,
                                                camera_pose=None, body_pose=None))
        frame["detections"] = [
            dict(class_name=d.class_name, confidence=d.confidence,
                 bbox=list(d.bbox) if d.bbox is not None else None,
                 distance=d.distance, bearing_rad=d.bearing_rad, viewing_angle_rad=d.viewing_angle_rad,
                 position=list(d.position) if d.position is not None else None)
            for d in dets if d.class_name is not None]
        self.frames.append(frame)
        return dets


def save_raw_readings(a, trial_id, extra, frames):
    """Writes every reading of one capture: the image in
    <out>/frames/<robot>/ (in the background, FrameWriter) and one row per
    frame in <out>/raw/<robot>.jsonl. Returns a short note for the log."""
    global FRAMES
    if FRAMES is None:
        FRAMES = FrameWriter()
    raw_dir = os.path.join(a.out, "raw")
    img_dir = os.path.join(a.out, "frames", a.robot_id)
    os.makedirs(raw_dir, exist_ok=True)
    rnd = (extra or {}).get("round")
    base = trial_id.replace("/", "_") + (f"_r{rnd}" if rnd is not None else "")
    n_img = 0
    rows = []
    for k, f in enumerate(frames):
        image = None
        img = f.get("img")
        if img is not None and not a.no_frames:
            os.makedirs(img_dir, exist_ok=True)
            name = f"{base}_k{k}.jpg"
            if FRAMES.put(os.path.join(img_dir, name), img, a.jpeg_quality):
                image = f"frames/{a.robot_id}/{name}"
                n_img += 1
        rows.append(dict(
            trial_id=trial_id, layout_id=a.layout_id, robot_id=a.robot_id, reading=k, t=f["t"],
            frame_age_s=f["age"], image=image,
            width=None if img is None else int(img.shape[1]),
            height=None if img is None else int(img.shape[0]),
            camera_pose=None if f["camera_pose"] is None else list(f["camera_pose"]),
            body_pose=None if f["body_pose"] is None else list(f["body_pose"]),
            n_detections=len(f["detections"]), detections=f["detections"], **(extra or {})))
    with open(os.path.join(raw_dir, f"{a.robot_id}.jsonl"), "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_jsonable) + "\n")
    shot = [f["img"] for f in frames if f.get("img") is not None]
    if not shot:
        return "NO CAMERA FRAME"
    queued = FRAMES.pending()
    return (f"{len(shot)}/{len(frames)} frame(s) {shot[0].shape[1]}x{shot[0].shape[0]}"
            + (f", saved in frames/{a.robot_id}/" if n_img else "")
            + (f" ({queued} still being written: slow disk)" if queued > 3 * len(frames) else ""))


def write_raw_meta(a, boxes, camera_matrix, dist_coeffs):
    """Everything needed to re-process the saved frames or label them:
    <out>/raw/meta_<layout>_<robot>.json (overwritten at every start)."""
    raw_dir = os.path.join(a.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    meta = dict(
        robot_id=a.robot_id, layout_id=a.layout_id, written=time.strftime("%Y-%m-%d %H:%M:%S"),
        camera_resolution=a.camera_resolution, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
        camera_calib_file=a.camera_calib, aruco_marker_size_m=a.aruco_marker_size,
        aruco_dict=a.aruco_dict, aruco_classes_file=a.aruco_classes,
        camera_offset_forward_m=a.camera_offset_forward_m, camera_offset_right_m=a.camera_offset_right_m,
        frame_correction_file=a.frame_correction_file, yaw_offset_deg=a.yaw_offset_deg,
        optitrack_rb_id=a.optitrack_rb_id, n_readings=a.n_readings,
        instance_threshold_m=a.instance_threshold_m, settle_s=a.settle_s, jpeg_quality=a.jpeg_quality,
        ground_truth_file=a.ground_truth, true_boxes=boxes,
        pose_convention="(x, y, yaw_rad) in the OptiTrack planar frame; camera_pose = body pose "
                        "moved by the camera offsets",
        detection_convention="distance = camera->marker [m]; bearing_rad positive to the right of "
                             "the optical axis; viewing_angle_rad 0 = marker facing the camera")
    with open(os.path.join(raw_dir, f"meta_{a.layout_id}_{a.robot_id}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)


class RobotHandle:
    """Everything needed to drive ONE robot and capture with it, real or
    simulated: chassis, body_pose(), camera_pose(), capture(trial_id),
    close(). `others` (optional) -> positions of the other robots."""

    def __init__(self, a, boxes, start_pose=None):
        self.a = a
        self.boxes = boxes
        self.rng = random.Random(a.seed)
        self.robot = self.camera = self.hub = self.detector = None
        self.others = None
        if a.sim:
            x, y, yaw = start_pose
            self.fake = FakeRobot(x, y, yaw, speedup=a.sim_speedup)
            self.chassis, self.body_pose, self.camera_pose = self.fake, self.fake.pose, self.fake.pose
            print(CONNECTED_MSG, flush=True)
            return
        if a.optitrack_rb_id is None or a.aruco_marker_size is None:
            raise SystemExit("--optitrack-rb-id and --aruco-marker-size are required (or use --sim)")
        from robomaster_sdk import Robot
        from robomaster_sdk.camera import Camera as CameraSDK
        from vf.localization.mocap_pose import MocapHub
        from vf.localization.localization import CameraOffsetPose, FrameCorrectedPose
        from vf.perception.aruco_detector import (ArucoDetector, dictionary_from_name,
                                                  load_aruco_classes, load_calibration)
        from vf.perception.perception import ObjectDetector

        self.hub = hub = MocapHub([a.optitrack_rb_id] + list(a.others), a.optitrack_server, a.optitrack_local)
        hub.wait_ready()
        yaw_offset = math.radians(a.yaw_offset_deg)

        def body_pose():
            """Robot pose from OptiTrack, yaw corrected by --yaw-offset-deg
            (tools/check_motion.py measures it)."""
            q = hub.get_pose(a.optitrack_rb_id)
            return None if q is None else (q[0], q[1], wrap_pi(q[2] + yaw_offset))

        def others():
            out = []
            for rb in a.others:
                q = hub.get_pose(rb)
                if q is not None:
                    out.append(q[:2])
            return out

        class _Provider:                       # pose-provider interface of localization.py
            def pose(self):
                return body_pose()

            def position(self):
                q = self.pose()
                return None if q is None else q[:2]

            def current_zone(self):
                return "unknown"

            def close(self):
                pass

        localizer = _Provider()
        if a.frame_correction_file:
            with open(a.frame_correction_file) as f:
                corr = json.load(f)
            localizer = FrameCorrectedPose(localizer, rotation_rad=corr["rotation_rad"],
                                           dx_m=corr["dx_m"], dy_m=corr["dy_m"])
        localizer = CameraOffsetPose(localizer, offset_forward_m=a.camera_offset_forward_m,
                                     offset_right_m=a.camera_offset_right_m)
        self.body_pose = body_pose
        self.camera_pose = localizer.pose
        self.others = others if a.others else None
        cm, dc = load_calibration(a.camera_calib) if a.camera_calib else (None, None)
        raw_detector = ArucoDetector(
            marker_size_m=a.aruco_marker_size, camera_matrix=cm, dist_coeffs=dc,
            dictionary=dictionary_from_name(a.aruco_dict),
            marker_to_class=load_aruco_classes(a.aruco_classes) if a.aruco_classes else None,
            max_plausible_distance_m=None if a.aruco_max_distance_m < 0 else a.aruco_max_distance_m)
        self.chassis = None
        try:
            self.robot = Robot(a.robot_id, sn=a.sn, ip=a.ip)
            self.robot.connect()
            print(CONNECTED_MSG, flush=True)   # launch_all.py waits for this before the next robot
            self.chassis = self.robot.chassis
            self.camera = CameraSDK(self.robot.camera).start(resolution=a.camera_resolution)
            self.cam_rec = RecordingCamera(self.camera, self.camera_pose, self.body_pose)
            self.detector = ObjectDetector(camera=self.cam_rec, robot_id=a.robot_id,
                                           current_zone_fn=lambda: "unknown",
                                           detector=raw_detector, pose_fn=self.camera_pose)
            time.sleep(0.5)
            write_raw_meta(a, boxes, cm, dc)
        except BaseException:
            self.close()                       # never leave the robot connected
            raise

    def capture(self, trial_id, extra=None):
        """(rows, note) of one capture at the current stop. On the real
        robot every reading (frame + all its detections) is also saved, see
        save_raw_readings()."""
        a = self.a
        if a.sim:
            pose = self.camera_pose()
            rows = simulated_rows(self.boxes, pose, box_side_m=a.sim_box_side, rng=self.rng)
            for r in rows:
                r["robot_pose"] = list(pose)
            return rows or [dict(class_name=None, confidence=0.0, distance=None, bearing_rad=None,
                                 viewing_angle_rad=None, robot_pose=list(pose))], ""
        from acquisition.static_trial import capture_trial
        rec = RecordingDetector(self.detector, self.cam_rec)
        phase(f"capturing {trial_id} (camera + marker detection)")
        t0 = time.time()
        readings = capture_trial(rec, n_readings=a.n_readings,
                                 instance_threshold_m=a.instance_threshold_m)
        t1 = time.time()
        phase(f"saving {trial_id}")
        note = save_raw_readings(a, trial_id, extra, rec.frames)
        note += f"; capture {t1 - t0:.1f} s, save {time.time() - t1:.1f} s"
        pose_now = self.camera_pose()
        rows = []
        for r in readings or [None]:
            rows.append(dict(
                class_name=r.class_name if r else None, confidence=r.confidence if r else 0.0,
                position=list(r.position) if (r and r.position) else None,
                distance=r.distance if r else None, bearing_rad=r.bearing_rad if r else None,
                viewing_angle_rad=r.viewing_angle_rad if r else None,
                robot_pose=(list(r.robot_pose) if (r and r.robot_pose is not None)
                            else list(pose_now) if pose_now is not None else None)))
        return rows, note

    def close(self):
        """Stops the robot (zero speed, sent a few times), then closes the
        camera stream, the robot connection and OptiTrack. Every step is
        tried even if a previous one fails or hangs."""
        chassis = getattr(self, "chassis", None)
        if chassis is not None:
            for _ in range(3):
                stop(chassis)
                time.sleep(0.05)
        if self.camera is not None:
            _with_timeout(self.camera.stop, 5.0, "camera stop")
            self.camera = None
        if self.robot is not None:
            _with_timeout(self.robot.disconnect, 8.0, "robot disconnect")
            self.robot = None
            print("robot disconnected", flush=True)
        if self.hub is not None:
            _with_timeout(self.hub.close, 3.0, "OptiTrack close")
            self.hub = None
        if FRAMES is not None and FRAMES.pending():
            print(f"writing the last {FRAMES.pending()} frame(s)...", flush=True)
            FRAMES.flush()


def visit_stop(a, rh, captures, done, out_path, center, boxes, arena, robots=(), log=print, extra=None,
               in_place=False):
    """Drives to one stop and makes its captures (one per yaw offset) not
    already in `done`. robots: positions of the other, standing robots.
    Returns "ok", "done" (nothing left to do), "blocked" (no safe path now,
    e.g. another robot in the way), "failed" (stop not reached) or "fence"
    (robot too close to the arena border: the survey must stop)."""
    todo = [c for c in captures
            if f"{a.layout_id}/{a.robot_id}/{c['stop']:03d}/{c['yaw_offset_deg']:+.0f}" not in done]
    if not todo:
        return "done"
    c0 = todo[0]
    tag = f"{a.layout_id}/{a.robot_id}/{c0['stop']:03d}"
    here = rh.body_pose()
    if here is None:
        log(f"[{tag}] no pose")
        return "failed"
    # geofence: never closer than margin/2 to the border; a robot that
    # already starts closer (placed there) may only move further inside
    limit = min(a.margin / 2, border_distance(here, arena) - 0.03)
    if limit < 0.05:
        log(f"[{tag}] robot at {border_distance(here, arena):.2f} m from the arena border: survey stopped "
            f"(move it further inside and run the same command again)")
        return "fence"

    def fence(q):
        return in_place or border_distance(q, arena) >= limit

    if in_place:                     # round 0: only turn on the spot where the robot was placed
        waypoints = [(c0["x"], c0["y"])]
    else:
        waypoints = route(here[:2], (c0["x"], c0["y"]), center, boxes, a.keep_out_m, arena, a.margin,
                          robots=robots, robot_clearance_m=getattr(a, "robot_clearance", 0.7))
    if waypoints is None:
        log(f"[{tag}] no path clear of boxes{' and robots' if robots else ''} inside the arena")
        return "blocked"
    for c in todo:
        if STOP.is_set():
            return "stopped"
        trial_id = f"{a.layout_id}/{a.robot_id}/{c['stop']:03d}/{c['yaw_offset_deg']:+.0f}"
        legs = waypoints if c is c0 else [(c["x"], c["y"])]
        for i_leg, wp in enumerate(legs):
            last = wp == (c["x"], c["y"])
            phase(f"driving to {trial_id}" + (f" (leg {i_leg + 1}/{len(legs)})" if len(legs) > 1 else ""))
            ok = go_to(rh.body_pose, rh.chassis, wp, c["yaw"] if last else None,
                       others=rh.others, stop_distance_m=a.stop_distance,
                       fence=fence, pos_tol=0.15 if in_place else 0.05, cancel=STOP.is_set, log=log)
            if STOP.is_set():
                return "stopped"
            if not ok:
                log(f"[{trial_id}] stop not reached")
                return "failed"
        phase(f"arrived at {trial_id}, settling")
        time.sleep(a.settle_s / (a.sim_speedup if a.sim else 1.0))
        rows, note = rh.capture(trial_id, extra)
        phase(f"writing {trial_id}")
        with open(out_path, "a") as f:
            for r in rows:
                r.update(trial_id=trial_id, t=time.time(), robot_id=a.robot_id, layout_id=a.layout_id,
                         stop=c["stop"], radius_m=c["radius_m"], angle_deg=c["angle_deg"],
                         yaw_offset_deg=c["yaw_offset_deg"], fault_injected=False, dropped=False,
                         zone="unknown", **(extra or {}))
                f.write(json.dumps(r) + "\n")
        done.add(trial_id)
        seen = sum(1 for r in rows if r.get("class_name"))
        q = rh.body_pose()
        at = "" if q is None else (f" at ({q[0]:+.2f}, {q[1]:+.2f}, {math.degrees(q[2]):+.0f} deg), "
                                   f"planned ({c['x']:+.2f}, {c['y']:+.2f}, {math.degrees(c['yaw']):+.0f} deg)")
        log(f"[{trial_id}] {seen} marker(s){at}" + (f"; {note}" if note else ""))
    return "ok"


def add_survey_arguments(p):
    """Arguments shared by survey.py and survey_team.py."""
    p.add_argument("--layout-id", required=True, help="box layout, key of --ground-truth")
    p.add_argument("--ground-truth", required=True, help="tools/record_ground_truth.py output")
    p.add_argument("--out", default="data/survey/")
    p.add_argument("--radii", type=float, nargs="+", default=[1.0, 1.4, 1.8],
                   help="ring radii around the layout centre [m]")
    p.add_argument("--step-deg", type=float, default=30.0, help="angle between stops on a ring")
    p.add_argument("--yaw-offsets-deg", type=float, nargs="+", default=[0.0],
                   help="captures per stop: camera turned by these angles from the centre "
                        "(default only 0: facing the boxes)")
    p.add_argument("--sweep-deg", type=float, default=None,
                   help="round mode: total angle every robot travels on its arc (default: 360 / number of "
                        "robots with --radius, so the team covers the whole circle once; 180 otherwise)")
    p.add_argument("--radius", type=float, default=None,
                   help="round mode: FIXED positions: the robots stand on a circle of this radius [m] around "
                        "the centre of the boxes (same distance and angles for every layout), evenly spaced, and rotate on it; each robot "
                        "first drives from where it was placed to the nearest free position. Without it the "
                        "robots start where they are placed and rotate around the boxes")
    p.add_argument("--circle-center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="with --radius: centre of the circle [m] (default: centre of the boxes of the layout, "
                        "recomputed for every layout)")
    p.add_argument("--first-angle-deg", type=float, default=0.0,
                   help="with --radius: angle of the first position on the circle (0 = +x direction)")
    p.add_argument("--run-id", default=None, help="(launch_all.py) id of this run, written in every row")
    p.add_argument("--direction", choices=["ccw", "cw"], default="ccw",
                   help="round mode: all the robots turn this way around the boxes")
    p.add_argument("--keep-out-m", type=float, default=0.45,
                   help="clearance from the boxes [m]: the robot never passes closer than this to a box, "
                        "and the rings must be farther than (farthest box + this) from the centre")
    p.add_argument("--margin", type=float, default=0.5,
                   help="clearance from the arena border [m]: no stop and no waypoint closer than this; "
                        "the robot is stopped if it gets closer than half of it")
    p.add_argument("--regions", default=None, help="regions JSON, for the arena bounds")
    p.add_argument("--settle-s", type=float, default=1.0, help="wait after arriving, before capturing")
    p.add_argument("--n-readings", type=int, default=5)
    p.add_argument("--no-frames", action="store_true",
                   help="do not save the camera images in <out>/frames/ (the raw rows are still written)")
    p.add_argument("--jpeg-quality", type=int, default=95, help="quality of the saved frames (1-100)")
    p.add_argument("--instance-threshold-m", type=float, default=0.5)
    p.add_argument("--stop-distance", type=float, default=0.6,
                   help="stand still while another robot is closer than this and in the way [m]")
    p.add_argument("--plan-only", action="store_true", help="print the plan and exit")
    p.add_argument("--plan-png", default=None, help="with --plan-only: also draw the plan in this PNG")
    p.add_argument("--sim", action="store_true", help="fake robot + synthetic observations, no hardware")
    p.add_argument("--sim-speedup", type=float, default=5.0,
                   help="simulated time / wall time with --sim (below 10, see FakeRobot)")
    p.add_argument("--sim-box-side", type=float, default=0.20,
                   help="with --sim: box side [m], for occlusions (0 = boxes never hide each other)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optitrack-server", default="127.0.0.1")
    p.add_argument("--optitrack-local", default="127.0.0.1")
    p.add_argument("--camera-offset-forward-m", type=float, default=0.0)
    p.add_argument("--camera-offset-right-m", type=float, default=0.0)
    p.add_argument("--frame-correction-file", default=None)
    p.add_argument("--aruco-marker-size", type=float, default=None)
    p.add_argument("--aruco-dict", default="4X4_50")
    p.add_argument("--aruco-classes", default=None)
    p.add_argument("--aruco-max-distance-m", type=float, default=8.0)
    p.add_argument("--camera-resolution", choices=["360", "540", "720"], default="720")


def load_plan(a, start_deg=0.0, quiet=False, need_plan=True):
    """Layout, arena and full plan (rings of stops) from the arguments.
    need_plan=False (round mode, where the robots move on arcs from their
    starting positions): the rings are not needed."""
    boxes, center, reach = layout_from_ground_truth(a.ground_truth, a.layout_id)
    keep_out = reach + a.keep_out_m
    if a.regions:
        from vf.fusion.sensor_model import arena_from_regions
        arena = arena_from_regions(a.regions)
    else:
        arena = ARENA_DEFAULT
    if not need_plan:
        if not quiet:
            print(f"layout {a.layout_id}: {len(boxes)} box(es), centre ({center[0]:+.2f}, {center[1]:+.2f})")
        return boxes, center, arena, []
    radii = [r for r in a.radii if r > keep_out]
    if len(radii) < len(a.radii) and not quiet:
        print(f"radii {[r for r in a.radii if r <= keep_out]} removed: inside the keep-out circle "
              f"({keep_out:.2f} m = farthest box {reach:.2f} m from the centre + --keep-out-m {a.keep_out_m:.2f})")
    plan = plan_stops(center, radii, a.step_deg, start_deg, arena, a.margin, a.yaw_offsets_deg)
    if not quiet:
        print(f"layout {a.layout_id}: {len(boxes)} box(es), centre ({center[0]:+.2f}, {center[1]:+.2f}), "
              f"keep-out {keep_out:.2f} m; {len({c['stop'] for c in plan})} stops, {len(plan)} captures")
    if not plan:
        sys.exit(f"no stop to visit: use --radii larger than {keep_out:.2f} m that still fit in the arena, "
                 f"a smaller --keep-out-m, or boxes closer to each other (check with --plan-only)")
    return boxes, center, arena, plan


def sim_start(first, center):
    """Fake robot start: 0.2 m from the first stop, along the ring, turned by 90 degrees."""
    tx, ty = -(first["y"] - center[1]), first["x"] - center[0]
    k = 0.2 / max(math.hypot(tx, ty), 1e-6)
    return first["x"] + k * tx, first["y"] + k * ty, wrap_pi(first["yaw"] + math.pi / 2)


# ====================================================================
# Rounds: all the robots move together, one formation at a time
# ====================================================================

def split_by_starts(plan, starts, center):
    """Sectors from where the robots ACTUALLY are at the start.

    starts: {robot_id: (x, y)}. Seen from the layout centre, every robot
    owns the stops NEAREST to it in angle: its sector goes from halfway to
    the previous robot to halfway to the next one, so every robot starts
    inside its own sector and no side is left to a robot that cannot use
    it. Order inside a sector (the same pattern for every robot, so that in
    each round they all turn the same way and do not run into each other):
    first ring from the start counter-clockwise to the end of the sector,
    the next ring back clockwise over the whole sector, and so on; the part
    of the first ring on the other side of the start comes last. A stop
    where the robot already stands (within 0.15 m) is left out: round 0
    covers it. Returns {robot_id: [captures in order]}."""
    cx, cy = center
    ang = {r: math.degrees(math.atan2(p[1] - cy, p[0] - cx)) % 360.0 for r, p in starts.items()}
    order = sorted(ang, key=lambda r: ang[r])
    n = len(order)
    by_stop = {}
    for c in plan:
        by_stop.setdefault(c["stop"], []).append(c)
    radii = sorted({by_stop[k][0]["radius_m"] for k in by_stop})
    out = {}
    for i, r in enumerate(order):
        if n > 1:
            to_next = (ang[order[(i + 1) % n]] - ang[r]) % 360.0 or 360.0
            from_prev = (ang[r] - ang[order[(i - 1) % n]]) % 360.0 or 360.0
            hi, lo = to_next / 2.0, -from_prev / 2.0
        else:
            hi, lo = 180.0, -180.0
        sx, sy = starts[r]
        d = {}
        for k, caps in by_stop.items():
            dk = (caps[0]["angle_deg"] - ang[r] + 180.0) % 360.0 - 180.0
            if lo <= dk < hi and math.hypot(caps[0]["x"] - sx, caps[0]["y"] - sy) > 0.15:
                d[k] = dk

        def ring(rad, cond, reverse):
            ks = [k for k in d if by_stop[k][0]["radius_m"] == rad and cond(d[k])]
            return sorted(ks, key=lambda k: d[k], reverse=reverse)

        seq = ring(radii[0], lambda v: v >= 0, False) if radii else []     # ccw, away from the start
        ccw = True
        for rad in radii[1:]:                                              # whole sector, alternating
            ccw = not ccw
            seq += ring(rad, lambda v: True, not ccw)
        if radii:                                                          # first ring, other side
            seq += ring(radii[0], lambda v: v < 0, ccw)
        out[r] = [c for k in seq for c in by_stop[k]]
    return out


def team_arcs(starts, team, center, boxes, arena, a, min_sep_m=0.7, radial_search=True):
    """Round mode: every robot moves on an ARC around the layout centre,
    starting from where it was placed. In round k (k = 1, 2, ...) robot i
    goes to its start turned by k * --step-deg about the centre
    (--direction ccw or cw), at the same distance from the centre, facing
    the centre; --sweep-deg is the total angle (180 = half circle). All the
    robots turn by the same angle in every round, so the formation rotates
    as a whole and the robots keep their spacing; they can be placed close
    to each other.

    radial_search=False (fixed positions, --radius): a point that does not
    fit is never moved, the robot stays where it is in that round.

    Where the point does not fit (closer than --keep-out-m to a box, closer
    than --margin to the arena border, or closer to where another robot
    goes in the same round than min_sep_m, or than 90% of how close the two
    were placed if that was already less) it is moved along the same
    direction to the nearest distance from the centre that fits; only if
    nothing fits the robot stays where it is in that round. Every robot
    computes the plan of the whole team (same starts, same rules), so they
    all agree on it.

    starts: {robot_id: (x, y)}; team: ids in a fixed order. Returns
    {robot_id: [[captures of round 1], [round 2], ...]} ([] = stay)."""
    cx, cy = center
    sign = 1.0 if a.direction == "ccw" else -1.0
    n = int(round(a.sweep_deg / a.step_deg))
    r0 = {r: math.hypot(starts[r][0] - cx, starts[r][1] - cy) for r in team}
    th0 = {r: math.atan2(starts[r][1] - cy, starts[r][0] - cx) for r in team}
    # two robots never get closer than min_sep_m, or than 90% of how close
    # they were placed if that was already less
    sep = {(p, q): min(min_sep_m, 0.9 * math.hypot(starts[p][0] - starts[q][0], starts[p][1] - starts[q][1]))
           for p in team for q in team if p != q}

    def candidates(r, th):
        """Points on the direction th that fit boxes and arena, nearest
        distance from the centre first."""
        pts = []
        steps = (sorted({round(i * 0.025, 3) for i in range(-int(r0[r] / 0.025), 121)}, key=abs)
                 if radial_search else [0.0])
        for dr in steps:
            rr = r0[r] + dr
            if rr <= 0.2:
                continue
            x, y = cx + rr * math.cos(th), cy + rr * math.sin(th)
            if inside((x, y), arena, a.margin) and segment_clear((x, y), (x, y), boxes, a.keep_out_m):
                pts.append((x, y, rr))
        return pts

    def first_free(r, pts, taken):
        for x, y, rr in pts:
            if all(math.hypot(x - q[0], y - q[1]) >= sep[(r, o)] for o, q in taken.items() if o != r):
                return (x, y, rr)
        return None

    cur = {r: tuple(starts[r][:2]) for r in team}
    out = {r: [] for r in team}
    for k in range(1, n + 1):
        cand = {r: candidates(r, th0[r] + sign * math.radians(k * a.step_deg)) for r in team}
        chosen, placed = {}, {}
        for r in team:
            chosen[r] = first_free(r, cand[r], placed)
            placed[r] = chosen[r][:2] if chosen[r] else cur[r]
        # a robot that has to stay may now be too close to one placed before
        # it: place that one again, keeping away from the robots that stay
        for r in team:
            if chosen[r] is None:
                for o in team:
                    if chosen[o] is not None and math.hypot(placed[o][0] - cur[r][0],
                                                            placed[o][1] - cur[r][1]) < sep[(o, r)]:
                        chosen[o] = first_free(o, cand[o], {q: placed[q] for q in team if q != o})
                        placed[o] = chosen[o][:2] if chosen[o] else cur[o]
        for r in team:
            if chosen[r] is None:
                out[r].append([])                    # nowhere to go in this round: stay
                continue
            x, y, rr = chosen[r]
            cur[r] = (x, y)
            face = math.atan2(cy - y, cx - x)
            th = th0[r] + sign * math.radians(k * a.step_deg)
            out[r].append([dict(stop=k, radius_m=round(rr, 3), angle_deg=round(math.degrees(th) % 360.0, 1),
                                x=x, y=y, yaw=wrap_pi(face + math.radians(off)), yaw_offset_deg=off)
                           for off in a.yaw_offsets_deg])
    return out


def circle_center_of(a, arena, boxes):
    """Centre of the circle: the centre of the boxes of the layout,
    recomputed from the ground truth for every layout (--circle-center
    overrides it)."""
    if a.circle_center is not None:
        return (a.circle_center[0], a.circle_center[1])
    return (sum(b["x"] for b in boxes) / len(boxes), sum(b["y"] for b in boxes) / len(boxes))


def fixed_stations(n, center, radius, first_deg):
    """n positions evenly spaced on the circle, the first at first_deg."""
    return [(center[0] + radius * math.cos(math.radians(first_deg + 360.0 * i / n)),
             center[1] + radius * math.sin(math.radians(first_deg + 360.0 * i / n))) for i in range(n)]


def assign_stations(starts, team, stations):
    """{robot: station}: every robot of the team gets the station that
    makes the total (squared) distance from where the robots were placed
    smallest. Deterministic, so every robot process finds the same."""
    from itertools import permutations
    best = None
    for perm in permutations(range(len(stations)), len(team)):
        cost = sum((starts[r][0] - stations[j][0]) ** 2 + (starts[r][1] - stations[j][1]) ** 2
                   for r, j in zip(team, perm))
        if best is None or cost < best[0] - 1e-9:
            best = (cost, perm)
    return {r: stations[j] for r, j in zip(team, best[1])}


def station_problem(xy, boxes, arena, a):
    """Why a fixed position cannot be used ('' if it can)."""
    if not inside(xy, arena, a.margin):
        return f"closer than --margin {a.margin:g} m to the arena border"
    if not segment_clear(xy, xy, boxes, a.keep_out_m):
        return f"closer than --keep-out-m {a.keep_out_m:g} m to a box"
    return ""


def team_plan(starts, team, boxes, box_center, arena, a, n_stations=None):
    """The round plan of the whole team, the same in every robot process
    and in the launcher. Returns (face_center, round0, arcs):
      face_center: the point every robot faces
      round0: {robot: (x, y)} position of round 0
      arcs: {robot: [[captures of round 1], [round 2], ...]} ([] = stay)
    Without --radius: round 0 where each robot was placed, arcs around the
    boxes. With --radius: fixed positions on the circle around the centre
    of the boxes (n_stations evenly spaced, default one per robot of the
    team), at the same distance and angles for every layout;
    the robots rotate on the circle and always face its centre."""
    if a.radius is None:
        if a.sweep_deg is None:
            a.sweep_deg = 180.0
        r0 = {r: tuple(starts[r][:2]) for r in team}
        return box_center, r0, team_arcs(r0, team, box_center, boxes, arena, a)
    n = n_stations or len(team)
    if a.sweep_deg is None:
        a.sweep_deg = 360.0 / n
    c = circle_center_of(a, arena, boxes)
    r0 = assign_stations(starts, team, fixed_stations(n, c, a.radius, a.first_angle_deg))
    return c, r0, team_arcs(r0, team, c, boxes, arena, a, radial_search=False)


def min_radius(a, arena, boxes):
    """Smallest --radius whose whole circle stays --keep-out-m away from
    every box of this layout."""
    c = circle_center_of(a, arena, boxes)
    return max(math.hypot(b["x"] - c[0], b["y"] - c[1]) for b in boxes) + a.keep_out_m


def max_radius(a, arena, boxes):
    """Largest --radius whose whole circle stays --margin inside the arena."""
    c = circle_center_of(a, arena, boxes)
    return min(c[0] - arena[0], arena[1] - c[0], c[1] - arena[2], arena[3] - c[1]) - a.margin


def sim_team_starts(center, arena, team, a):
    """--sim: where the fake robots are 'placed': close to each other, on
    one side of the boxes (1.6 m from the centre, 40 degrees apart), to
    show that the rotating formation keeps them apart."""
    out = {}
    for i, r in enumerate(team):
        th = math.radians(200.0 + 40.0 * i)
        x, y = center[0] + 1.6 * math.cos(th), center[1] + 1.6 * math.sin(th)
        out[r] = (x, y, math.atan2(center[1] - y, center[0] - x))
    return out


def start_captures(pose, center, yaw_offsets_deg, index):
    """Round 0: the captures at the starting position, where the robot was
    placed (it only turns on the spot to face the centre, plus the yaw
    offsets). Stop number 1000 + position of the robot in the team."""
    x, y = pose[0], pose[1]
    face = math.atan2(center[1] - y, center[0] - x)
    r = math.hypot(x - center[0], y - center[1])
    ang = round(math.degrees(math.atan2(y - center[1], x - center[0])) % 360.0, 1)
    return [dict(stop=1000 + index, radius_m=round(r, 3), angle_deg=ang, x=x, y=y,
                 yaw=wrap_pi(face + math.radians(off)), yaw_offset_deg=off) for off in yaw_offsets_deg]


def group_by_stop(captures):
    rounds = []
    for c in captures:
        if not rounds or rounds[-1][0]["stop"] != c["stop"]:
            rounds.append([])
        rounds[-1].append(c)
    return rounds


def sector_rounds(plan, n_sectors, sector, center):
    """The stops of ONE robot in round mode: its sector (split_groups) as a
    list of rounds, each round = the captures of one stop. Also returns the
    number of rounds of the whole team (the longest sector)."""
    groups = split_groups(plan, n_sectors, center)
    rounds = []
    for c in groups[sector]:
        if not rounds or rounds[-1][0]["stop"] != c["stop"]:
            rounds.append([])
        rounds[-1].append(c)
    n_rounds = max(len({c["stop"] for c in g}) for g in groups)
    return rounds, n_rounds, groups


class RoundSync:
    """File-based barrier between the survey.py processes of one team
    (launch_all.py): after round k every robot writes <dir>/r<k>_<robot>,
    then waits until every robot of the team has written it. A robot that
    gets too close to the arena border writes <dir>/abort: all stop."""

    def __init__(self, folder, robot_id, team, timeout_s):
        self.folder, self.robot_id, self.team, self.timeout_s = folder, robot_id, team, timeout_s
        os.makedirs(folder, exist_ok=True)

    def _mark(self, name):
        with open(os.path.join(self.folder, name), "w") as f:
            f.write(str(time.time()))

    def publish_start(self, pose):
        with open(os.path.join(self.folder, f"start_{self.robot_id}.json"), "w") as f:
            json.dump(dict(x=pose[0], y=pose[1], yaw=pose[2]), f)

    def read_starts(self):
        starts = {}
        for r in self.team:
            fn = os.path.join(self.folder, f"start_{r}.json")
            if os.path.exists(fn):
                with open(fn) as f:
                    q = json.load(f)
                starts[r] = (q["x"], q["y"])
        return starts

    def aborted(self):
        return STOP.is_set() or os.path.exists(os.path.join(self.folder, "abort"))

    def wait_go(self):
        """Waits for the file "go" (launch_all.py writes it when the path
        preview is confirmed). False if the survey is stopped meanwhile."""
        phase("waiting for the start confirmation", idle=True)
        while not os.path.exists(os.path.join(self.folder, "go")):
            if self.aborted():
                return False
            time.sleep(0.2)
        return True

    def abort(self):
        self._mark("abort")

    def done_with(self, k, log=print):
        """Round k finished for this robot: wait for the others. False if the
        team must stop."""
        phase(f"round {k} done, waiting for the team", idle=True)
        self._mark(f"r{k}_{self.robot_id}")
        t0 = time.time()
        said = 0.0
        while True:
            if self.aborted() or STOP.is_set():
                return False
            missing = [r for r in self.team if not os.path.exists(os.path.join(self.folder, f"r{k}_{r}"))]
            if not missing:
                return True
            if time.time() - said > 10.0:
                if said:
                    what = []
                    for r in missing:
                        ph = read_phase(self.folder, r)
                        what.append(r if ph is None else f"{r}: {ph[0]} since {ph[1]:.0f} s")
                    log(f"round {k} done, standing still ({time.time() - t0:.0f} s): waiting for "
                        f"{'; '.join(what)}")
                said = time.time()
            if time.time() - t0 > self.timeout_s:
                log(f"round {k}: still waiting for {missing} after {self.timeout_s:.0f} s, going on")
                return True
            time.sleep(0.2)


def draw_sectors(path, arena, margin, boxes, groups, names, paths, circle=None, placed=None):
    """Top view of the round plan: every robot's sector, its stops in order
    and its starting stop (square). circle = (center, radius): the fixed
    circle, drawn dashed; placed = {name: (x, y)}: where the robots are now
    (cross), when that is not their round-0 position."""
    # matplotlib Figure without pyplot: the backend of the program that
    # calls this (e.g. the live window of launch_all.py) is not touched
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    x0, x1, y0, y1 = arena
    fig = Figure(figsize=(7, 7 * (y1 - y0) / (x1 - x0)))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="k"))
    ax.add_patch(Rectangle((x0 + margin, y0 + margin), x1 - x0 - 2 * margin, y1 - y0 - 2 * margin,
                               fill=False, color="0.6", ls="--"))
    for b in boxes:
        ax.add_patch(Rectangle((b["x"] - 0.1, b["y"] - 0.1), 0.2, 0.2, color="0.3"))
    if circle is not None:
        from matplotlib.patches import Circle
        ax.add_patch(Circle(circle[0], circle[1], fill=False, color="0.5", ls=":", lw=1))
        ax.plot(circle[0][0], circle[0][1], "+", color="0.5")
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (g, name) in enumerate(zip(groups, names)):
        col = colors[k % len(colors)]
        for pts in paths[k]:
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, lw=0.8)
        order = []
        for c in g:
            if not order or order[-1]["stop"] != c["stop"]:
                order.append(c)
        for i, c in enumerate(order):
            ax.plot(c["x"], c["y"], "o", color=col)
            ax.text(c["x"] + 0.05, c["y"] + 0.05, f"r{c['stop']}", fontsize=7, color=col)
        if order:
            ax.plot(order[0]["x"], order[0]["y"], "s", ms=12, mfc="none", mec=col, mew=2,
                    label=f"{name} (start)")
        if placed and name in placed:
            ax.plot(placed[name][0], placed[name][1], "x", ms=9, mew=2, color=col)
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.1, x1 + 0.1)
    ax.set_ylim(y0 - 0.1, y1 + 0.1)
    ax.set_xlabel("x [m] (OptiTrack planar frame)")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def run_rounds(a, rh, center, boxes, arena, sync, path):
    """Round mode (launch_all.py). Every launch is a new run: every round is
    driven and captured, nothing is skipped because an earlier run was there.
    Round 0: with --radius every robot drives to its fixed position on the
    circle; without it, it captures where it was placed (turning on the
    spot only). Then, in every round, ALL the robots move together by
    --step-deg (see team_plan) and capture; the next round starts when every
    robot is done."""
    n_ok = n_fail = 0
    done = set()
    here = rh.body_pose()
    if here is None:
        print("no OptiTrack pose at the start: this robot cannot take part", flush=True)
        return n_ok, n_fail
    sync.publish_start(here)
    if not sync.done_with("start"):          # every robot connected and its start known
        print("survey stopped before the start", flush=True)
        return n_ok, n_fail
    starts = sync.read_starts()
    if a.robot_id not in starts:
        starts[a.robot_id] = here[:2]
    team = [r for r in sync.team if r in starts]
    fixed = a.radius is not None
    face, r0, arcs = team_plan(starts, team, boxes, center, arena, a, n_stations=len(sync.team) or None)
    start_xy = r0[a.robot_id]
    if fixed:
        why = station_problem(start_xy, boxes, arena, a)
        if why:
            # the launcher shows the whole plan with this error and stops the team
            print(f"fixed position ({start_xy[0]:+.2f}, {start_xy[1]:+.2f}) is {why}", flush=True)
            if not a.wait_go:
                sync.abort()
                return n_ok, n_fail
    others_start = [math.hypot(start_xy[0] - q[0], start_xy[1] - q[1]) for o, q in r0.items()
                    if o != a.robot_id]
    if others_start and 0.8 * min(others_start) < a.stop_distance:
        # robots close to each other: the "someone in my way" stop must
        # not trigger at the spacing of the formation
        a.stop_distance = max(0.3, 0.8 * min(others_start))
        print(f"{min(others_start):.2f} m from another robot: stop distance {a.stop_distance:.2f} m",
              flush=True)
    rounds = [start_captures(start_xy, face, a.yaw_offsets_deg, 0)] + arcs[a.robot_id]
    for c in rounds[0]:
        c["stop"] = 0
    n_rounds = len(rounds)                    # the same for every robot (same sweep and step)
    holds = sum(1 for r in rounds[1:] if not r)
    where = (f"fixed position ({start_xy[0]:+.2f}, {start_xy[1]:+.2f}) on the circle r={a.radius:g} m"
             if fixed else f"start ({start_xy[0]:+.2f}, {start_xy[1]:+.2f})")
    print(f"{where}; {n_rounds - 1} round(s) ({a.sweep_deg:g} deg {a.direction}, every {a.step_deg:g} deg)"
          + (f", {holds} of them standing still (no room there)" if holds else ""), flush=True)
    if a.wait_go:
        print("path planned: waiting for the start confirmation in the launcher", flush=True)
        if not sync.wait_go():
            return n_ok, n_fail
    print("all robots ready: round 0 " + ("at the fixed positions" if fixed else "where they were placed"),
          flush=True)
    for k in range(n_rounds):
        if sync.aborted():
            print("another robot stopped the survey")
            break
        if rounds[k]:
            robots_now = rh.others() if (rh.others is not None and (k > 0 or fixed)) else ()
            status = visit_stop(a, rh, rounds[k], done, path, center, boxes, arena, robots=robots_now,
                                extra=dict(round=k, run_id=a.run_id), in_place=(k == 0 and not fixed))
            if status == "stopped":
                print("stop requested", flush=True)
                break
            if status == "fence":
                sync.abort()
                break
            if status == "ok":
                n_ok += 1
            elif status in ("blocked", "failed"):
                n_fail += 1
        if not sync.done_with(k):
            print("survey stopped by another robot")
            break
        print(f"round {k} done by the whole team", flush=True)
    return n_ok, n_fail


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-id", required=True)
    p.add_argument("--sn", default=None)
    p.add_argument("--ip", default=None)
    p.add_argument("--start-deg", type=float, default=0.0, help="angle of the first stop")
    p.add_argument("--others", type=int, nargs="*", default=[],
                   help="OptiTrack rigid body ids of the OTHER robots, for collision avoidance")
    p.add_argument("--optitrack-rb-id", type=int, default=None)
    p.add_argument("--yaw-offset-deg", type=float, default=0.0,
                   help="added to the OptiTrack yaw of this robot so that it points where the robot's front "
                        "points; measure it with tools/check_motion.py")
    p.add_argument("--camera-calib", default=None)
    p.add_argument("--team", nargs="*", default=[], help="round mode: ids of all the robots of the team")
    p.add_argument("--sync-dir", default=None,
                   help="round mode (launch_all.py sets it): folder used to wait for the team")
    p.add_argument("--ignore-ctrl-c", action="store_true",
                   help="(launch_all.py) Ctrl+C is handled by the launcher, which asks every robot to stop")
    p.add_argument("--wait-go", action="store_true",
                   help="round mode (launch_all.py): wait for the path preview to be confirmed before moving")
    p.add_argument("--sync-timeout", type=float, default=180.0,
                   help="round mode: go on after waiting this long for the others [s]")
    p.add_argument("--stuck-s", type=float, default=60.0,
                   help="round mode: a step (driving one leg, capturing, saving) lasting longer than this "
                        "prints where the process is stuck (0 = never)")
    add_survey_arguments(p)
    a = p.parse_args()

    if a.ignore_ctrl_c:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if a.sync_dir:
        global WATCH
        watch_abort_file(os.path.join(a.sync_dir, "abort"))
        WATCH = PhaseWatch(a.robot_id, a.sync_dir, a.stuck_s)
        boxes, center, arena, plan = load_plan(a, 0.0, need_plan=False)
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"{a.robot_id}.jsonl")
        team = a.team or [a.robot_id]
        sim_pose = sim_team_starts(center, arena, team, a)[a.robot_id] if a.sim else None
        sync = RoundSync(a.sync_dir, a.robot_id, team, a.sync_timeout)
        rh = None
        n_ok = n_fail = 0
        try:
            rh = RobotHandle(a, boxes, start_pose=sim_pose)
            n_ok, n_fail = run_rounds(a, rh, center, boxes, arena, sync, path)
        except KeyboardInterrupt:
            print("interrupted", flush=True)
        finally:
            if rh is not None:
                rh.close()
        print(f"done: {n_ok} stop(s) visited, {n_fail} skipped, data in {path}")
        return

    boxes, center, arena, plan = load_plan(a, a.start_deg)
    if a.plan_only:
        prev, paths, kept = None, [], []
        for c in plan:
            legs = ""
            if prev is not None and (c["x"], c["y"]) != prev:
                wps = route(prev, (c["x"], c["y"]), center, boxes, a.keep_out_m, arena, a.margin)
                legs = ("  <-- NO SAFE PATH from the previous stop: skipped" if wps is None
                        else "" if len(wps) == 1 else f"  (round the boxes, {len(wps)} waypoints)")
                if wps is None:
                    continue
                paths.append([prev] + wps)
            print(f"  stop {c['stop']:3d}  r={c['radius_m']:.2f}  angle={c['angle_deg']:6.1f}  "
                  f"({c['x']:+.2f}, {c['y']:+.2f})  yaw={math.degrees(c['yaw']):+6.1f}  "
                  f"offset={c['yaw_offset_deg']:+.0f}{legs}")
            prev = (c["x"], c["y"])
            kept.append(c)
        if a.plan_png:
            draw_plan(a.plan_png, arena, a.margin, boxes, center, kept, paths)
            print(f"plan drawn in {a.plan_png}")
        return

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.robot_id}.jsonl")
    done = done_trials(path)
    rh = RobotHandle(a, boxes, start_pose=sim_start(plan[0], center) if a.sim else None)
    by_stop = {}
    for c in plan:
        by_stop.setdefault(c["stop"], []).append(c)
    n_before = len(done)
    n_fail = 0
    try:
        for k in sorted(by_stop, key=lambda k: plan.index(by_stop[k][0])):
            status = visit_stop(a, rh, by_stop[k], done, path, center, boxes, arena)
            if status == "fence":
                break
            if status in ("blocked", "failed"):
                n_fail += 1
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        rh.close()
    print(f"done: {len(done) - n_before} capture(s) written to {path}, {n_fail} stop(s) skipped")


if __name__ == "__main__":
    main()