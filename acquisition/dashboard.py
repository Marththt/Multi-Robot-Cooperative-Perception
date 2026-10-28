"""Live dashboard: continuously reads the state that the robot processes
(static_trial.py --live --state-out ...) write, and shows on screen --
updated in real time, not a single capture -- what EACH robot sees, how
physical instances are matched across different robots, and what each
fusion baseline decides.

No robot connection here: only reading of files written by other processes
(same "one process per robot" scheme as always -- here in addition a third
process, without a robot, that only mirrors what the other two write).
Reuses the SAME cross-robot clustering logic (split_instances) and the SAME
aggregation baselines as merge_static_trials.py -- not a copy, a direct
import: so the live merge and the one on saved trials are guaranteed
identical, no duplicate code to keep in sync by hand.

Usage (three terminals)
-----------------------
Terminal 1 (camera P0 + live state):
    python -m acquisition.static_trial --robot-id P0 --live --state-out live/P0.json \\
        --regions config/optitrack_regions.json --localization optitrack \\
        --optitrack-rb-id 8 --detector aruco --aruco-marker-size 0.08 \\
        --camera-calib config/checkerboard_calib_P0.json

Terminal 2 (camera P1 + live state):
    python -m acquisition.static_trial --robot-id P1 --live --state-out live/P1.json \\
        --regions config/optitrack_regions.json --localization optitrack \\
        --optitrack-rb-id 7 --detector aruco --aruco-marker-size 0.08

Terminal 3 (dashboard, NO robot -- can also run on another machine that
only sees the live/ folder):
    python -m acquisition.dashboard --state-dir live/

Multi-object localization (--localize): additionally shows, for every
class, how many objects and WHERE according to the methods of Section 3.2
of the document (zone_class, union, mht, phd, cphd --
vf/fusion/localization_methods.py), computed on the current live frame.
Sensor model WITHOUT faults by default (p_D=1 inside the field of view, no
false alarms), see --p-d/--clutter:
    python -m acquisition.dashboard --state-dir live/ --regions config/optitrack_regions.json --localize

Terminal 3 refreshes by itself (default ~3 times per second). 'q' + enter,
or Ctrl+C, to quit. The two video windows (terminals 1 and 2, the usual
--live) stay separate: the camera is seen there, the merge is seen here --
three windows/terminals for two robots, on purpose (see the "one process
per robot" constraint already documented in patrol.py/static_trial.py).

"Old" data
----------
If a robot stops updating its file (window closed, process stuck, robot
disconnected), its last state gets progressively older -- beyond
--max-age seconds (default 2.0) the dashboard explicitly marks it as STALE
and treats it as "nothing seen" instead of continuing to show, and fuse,
data that is no longer true: old data passed off as current would be worse
than no data.
"""

import argparse
import glob
import json
import math
import os
import threading
import time

from acquisition.merge_static_trials import (
    split_instances, BASELINE)
from vf.fusion.multi_robot_simulation import Observation
from vf.fusion.localization_methods import METHODS, localize_live
from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_regions,
                                    sensor_from_arguments, zones_from_arguments)


def methods_from_argument(value):
    """--localize all | mht,cphd,... -> list of valid names."""
    if value in (None, "", "all"):
        return list(METHODS)
    methods = [m.strip() for m in value.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"--localize: unknown methods {unknown}, "
                         f"choose among {list(METHODS)}")
    return methods


def robots_for_localization(per_robot_rows, pose_per_robot, stale_per_robot):
    """From the live frame to the localize_live format: only live robots
    (stale data is no longer true) -- those without a pose are then
    excluded by localize_live itself."""
    robots = {}
    for rid, rows in per_robot_rows.items():
        if stale_per_robot.get(rid):
            continue
        robots[rid] = dict(
            pose=pose_per_robot.get(rid),
            observations=[dict(class_name=r["class_name"], distance=r.get("distance"),
                               bearing=r.get("bearing_rad"), position=r.get("position"))
                          for r in rows if r.get("class_name") is not None])
    return robots


def format_localization(used, results, excluded):
    lines = ["LOCALIZATION (per class; robots used: "
             + (", ".join(used) or "none") + ")"]
    if excluded:
        lines.append(f"  excluded (no pose or stale): {', '.join(excluded)}")
    if not results:
        lines.append("  (no object with a position)")
    for class_name, per_method in results.items():
        lines.append(f"  {class_name}:")
        for name, out in per_method.items():
            pos = " ".join(f"({x:+.2f},{y:+.2f})" for x, y in out["objects"]) or "-"
            lines.append(f"    {name:<15s} n={out['n']}  {pos}")
    return lines


def read_state(path):
    """State of ONE robot from its live file, or None if the file does not
    exist yet (process not started yet), is corrupted (read half-way while
    the other process was writing it -- rare thanks to the atomic write in
    static_trial.py write_live_state, but not impossible), or is
    momentarily inaccessible (on Windows: the read may happen at the exact
    instant the other process is doing the atomic rename, or an antivirus
    touches the file -- the same kind of transient hiccup that
    write_live_state already retries against on the writing side). In
    EVERY case this round is skipped and retried at the next refresh
    (--hz), instead of bringing down the whole dashboard for a single file
    out of dozens of reads per second."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def rows_from_state(state, max_age):
    """state of ONE robot (dict read by read_state) -> (rows, age, stale).
    rows is a list of merge_static_trials-style dicts (one per detection;
    a single one with class_name=None if the robot sees nothing or the data
    is stale)."""
    robot_id = state["robot_id"]
    age = time.time() - state["t"]
    empty = [dict(robot_id=robot_id, class_name=None, confidence=0.0,
                  position=None, distance=None,
                  fault_injected=False, dropped=False)]

    if age > max_age:
        return empty, age, True

    detections = state.get("detections") or []
    non_empty = [r for r in detections if r.get("class_name") is not None]
    if not non_empty:
        return empty, age, False

    rows = [dict(robot_id=robot_id, class_name=r["class_name"],
                 confidence=r.get("confidence", 1.0),
                 position=r.get("position"),
                 distance=r.get("distance"),
                 bearing_rad=r.get("bearing_rad"),
                 fault_injected=False, dropped=False)
            for r in non_empty]
    return rows, age, False


def _format_xy(position):
    if position is None:
        return "xy=(n/a)"
    return f"xy=({position[0]:+.2f}, {position[1]:+.2f})"


def _format_robot_pose(pose):
    """The RAW pose (x, y, yaw) that the robot itself reports (OptiTrack or
    odometry, with the camera offset already applied if --camera-offset-*
    was used) -- not the position of an object. Shown in the ROBOTS
    section to tell apart an error in the robot's POSE (x/y/yaw wrong at
    the source -- it would shift IN THE SAME WAY any object that robot
    sees, regardless of its bearing) from an error in the object estimate
    (which would instead depend on how off-centre the object is, see
    _format_d_b): without seeing this number, the two cases are
    indistinguishable by looking only at the objects' xy."""
    if pose is None:
        return "pose=(n/a)"
    x, y, yaw_rad = pose
    return f"pose=({x:+.2f}, {y:+.2f}) yaw={math.degrees(yaw_rad):+.0f}deg"


def _format_d_b(distance, bearing_rad):
    """d=measured distance, b=angle relative to the camera axis of THAT
    robot (not an absolute angle in the arena) -- shown together with xy
    precisely to tell whether two robots disagree on an instance because of
    a DISTANCE error (different d, similar b) or an ANGLE/yaw error
    (similar d, b that alone would not explain the xy gap) instead of
    seeing only the final result already fused into (x,y)."""
    if distance is None:
        return "d=n/a"
    text = f"d={distance:.2f}m"
    if bearing_rad is not None:
        text += f" b={math.degrees(bearing_rad):+.0f}deg"
    return text


def draw_screen(per_robot_rows, age_per_robot, stale_per_robot,
                instance_threshold_m, baselines_to_use, pose_per_robot=None,
                p_d_per_robot=None,
                localization=None):
    """Layout in separate blocks (ROBOTS / one section per instance), no
    longer a single compressed line with id+class+confidence+baseline all
    together -- that was unreadable with more than one robot or instance.
    Each instance section lists, one robot per line, the measured
    DISTANCE/ANGLE AND the absolute (x, y) COORDINATES that THAT robot
    assigned to the object -- previously they did not appear in the
    dashboard at all, only class/confidence. Showing d/b separately from xy
    helps diagnose a disagreement between robots: if d is similar but xy
    is not, the problem is probably an orientation/pose (yaw) offset, not
    the distance estimate.

    pose_per_robot (optional, robot_id -> (x,y,yaw_rad) or None) adds the
    robot's own RAW pose in the ROBOTS section -- it helps tell apart an
    error in the robot's pose (shifts every object seen by that robot in
    the same way, whatever its bearing) from an error that depends on the
    object/bearing (see _format_robot_pose). None or omitted -> unchanged
    behaviour (no pose line, backward compatible with callers that do not
    pass it)."""
    width = 70
    parts = ["=" * width,
             "LIVE DASHBOARD -- multi-robot static trial".center(width),
             time.strftime("%H:%M:%S").center(width),
             "=" * width, "", "ROBOTS"]

    pose_per_robot = pose_per_robot or {}
    for robot_id in sorted(age_per_robot):
        if stale_per_robot[robot_id]:
            state_str = f"STALE (last data {age_per_robot[robot_id]:.1f}s ago)"
        else:
            state_str = f"live (updated {age_per_robot[robot_id]:.1f}s ago)"
        n_seen = sum(1 for r in per_robot_rows.get(robot_id, [])
                     if r["class_name"] is not None)
        line = f"  {robot_id}: {state_str} -- {n_seen} object(s) seen"
        if robot_id in pose_per_robot:
            line += f"  {_format_robot_pose(pose_per_robot[robot_id])}"
        robot_p_d = (p_d_per_robot or {}).get(robot_id)
        if robot_p_d is not None and robot_p_d < 1.0:
            line += f"  [interference p_D={robot_p_d:.2f}]"
        parts.append(line)
    parts.append("")

    robot_rows = [(rid, r) for rid, rows in per_robot_rows.items()
                  for r in rows]
    if not robot_rows:
        parts.append("(no data yet)")
        return "\n".join(parts)

    if localization is not None:
        parts.extend(format_localization(*localization))
        parts.append("")

    groups = split_instances(robot_rows, threshold_m=instance_threshold_m)
    multi = len(groups) > 1
    for i, rows in enumerate(groups, start=1):
        observations = {
            robot_id: Observation(
                robot_id=robot_id, class_name=r["class_name"],
                confidence=r["confidence"],
                fault_injected=r.get("fault_injected", False),
                dropped=r.get("dropped", False))
            for robot_id, r in rows.items()
        }
        label = f"INSTANCE #{i}" if multi else "FUSION (single object in scene)"
        parts.append(label)
        for robot_id in sorted(rows):
            r = rows[robot_id]
            if r["class_name"] is None:
                parts.append(f"  {robot_id}: nothing seen")
            else:
                parts.append(f"  {robot_id}: {r['class_name']:<10s} "
                             f"conf={r['confidence']:.2f}  "
                             f"{_format_d_b(r.get('distance'), r.get('bearing_rad'))}  "
                             f"{_format_xy(r.get('position'))}")
        parts.append("  " + "-" * (width - 2))
        for name, fn in baselines_to_use:
            decision = fn(observations)
            parts.append(f"  {name:<20s}-> {decision}")
        parts.append("")

    return "\n".join(parts).rstrip("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dir", required=True,
                   help="folder with <robot-id>.json written by "
                        "static_trial.py --live --state-out")
    p.add_argument("--baseline", choices=list(BASELINE.keys()) + ["all"],
                   default="all")
    p.add_argument("--instance-threshold-m", type=float, default=0.5,
                   help="same meaning as in merge_static_trials.py")
    p.add_argument("--max-age", type=float, default=2.0,
                   help="seconds after which a robot's data is considered "
                        "stale instead of live")
    p.add_argument("--hz", type=float, default=3.0,
                   help="how many times per second to refresh the screen")
    p.add_argument("--localize", nargs="?", const="all", default=None,
                   help="show the multi-object localization: 'all' "
                        "(default if you write just --localize) or a list "
                        "like mht,cphd among: " + ",".join(METHODS))
    p.add_argument("--regions", default=None,
                   help="regions JSON: arena bounds, required with --localize (the "
                        "fusion zones are a grid of --zone-side cells over the arena)")
    add_sensor_arguments(p)
    args = p.parse_args()

    loc_methods, loc_sensor, loc_zones = None, None, None
    if args.localize is not None:
        if not args.regions:
            p.error("--localize requires --regions (arena bounds and zones)")
        loc_methods = methods_from_argument(args.localize)
        arena = arena_from_regions(args.regions)
        loc_sensor = sensor_from_arguments(args, arena)
        loc_zones = zones_from_arguments(args, arena)

    baselines_to_use = (list(BASELINE.items()) if args.baseline == "all"
                        else [(args.baseline, BASELINE[args.baseline])])

    stop = threading.Event()

    def wait_for_quit():
        try:
            while True:
                line = input()
                if line.strip().lower() == "q":
                    break
        except EOFError:
            pass
        stop.set()

    threading.Thread(target=wait_for_quit, daemon=True).start()

    print(f"Waiting for state files in {args.state_dir}/ ... "
          f"'q' + enter or Ctrl+C to quit.")

    try:
        while not stop.is_set():
            paths = sorted(glob.glob(os.path.join(args.state_dir, "*.json")))
            per_robot_rows, age_per_robot, stale_per_robot = {}, {}, {}
            pose_per_robot = {}
            p_d_per_robot = {}
            for path in paths:
                state = read_state(path)
                if state is None:
                    continue
                rows, age, stale = rows_from_state(state, args.max_age)
                per_robot_rows[state["robot_id"]] = rows
                age_per_robot[state["robot_id"]] = age
                stale_per_robot[state["robot_id"]] = stale
                # RAW robot pose (x,y,yaw), already written by
                # write_live_state -- None if the robot had no valid pose
                # in that frame (OptiTrack tracking lost) or if the data is
                # stale.
                pose_per_robot[state["robot_id"]] = (
                    state.get("pose") if not stale else None)
                # p_D injected by that robot (static_trial.py --p-d),
                # absent in files written by earlier versions
                if state.get("p_d") is not None:
                    p_d_per_robot[state["robot_id"]] = state["p_d"]

            localization = None
            if loc_methods is not None:
                loc_robots = robots_for_localization(
                    per_robot_rows, pose_per_robot, stale_per_robot)
                used, results = localize_live(loc_robots, loc_sensor,
                                              loc_methods, loc_zones)
                excluded = sorted(set(per_robot_rows) - set(used))
                localization = (used, results, excluded)

            os.system("cls" if os.name == "nt" else "clear")
            if not per_robot_rows:
                print(f"no state file yet in {args.state_dir}/ "
                      f"-- check that static_trial.py --live "
                      f"--state-out is running in the other "
                      f"terminals")
            else:
                print(draw_screen(per_robot_rows, age_per_robot,
                                  stale_per_robot,
                                  args.instance_threshold_m,
                                  baselines_to_use,
                                  pose_per_robot=pose_per_robot,
                                  p_d_per_robot=p_d_per_robot,
                                  localization=localization))
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()