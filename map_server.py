"""Local server for the visual map (map.html): reads THE SAME live state
files that dashboard.py reads (written by static_trial.py --live
--state-out ...) and re-exposes them as JSON over HTTP, so an HTML page in
the browser can draw a top-down map (robots + detected objects) instead of
text only.

DIRECTLY reuses the functions of dashboard.py and merge_static_trials.py
(read_state, rows_from_state, split_instances, BASELINE)
-- same principle already followed there: a single place where the
reading/fusion logic lives, never a rewritten copy here
that could drift from the one used by the text dashboard.

It does NOT replace dashboard.py (which stays useful on a machine without
a browser, e.g. over SSH): it is a visual alternative/addition that reads
the SAME --state-dir folder, in a separate process.

Usage (in one more terminal besides the static_trial.py ones, e.g.
    python static_trial.py --robot-id P0 --live --state-out live/P0.json \\
        --regions config/optitrack_regions.json --localization optitrack \\
        --optitrack-rb-id 8 --detector aruco --aruco-marker-size 0.08 \\
        --camera-calib config/checkerboard_calib_P0.json
for each robot):
    python map_server.py --state-dir live/ --regions config/optitrack_regions.json

With multi-object localization (all the methods of Section 3.2 of the
document, without faults by default -- see dashboard.py --localize):
    python map_server.py --state-dir live/ --regions config/optitrack_regions.json --localize

then open in the browser:
    http://localhost:8765/map.html

'q' + enter, or Ctrl+C, to stop the server. map_server.py and map.html must
be kept in the SAME folder (the project one, together with
dashboard.py/static_trial.py) -- the server serves map.html from there.
"""

import argparse
import glob
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard import (read_state, rows_from_state, methods_from_argument,
                       robots_for_localization)
from vf.fusion.localization_methods import METHODS, localize_live
from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_regions,
                                    sensor_from_arguments, zones_from_arguments)
from merge_static_trials import (
    split_instances, BASELINE)
from vf.fusion.multi_robot_simulation import Observation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOC_LOCK = threading.Lock()
_LOC_CACHE = {"t": 0.0, "value": None}
ARGS = None  # set in main(), read by Handler.do_GET -- a module global
             # instead of subclassing Handler for every argument: small
             # script, a single process, no need for more ceremony than this


def _read_regions(path):
    """--regions (the same JSON file as static_trial.py/dashboard) ->
    bounding box {x_min,x_max,y_min,y_max} ONLY to know how large the
    arena to draw is -- it does not touch the actual zone logic, which
    stays where it already is (localization.py)."""
    if not path:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    regions = data.get("regions") or []
    if not regions:
        return None
    return dict(
        x_min=min(r["x_min"] for r in regions),
        x_max=max(r["x_max"] for r in regions),
        y_min=min(r["y_min"] for r in regions),
        y_max=max(r["y_max"] for r in regions))


def _current_state():
    """A complete frame for the map: raw state per robot (to draw the
    triangles), clustered instances with the decisions of every baseline
    (for the "fusion" panel) and the localization if --localize is
    active -- the same information content as draw_screen()
    in dashboard.py, but as data instead of formatted text."""
    paths = sorted(glob.glob(os.path.join(ARGS.state_dir, "*.json")))
    robots = {}
    robot_rows = []
    p_d_state = {}
    for path in paths:
        state = read_state(path)
        if state is None:
            continue
        rows, age, stale = rows_from_state(state, ARGS.max_age)
        rid = state["robot_id"]
        if state.get("p_d") is not None:
            p_d_state[rid] = state["p_d"]
        robots[rid] = dict(
            p_d=state.get("p_d"),
            pose=(state.get("pose") if not stale else None),
            age=round(age, 2), stale=stale,
            detections=[
                dict(class_name=r["class_name"], confidence=r["confidence"],
                     distance=r.get("distance"),
                     bearing_rad=r.get("bearing_rad"),
                     position=r.get("position"))
                for r in rows if r["class_name"] is not None])
        robot_rows.extend((rid, r) for r in rows)

    instances = []
    if robot_rows:
        groups = split_instances(robot_rows, threshold_m=ARGS.instance_threshold_m)
        multi = len(groups) > 1
        for i, rows in enumerate(groups, start=1):
            observations = {
                rid: Observation(
                    robot_id=rid, class_name=r["class_name"],
                    confidence=r["confidence"],
                    fault_injected=r.get("fault_injected", False),
                    dropped=r.get("dropped", False))
                for rid, r in rows.items()}
            per_robot = {
                rid: dict(class_name=r["class_name"], confidence=r["confidence"],
                          distance=r.get("distance"),
                          bearing_rad=r.get("bearing_rad"),
                          position=r.get("position"))
                for rid, r in rows.items()}
            baselines = {name: fn(observations) for name, fn in BASELINE.items()}
            instances.append(dict(
                label=(f"INSTANCE #{i}" if multi else "FUSION (single object in scene)"),
                per_robot=per_robot, baselines=baselines))

    return dict(t=time.time(), robots=robots, instances=instances,
                regions=ARGS.regions_bounds,
                localization=_localization(robots))


def _localization(robots):
    """Same localization as dashboard.py --localize. Computed at most every
    --localize-interval seconds and reused between polls: the browser asks
    for the state every ~0.4 s, and several open pages must not multiply
    the computation."""
    if ARGS.loc_methods is None:
        return None
    with _LOC_LOCK:
        now = time.time()
        if (_LOC_CACHE["value"] is not None
                and now - _LOC_CACHE["t"] < ARGS.localize_interval):
            return _LOC_CACHE["value"]
        per_robot_rows = {rid: [dict(class_name=d["class_name"], distance=d.get("distance"),
                                     bearing_rad=d.get("bearing_rad"), position=d["position"])
                                for d in r["detections"]] for rid, r in robots.items()}
        pose = {rid: r["pose"] for rid, r in robots.items()}
        stale = {rid: r["stale"] for rid, r in robots.items()}
        used, results = localize_live(
            robots_for_localization(per_robot_rows, pose, stale),
            ARGS.loc_sensor, ARGS.loc_methods, ARGS.loc_zones)
        value = dict(
            methods=ARGS.loc_methods, used=used,
            excluded=sorted(set(robots) - set(used)),
            sensor=dict(hfov_deg=ARGS.hfov_deg, r_max=ARGS.r_max,
                        p_d=ARGS.p_d, sigma_pos=ARGS.sigma_pos),
            per_class={c: {m: dict(n=o["n"], objects=[list(x) for x in o["objects"]])
                           for m, o in pm.items()}
                       for c, pm in results.items()})
        _LOC_CACHE.update(t=now, value=value)
        return value


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            try:
                body = json.dumps(_current_state()).encode("utf-8")
                code = 200
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode("utf-8")
                code = 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/map.html")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, fmt, *args_):
        pass  # otherwise one log line for EVERY single browser poll
              # (every poll_ms) -- useless noise in the terminal


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dir", required=True,
                   help="same folder as dashboard.py --state-dir")
    p.add_argument("--regions", default=None,
                   help="regions JSON (e.g. config/optitrack_regions.json): arena "
                        "bounds, to draw the map and for --localize (the fusion zones "
                        "are a grid of --zone-side cells over the arena)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-age", type=float, default=2.0,
                   help="same meaning as in dashboard.py")
    p.add_argument("--instance-threshold-m", type=float, default=0.5,
                   help="same meaning as in merge_static_trials.py")
    p.add_argument("--localize", nargs="?", const="all", default=None,
                   help="add the multi-object localization: 'all' "
                        "(default if you write just --localize) or a list "
                        "like mht,cphd among: " + ",".join(METHODS))
    p.add_argument("--localize-interval", type=float, default=0.3,
                   help="how often to recompute it, in seconds (default 0.3)")
    add_sensor_arguments(p)
    args = p.parse_args()

    # Resolved to ABSOLUTE paths relative to the folder you launch it from,
    # BEFORE the chdir below (otherwise "--state-dir live/" would break
    # when the script changes working directory).
    args.state_dir = os.path.abspath(args.state_dir)
    if args.regions:
        args.regions = os.path.abspath(args.regions)
    args.regions_bounds = _read_regions(args.regions)
    args.loc_methods = args.loc_sensor = args.loc_zones = None
    if args.localize is not None:
        if not args.regions:
            p.error("--localize requires --regions (arena bounds and zones)")
        args.loc_methods = methods_from_argument(args.localize)
        arena = arena_from_regions(args.regions)
        args.loc_sensor = sensor_from_arguments(args, arena)
        args.loc_zones = zones_from_arguments(args, arena)
        print("localization:", ", ".join(args.loc_methods), "--",
              args.loc_sensor.describe())
    ARGS = args

    # SimpleHTTPRequestHandler serves files from the CURRENT working
    # directory -- we move here into the script's folder so map.html is
    # always found, regardless of where you launch "python map_server.py".
    os.chdir(SCRIPT_DIR)

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True

    def wait_for_quit():
        try:
            while True:
                line = input()
                if line.strip().lower() == "q":
                    break
        except EOFError:
            pass
        httpd.shutdown()

    # Only if stdin is a real interactive terminal: if the script is
    # launched by something that does not provide an interactive stdin (a
    # launcher, a service), input() would immediately raise EOFError and
    # this thread would shut the server down an instant after start --
    # silent and very confusing to diagnose. In that case the server stays
    # up until Ctrl+C (SIGINT) arrives, no "q" shortcut.
    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=wait_for_quit, daemon=True).start()

    print(f"Live map: open http://localhost:{args.port}/map.html in the "
          f"browser (state read from {args.state_dir}/). 'q' + enter or "
          f"Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()