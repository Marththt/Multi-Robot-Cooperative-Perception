"""Runs the survey with all the robots from ONE command, ROUND BY ROUND.

Fixed positions (--radius R, the systematic mode): the robots stand on a
circle of radius R around the centre of the boxes (recomputed from the
ground truth for every layout; --circle-center X Y overrides it), evenly
spaced (one position per robot, the first at --first-angle-deg): the same
distance and angles for every layout. Place the robots roughly near the circle:
each one drives to the nearest free position (round 0), then, round after
round, ALL the robots rotate together on the circle by --step-deg, always
facing its centre, for --sweep-deg in total (default 360 / number of
robots: together they cover the whole circle once). A position too close
to a box or to the arena border is not used: that robot stays there in
that round (the path preview shows it).

Without --radius: round 0 is captured where the robots were placed and
the formation rotates around the boxes at the distance where each robot
was placed (default --sweep-deg 180).

The next round starts only when every robot has captured. Each robot
stands still if another robot is in its way (--others is set
automatically).

The boxes are read from OptiTrack at every launch (their rigid bodies:
--box RB:CLASS, or the rb_id saved in --ground-truth by
tools/record_ground_truth.py): the plan, the live figures and the analysis
use where the boxes are NOW. They are written in <run>/ground_truth.json and
added to the --ground-truth file under the run id. --fixed-boxes uses the
file instead.

Every launch is a NEW RUN, saved in its own folder <out>/<layout>_<date-time>/
(captures, raw readings, frames, path preview, live figures): nothing is
skipped because an earlier run was already at the same position.
pipeline/analyze_rounds.py / pipeline/build_scans.py read one run folder or the whole
<out>/ folder (every run is kept separate).

The robots are connected ONE AT A TIME (the DJI SDK looks for the robot
through a UDP port that only one process at a time can use); the first
round starts when all of them are connected. Ctrl+C stops all of them:
each robot stops its motion, closes camera and connection, then exits.

    python -m acquisition.launch_all --robot P0:8 --robot P1:7 --robot P2:9 --robot P3:10 \\
        --layout-id L2 --ground-truth config/ground_truth_positions.json \\
        --regions config/optitrack_regions.json --out survey_rounds/ \\
        --aruco-marker-size 0.077 --radius 1.3 --step-deg 15
"""

import argparse
import json
import locale
import math
import os
import signal
import subprocess
import sys
import threading
import time

CONNECTED_MSG = "robot connected"      # printed by survey.py
CONNECT_TIMEOUT_S = 60.0


def parse_robot(spec):
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--robot {spec!r}: expected ID:RIGID_BODY[:CAMERA_CALIB]")
    rid, rb = parts[0], int(parts[1])
    calib = parts[2] if len(parts) > 2 and parts[2] else None
    if calib is None:
        default = os.path.join("config", f"checkerboard_calib_{rid}.json")
        calib = default if os.path.exists(default) else None
    return dict(robot_id=rid, rb=rb, calib=calib)


def pump(proc, tag, connected):
    """Prints the output of one robot process. Reads BYTES and never stops
    on a line it cannot decode or print: if this thread died, the pipe of
    that process would fill up and the robot would freeze at its next
    print."""
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode(locale.getpreferredencoding(False) or "latin-1", errors="replace")
        try:
            if line.strip():
                print(f"{tag:>4} | {line.rstrip()}", flush=True)
        except Exception:
            try:
                print(f"{tag:>4} | {line.rstrip().encode('ascii', 'replace').decode()}", flush=True)
            except Exception:
                pass
        if CONNECTED_MSG in line:
            connected.set()
    connected.set()                      # process ended: do not keep the others waiting


class StuckReports:
    """Prints what survey.py writes in <sync dir>/stuck_<robot>.txt when a
    robot stays in one step too long (stack of every thread)."""

    def __init__(self, sync_dir, team):
        self.sync_dir, self.team = sync_dir, team
        self.offset = {r: 0 for r in team}

    def poll(self):
        for r in self.team:
            path = os.path.join(self.sync_dir, f"stuck_{r}.txt")
            try:
                if os.path.getsize(path) <= self.offset[r]:
                    continue
                with open(path, "r", errors="replace") as f:
                    f.seek(self.offset[r])
                    text = f.read()
                    self.offset[r] = f.tell()
            except OSError:
                continue
            last_phase = ""
            out = []
            for line in text.splitlines():
                if line.startswith("--- "):
                    last_phase = line[4:]
                elif line.strip():
                    out.append(line)
            if out:
                print(f"STUCK {r} | no progress in: {last_phase or 'the current step'}; where it is stuck:",
                      flush=True)
                for line in out:
                    print(f"STUCK {r} | {line}", flush=True)


def arg_value(rest, name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default


def set_arg(rest, name, value):
    """rest with `name value` (replaced if already there)."""
    if name in rest:
        i = rest.index(name)
        return rest[:i] + [name, value] + rest[i + 2:]
    return rest + [name, value]


def box_rigid_bodies(box_specs, gt_path, layout_id):
    """[(rigid body id, class)] of the boxes: from --box RB:CLASS, or else
    the rb_id saved by tools/record_ground_truth.py in the ground truth file
    (entry of --layout-id first, otherwise the last entry that has them)."""
    if box_specs:
        out = []
        for v in box_specs:
            rb, _, cls = v.partition(":")
            out.append((int(rb), cls.strip() or "box"))
        return out, "--box"
    data = {}
    if gt_path and os.path.exists(gt_path):
        with open(gt_path) as f:
            data = json.load(f)
    keys = ([layout_id] if layout_id in data else []) + [k for k in reversed(list(data)) if k != layout_id]
    for k in keys:
        entries = data.get(k) or []
        if entries and all("rb_id" in e for e in entries):
            return [(int(e["rb_id"]), e.get("class_name", "box")) for e in entries], f"rigid bodies of {k} in {gt_path}"
    return None, None


def record_boxes(objects, server, local, seconds=2.0):
    """Positions of the boxes NOW from OptiTrack (average over `seconds`):
    [dict(class_name, x, y, rb_id)], or raises SystemExit if a box is not
    tracked."""
    from tools.record_ground_truth import average_positions
    from vf.localization.mocap_pose import MocapHub
    hub = MocapHub([rb for rb, _ in objects], server, local)
    try:
        try:
            hub.wait_ready()
        except RuntimeError as e:
            raise SystemExit(f"BOXES | {e}")
        samples = {rb: [] for rb, _ in objects}
        t0 = time.time()
        while time.time() - t0 < seconds:
            for rb, _ in objects:
                q = hub.get_pose(rb)
                if q is not None:
                    samples[rb].append((q[0], q[1]))
            time.sleep(0.02)
    finally:
        hub.close()
    avg = average_positions(samples)
    missing = [rb for rb, _ in objects if rb not in avg]
    if missing:
        raise SystemExit(f"BOXES | no OptiTrack reading for the box rigid bodies {missing}: check them in Motive")
    return [dict(class_name=cls, x=round(avg[rb][0], 4), y=round(avg[rb][1], 4), rb_id=rb) for rb, cls in objects]


def boxes_for_this_run(a, rest, layout_id, run_id, gt_out):
    """Reads where the boxes are NOW (OptiTrack) and writes them to gt_out
    as {layout_id: boxes, run_id: boxes}; if run_id is given, they are also
    added to the --ground-truth file under the run id, so the analysis of
    this run always uses the boxes of this run. Returns the new rest (with
    --ground-truth gt_out), or rest unchanged in --sim / --fixed-boxes."""
    gt_path = arg_value(rest, "--ground-truth")
    if "--sim" in rest or a.fixed_boxes:
        print(f"BOXES | positions from {gt_path} ({layout_id})", flush=True)
        return rest
    objects, source = box_rigid_bodies(a.box, gt_path, layout_id)
    if not objects:
        raise SystemExit("BOXES | no rigid body ids of the boxes: give them with --box RB:CLASS (one per box), "
                         "or record the layout once with tools/record_ground_truth.py")
    entries = record_boxes(objects, arg_value(rest, "--optitrack-server", "127.0.0.1"),
                           arg_value(rest, "--optitrack-local", "127.0.0.1"))
    old = {}
    if gt_path and os.path.exists(gt_path):
        with open(gt_path) as f:
            old = {e.get("rb_id"): e for e in (json.load(f).get(layout_id) or [])}
    print(f"BOXES | read now from OptiTrack ({source}):", flush=True)
    for e in entries:
        o = old.get(e["rb_id"])
        moved = "" if o is None else f"  ({math.hypot(e['x'] - o['x'], e['y'] - o['y']):.2f} m from {layout_id} in the file)"
        print(f"BOXES |   rb {e['rb_id']:>3} {e['class_name']:<6} x={e['x']:+.3f} y={e['y']:+.3f}{moved}", flush=True)
    data = {layout_id: entries}
    if run_id:
        data[run_id] = entries
    os.makedirs(os.path.dirname(gt_out) or ".", exist_ok=True)
    with open(gt_out, "w") as f:
        json.dump(data, f, indent=2)
    if run_id and gt_path and os.path.exists(gt_path):
        try:
            with open(gt_path) as f:
                allgt = json.load(f)
            allgt[run_id] = entries
            with open(gt_path, "w") as f:
                json.dump(dict(sorted(allgt.items())), f, indent=2)
                f.write("\n")
        except (OSError, ValueError) as e:
            print(f"BOXES | could not add {run_id} to {gt_path} ({e}); the positions are in {gt_out}", flush=True)
    return set_arg(rest, "--ground-truth", gt_out)


def show_plan(robots, rest):
    """--plan-only: reads where the robots are NOW (OptiTrack) and shows the
    arc every robot would travel from there; nothing moves."""
    from acquisition import survey
    sp = argparse.ArgumentParser()
    survey.add_survey_arguments(sp)
    a, _unknown = sp.parse_known_args(rest)
    boxes, center, arena, _plan = survey.load_plan(a, 0.0, need_plan=False)
    if a.sim:
        starts = {rid: q[:2] for rid, q in
                  survey.sim_team_starts(center, arena, [r["robot_id"] for r in robots], a).items()}
    else:
        from vf.localization.mocap_pose import MocapHub
        hub = MocapHub([r["rb"] for r in robots], a.optitrack_server, a.optitrack_local)
        try:
            hub.wait_ready()
            starts = {r["robot_id"]: hub.get_pose(r["rb"])[:2] for r in robots}
        finally:
            hub.close()
    team = [r["robot_id"] for r in robots]
    lines, problems, groups, paths, _track = plan_lines(starts, team, boxes, center, arena, a,
                                                        n_stations=len(team))
    for line in lines + problems:
        print(line)
    if a.plan_png:
        survey.draw_sectors(a.plan_png, arena, a.margin, boxes, groups, team, paths,
                            circle=(survey.circle_center_of(a, arena, boxes), a.radius) if a.radius else None,
                            placed=starts if a.radius else None)
        print(f"\nplan drawn in {a.plan_png}")


def plan_lines(starts, team, boxes, center, arena, a, n_stations):
    """Text and drawing of the team plan (survey.team_plan): (lines,
    problems, groups, paths, track). problems is not empty when a fixed
    position of round 0 cannot be used. n_stations: the robots of the WHOLE
    team (also one that is not running), so the fixed positions never
    depend on which robots are up."""
    from acquisition import survey
    face, r0, arcs = survey.team_plan(starts, team, boxes, center, arena, a, n_stations=n_stations)
    lines, problems = [], []
    if a.radius is not None:
        c = survey.circle_center_of(a, arena, boxes)
        which = "centre of the boxes" if a.circle_center is None else "--circle-center"
        lines.append(f"fixed positions: circle r={a.radius:g} m around ({c[0]:+.2f}, {c[1]:+.2f}) ({which}); "
                     f"every robot {a.sweep_deg:g} deg {a.direction}, one round every {a.step_deg:g} deg")
        lines.append(f"radius that fits: from {survey.min_radius(a, arena, boxes):.2f} m (clear of every box of "
                     f"this layout by --keep-out-m {a.keep_out_m:g}) to {survey.max_radius(a, arena, boxes):.2f} m "
                     f"(--margin {a.margin:g} from the arena border)")
    else:
        lines.append(f"every robot {a.sweep_deg:g} deg {a.direction} around the boxes, one round every "
                     f"{a.step_deg:g} deg:")
    groups, paths, track = [], [], {}
    for rid in team:
        x, y = r0[rid]
        if a.radius is not None:
            why = survey.station_problem((x, y), boxes, arena, a)
            if why:
                problems.append(f"ERROR: the fixed position of {rid} ({x:+.2f}, {y:+.2f}) is {why}: use a "
                                f"smaller --radius, another --circle-center, or move the boxes")
        seq, legs, prev, pts = [dict(stop=0, x=x, y=y)], [], (x, y), [(0, x, y)]
        if a.radius is not None:
            legs.append([tuple(starts[rid][:2]), (x, y)])
        line = f"{rid}: r0 ({x:+.2f}, {y:+.2f})"
        for k, caps in enumerate(arcs[rid], start=1):
            if not caps:
                line += f"  r{k} stays"
                continue
            c = caps[0]
            seq.append(c)
            legs.append([prev, (c["x"], c["y"])])
            prev = (c["x"], c["y"])
            pts.append((k, c["x"], c["y"]))
            line += f"  r{k} ({c['x']:+.2f}, {c['y']:+.2f})"
        lines.append(line)
        groups.append(seq)
        paths.append(legs)
        track[rid] = pts
    return lines, problems, groups, paths, track


def wait_for_enter(live, stop_req):
    """Waits for ENTER on this console without blocking (the live window
    stays responsive, Ctrl+C still works). True = start, False = cancelled."""
    if os.name == "nt":
        import msvcrt
        while not stop_req.is_set():
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    return True
            live.wait(0.2)
        return False
    import select
    while not stop_req.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            sys.stdin.readline()
            return True
        live.wait(0.2)
    return False


class LiveRounds:
    """Analyses every round as soon as the whole team has finished it."""

    def __init__(self, la, rest, team, sync_dir, show):
        from acquisition import survey
        from vf.fusion.sensor_model import arena_from_regions, sensor_from_arguments, zones_from_arguments
        from vf.fusion.localization_methods import METHODS
        sp = argparse.ArgumentParser()
        survey.add_survey_arguments(sp)
        sa, _unknown = sp.parse_known_args(rest)
        self.sa = sa
        self.track = None                     # planned path of every robot (path preview)
        self.layout, self.out, self.gt = sa.layout_id, sa.out, sa.ground_truth
        self.offsets = sa.yaw_offsets_deg
        self.arena = arena_from_regions(sa.regions) if sa.regions else survey.ARENA_DEFAULT
        self.sensor = sensor_from_arguments(la, self.arena)
        self.zones = zones_from_arguments(la, self.arena)
        self.methods = list(METHODS)
        self.range_fit = la.range_fit
        self.team, self.sync_dir = team, sync_dir
        self.folder = os.path.join(self.out, "rounds_live")
        os.makedirs(self.folder, exist_ok=True)
        self.done_rounds = set()
        self.show = show
        self.fig = None
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.ion()
            except Exception as e:
                print(f"live window not available ({e}): figures only saved", flush=True)
                self.show = False

    def _points(self):
        from pipeline.build_scans import load_points, load_truth
        pts = load_points(self.out, self.range_fit)
        pts = {k: v for k, v in pts.items() if v["round"] is not None}    # round-mode captures only
        truth = load_truth(self.gt).get(self.layout, [])
        return {k: v for k, v in pts.items() if v["layout"] == self.layout}, truth

    def preview(self, starts, wait_enter):
        """Draws the path every robot is going to travel (from where it was
        placed), shows it and, with wait_enter, waits for Enter before the
        robots start. Returns False if the start is cancelled."""
        from acquisition import survey
        boxes, center, arena, _plan = survey.load_plan(self.sa, 0.0, quiet=True, need_plan=False)
        team = [r for r in self.team if r in starts]
        lines, problems, groups, paths, track = plan_lines(starts, team, boxes, center, arena, self.sa,
                                                           n_stations=len(self.team))
        for line in lines + problems:
            print(f"PATH | {line}", flush=True)
        self.track = track
        png = os.path.join(self.out, f"path_preview_{self.layout}.png")
        fixed = self.sa.radius is not None
        survey.draw_sectors(png, arena, self.sa.margin, boxes, groups, team, paths,
                            circle=(survey.circle_center_of(self.sa, arena, boxes), self.sa.radius) if fixed else None,
                            placed=starts if fixed else None)
        print(f"PATH | preview saved in {png}", flush=True)
        if self.show:
            try:
                import matplotlib.pyplot as plt
                img = plt.imread(png)
                fig = plt.figure(figsize=(7, 6.5))
                plt.imshow(img)
                plt.axis("off")
                plt.title("path preview: r0 = " + ("fixed position (x = where the robot is now)" if fixed
                                                   else "where each robot is now") + ", r1, r2, ... the rounds")
                self._display(fig)
            except Exception as e:
                print(f"PATH | cannot show the preview ({e})", flush=True)
        if problems:
            print("PATH | not started: fix the plan and launch again", flush=True)
            return False
        if not wait_enter:
            return True
        print("PATH | press ENTER here to start the robots, Ctrl+C to cancel", flush=True)
        return wait_for_enter(self, self.stop_req)

    def _scene(self, name, title, members, pts, truth, rnd, off, keep_open):
        from pipeline.analyze_rounds import draw_scene, fuse, scene_rows
        poses, per_class, results = fuse(members, pts, self.sensor, self.zones, self.methods)
        rows = scene_rows(name, self.layout, rnd, off, members, pts, truth, results, self.methods)
        fig = draw_scene(os.path.join(self.folder, name + ".png"), title, members, pts, truth, poses,
                         per_class, results, self.methods, self.sensor, self.arena, rows, keep_open=keep_open,
                         track=self.track, round_now=rnd if isinstance(rnd, int) else None)
        return rows, fig

    def _display(self, fig):
        import matplotlib.pyplot as plt
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = fig
        fig.show()
        plt.pause(0.05)

    def poll(self):
        """Analyse the rounds completed since the last call."""
        k = 0
        while True:
            if k not in self.done_rounds:
                marks = [os.path.exists(os.path.join(self.sync_dir, f"r{k}_{r}")) for r in self.team]
                if not all(marks):
                    break
                self.analyse_round(k)
                self.done_rounds.add(k)
            k += 1

    def analyse_round(self, k):
        from pipeline.analyze_rounds import METHOD_LABELS, round_line
        try:
            pts, truth = self._points()
            if not truth:
                print(f"LIVE | no ground truth for {self.layout}: round {k} not analysed", flush=True)
                return
            header = "".join(f"{METHOD_LABELS[m]:>14}" for m in self.methods)
            for i, off in enumerate(self.offsets):
                members = sorted(key for key, v in pts.items() if v["round"] == k and v["yaw_offset"] == off)
                if not members:
                    continue
                name = f"{self.layout}_r{k:02d}_{off:+.0f}"
                show_it = self.show and i == 0
                rows, fig = self._scene(name, f"LIVE  layout {self.layout}, round {k}, yaw offset {off:+.0f} deg",
                                        members, pts, truth, k, off, keep_open=show_it)
                if i == 0:
                    print(f"LIVE | round {k} (estimated/true, OSPA):{header}", flush=True)
                print(f"LIVE | round {k} offset {off:+4.0f} deg:          {round_line(rows, self.methods)}",
                      flush=True)
                if show_it:
                    self._display(fig)
        except Exception as e:
            print(f"LIVE | round {k}: analysis failed ({e})", flush=True)

    def overall(self):
        from pipeline.analyze_rounds import round_line
        try:
            pts, truth = self._points()
            if not pts or not truth:
                return
            rows, fig = self._scene(f"{self.layout}_overall",
                                    f"layout {self.layout}, ALL rounds together ({len(pts)} captures)",
                                    sorted(pts), pts, truth, "all", "all", keep_open=self.show)
            print(f"LIVE | ALL rounds together:                {round_line(rows, self.methods)}", flush=True)
            if self.show:
                self._display(fig)
        except Exception as e:
            print(f"LIVE | overall analysis failed ({e})", flush=True)

    def wait(self, seconds):
        if self.show:
            import matplotlib.pyplot as plt
            plt.pause(seconds)
        else:
            time.sleep(seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", action="append", required=True, help="ID:RIGID_BODY[:CAMERA_CALIB], one per robot")
    p.add_argument("--no-live", action="store_true", help="no live window (the round figures are still saved)")
    p.add_argument("--no-confirm", action="store_true",
                   help="do not wait for ENTER after the path preview: the robots start at once")
    p.add_argument("--range-fit", type=float, nargs=2, default=None, metavar=("A", "B"),
                   help="live analysis: distance correction measured = A * true + B")
    p.add_argument("--box", action="append", default=[], metavar="RB:CLASS",
                   help="OptiTrack rigid body of one box (repeat for every box); default: the rigid bodies "
                        "saved in --ground-truth for --layout-id")
    p.add_argument("--fixed-boxes", action="store_true",
                   help="do NOT read the boxes from OptiTrack: use the positions in --ground-truth")
    from vf.fusion.sensor_model import add_sensor_arguments
    add_sensor_arguments(p)
    a, rest = p.parse_known_args()
    robots = [parse_robot(s) for s in a.robot]
    layout_id = arg_value(rest, "--layout-id", "run")
    if "--plan-only" in rest:
        import tempfile
        rest = boxes_for_this_run(a, rest, layout_id, None,
                                  os.path.join(tempfile.gettempdir(), "ground_truth_now.json"))
        show_plan(robots, [x for x in rest if x != "--plan-only"])
        return

    # every launch is a new run in its own folder: <out>/<layout>_<date-time>/
    out = "data/survey/"
    if "--out" in rest:
        i = rest.index("--out")
        out = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    run_id = f"{layout_id}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(out, run_id)
    os.makedirs(run_dir, exist_ok=True)
    rest = rest + ["--out", run_dir, "--run-id", run_id]
    # the boxes of THIS run, read now from OptiTrack: plan, live figures and
    # analysis all use them (written in <run>/ground_truth.json)
    rest = boxes_for_this_run(a, rest, layout_id, run_id, os.path.join(run_dir, "ground_truth.json"))
    sync_dir = os.path.join(run_dir, "sync")
    print(f"run {run_id}: data in {run_dir}", flush=True)
    team = [r["robot_id"] for r in robots]

    # Ctrl+C is handled HERE only: the survey.py processes ignore it and are
    # told to stop through the file "abort" in the sync folder, so each one
    # stops its robot, closes the camera and the connection, and exits by
    # itself. The processes are started exactly as before (same console, no
    # redirected input): the robot connection is not affected.
    stop_req = threading.Event()

    def on_ctrl_c(signum, frame):
        if stop_req.is_set():
            print("Ctrl+C again: still stopping the robots, please wait", flush=True)
        stop_req.set()

    signal.signal(signal.SIGINT, on_ctrl_c)

    procs = []
    for k, r in enumerate(robots):
        if stop_req.is_set():
            break
        others = [str(o["rb"]) for o in robots if o is not r]
        cmd = [sys.executable, "-u", "-m", "acquisition.survey", "--robot-id", r["robot_id"],
               "--optitrack-rb-id", str(r["rb"]), "--team"] + team + ["--sync-dir", sync_dir,
               "--ignore-ctrl-c", "--wait-go"]
        if others:
            cmd += ["--others"] + others
        if r["calib"]:
            cmd += ["--camera-calib", r["calib"]]
        cmd += rest
        print(f"{r['robot_id']}: rigid body {r['rb']}, calibration {r['calib'] or 'NONE'}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc.robot_id = r["robot_id"]
        connected = threading.Event()
        threading.Thread(target=pump, args=(proc, r["robot_id"], connected), daemon=True).start()
        procs.append(proc)
        t0 = time.time()
        while not connected.is_set() and not stop_req.is_set() and time.time() - t0 < CONNECT_TIMEOUT_S:
            time.sleep(0.1)
        if not connected.is_set() and not stop_req.is_set():
            print(f"{r['robot_id']}: not connected after {CONNECT_TIMEOUT_S:.0f} s, starting the next robot",
                  flush=True)
        time.sleep(1.0)                  # let the SDK release the discovery port

    live = LiveRounds(a, rest, team, sync_dir, show=not a.no_live)
    live.stop_req = stop_req
    # path preview: as soon as every running robot has published where it
    # stands, then the robots wait for "go"
    starts = None
    while not stop_req.is_set() and any(pr.poll() is None for pr in procs):
        alive = [pr.robot_id for pr in procs if pr.poll() is None]
        files = {r: os.path.join(sync_dir, f"start_{r}.json") for r in alive}
        if all(os.path.exists(f) for f in files.values()):
            starts = {}
            for r, f in files.items():
                with open(f) as fh:
                    q = json.load(fh)
                starts[r] = (q["x"], q["y"])
            break
        live.wait(0.2)
    if starts and not stop_req.is_set():
        if live.preview(starts, wait_enter=not a.no_confirm):
            with open(os.path.join(sync_dir, "go"), "w") as f:
                f.write("go\n")
            print("PATH | start confirmed: the robots begin", flush=True)
        else:
            stop_req.set()
    stuck = StuckReports(sync_dir, team)
    while any(pr.poll() is None for pr in procs) and not stop_req.is_set():
        live.poll()
        stuck.poll()
        live.wait(0.3)
    if stop_req.is_set():
        stop_all(procs, sync_dir)
    else:
        live.poll()
        live.overall()
    print(f"all robots finished; data in {run_dir}, round figures in {live.folder}", flush=True)
    if live.show and live.fig is not None and not stop_req.is_set():
        import matplotlib.pyplot as plt
        print("close the figure window to exit", flush=True)
        plt.ioff()
        plt.show()


def stop_all(procs, sync_dir, grace_s=30.0):
    """Asks every survey.py to stop (file "abort" in the sync folder) and
    waits for them to stop their robot and close the connection; only a
    process that does not exit in time is killed (its robot then stops by
    itself within 0.3 s, its watchdog)."""
    print("STOP: stopping all robots and closing the connections...", flush=True)
    try:
        os.makedirs(sync_dir, exist_ok=True)
        with open(os.path.join(sync_dir, "abort"), "w") as f:
            f.write("stop requested\n")
    except OSError as e:
        print(f"cannot write the abort file ({e}): killing the processes", flush=True)
        grace_s = 0.0
    t0 = time.time()
    while any(pr.poll() is None for pr in procs) and time.time() - t0 < grace_s:
        time.sleep(0.2)
    for pr in procs:
        if pr.poll() is None:
            print(f"{pr.robot_id}: not stopped after {grace_s:.0f} s, killed "
                  f"(its robot stops by itself within 0.3 s, its watchdog)", flush=True)
            pr.kill()
    for pr in procs:
        print(f"{pr.robot_id}: exit code {pr.poll()}", flush=True)


if __name__ == "__main__":
    main()