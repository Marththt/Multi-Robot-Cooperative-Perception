"""Driving a robot during the live demo, while its camera keeps streaming.

acquisition/static_trial.py --live --commands <file> starts a LiveDriver next to the camera loop. The page of
the demo (demo/live.py) writes the latest command for this robot into <file>; the driver picks it up within a
fraction of a second and drives with the controller of the captures (vf/control/goto.py), on a path that keeps
clear of the boxes and of the other robots (acquisition/survey.route), inside the arena. A new command replaces
the one in progress; "stop" halts the robot at once.

Commands (JSON, written atomically; seq grows with every command):
    {"seq": 7, "kind": "goto", "x": 0.4, "y": -1.2, "yaw_deg": null}   drive there; yaw_deg null = arrive
                                                                        facing the centre of the boxes
    {"seq": 8, "kind": "turn", "deg": 22.5}                              turn on the spot (+ = left)
    {"seq": 9, "kind": "face"}                                           turn to face the centre of the boxes
    {"seq": 10, "kind": "stop"}

The state of the last command (seq, state: driving / reached / failed / blocked / refused / stopped, goal,
note) is written with the live state of the robot, so the page shows it.
"""

import glob
import json
import math
import os
import threading
import time

from vf.control.goto import go_to, stop, wrap_pi

KEEP_OUT_M = 0.45            # no leg closer than this to a box centre (as the captures, survey --keep-out-m)
MARGIN_M = 0.35              # goals at least this far inside the arena border
STOP_DISTANCE_M = 0.6        # wait while another robot is this close and in the way (survey --stop-distance)
ROBOT_CLEARANCE_M = 0.6      # no leg closer than this to a standing robot
STALE_S = 2.0


def read_json(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return json.loads(data) if data else None
    except (OSError, ValueError):
        return None


class LiveDriver(threading.Thread):
    def __init__(self, chassis, pose_fn, command_path, live_dir, robot_id, arena, log=print):
        super().__init__(daemon=True)
        self.chassis, self.pose_fn, self.path = chassis, pose_fn, command_path
        self.live_dir, self.robot_id, self.arena, self.log = live_dir, robot_id, arena, log
        self.quit = threading.Event()
        old = read_json(command_path)                       # a command left from an earlier session is not run
        self.seq = old.get("seq") if old else None
        self.status = dict(seq=self.seq, state="idle", goal=None, note="")

    # ---- what the robot knows of the scene --------------------------------------------------------------

    def boxes(self):
        b = read_json(os.path.join(self.live_dir, "boxes.json"))
        if not b or time.time() - float(b.get("t", 0)) > STALE_S:
            return []
        return [dict(x=float(x["x"]), y=float(x["y"])) for x in b.get("boxes", [])]

    def others(self):
        """Positions of the other robots (their live states)."""
        out = []
        for path in glob.glob(os.path.join(self.live_dir, "*.json")):
            if os.path.basename(path) == "boxes.json" or path.endswith(".cmd.json"):
                continue
            s = read_json(path)
            if (s and s.get("robot_id") != self.robot_id and s.get("pose") is not None
                    and time.time() - float(s.get("t", 0)) <= STALE_S):
                out.append((float(s["pose"][0]), float(s["pose"][1])))
        return out

    def centre(self, boxes):
        if not boxes:
            return None
        return (sum(b["x"] for b in boxes) / len(boxes), sum(b["y"] for b in boxes) / len(boxes))

    # ---- commands ---------------------------------------------------------------------------------------

    def newer(self):
        """True when a command newer than the one running has arrived (or the process is quitting)."""
        if self.quit.is_set():
            return True
        c = read_json(self.path)
        return bool(c) and c.get("seq") != self.seq

    def report(self, state, note="", goal=None):
        self.status = dict(seq=self.seq, state=state, goal=goal if goal is not None else self.status.get("goal"),
                           note=note)
        if state not in ("driving",):
            self.log(f"[drive] command {self.seq}: {state}" + (f" ({note})" if note else ""))

    def run(self):
        while not self.quit.is_set():
            c = read_json(self.path)
            if c and c.get("seq") != self.seq:
                self.seq = c.get("seq")
                try:
                    self.execute(c)
                except Exception as e:                       # never leave the robot moving
                    stop(self.chassis)
                    self.report("failed", f"{type(e).__name__}: {e}")
            time.sleep(0.15)
        stop(self.chassis)

    def execute(self, c):
        from acquisition.survey import border_distance, inside, route
        kind = c.get("kind")
        if kind == "stop":
            stop(self.chassis)
            self.report("stopped")
            return
        here = self.pose_fn()
        if here is None:
            self.report("failed", "no pose from OptiTrack")
            return
        boxes = self.boxes()
        centre = self.centre(boxes)
        limit = min(MARGIN_M / 2, border_distance(here, self.arena) - 0.03)

        def fence(q):
            return border_distance(q, self.arena) >= limit

        def facing(xy):
            return None if centre is None else math.atan2(centre[1] - xy[1], centre[0] - xy[0])

        if kind in ("turn", "face"):
            yaw = wrap_pi(here[2] + math.radians(float(c.get("deg", 0.0)))) if kind == "turn" else facing(here)
            if yaw is None:
                self.report("refused", "no boxes to face")
                return
            self.report("driving", goal=[here[0], here[1], math.degrees(yaw)])
            ok = go_to(self.pose_fn, self.chassis, here[:2], yaw, fence=fence, pos_tol=0.15,
                       cancel=self.newer, log=self.log)
            self.report("reached" if ok else ("stopped" if self.newer() else "failed"))
            return
        if kind != "goto":
            self.report("refused", f"unknown command {kind!r}")
            return
        goal = (float(c["x"]), float(c["y"]))
        yaw = math.radians(c["yaw_deg"]) if c.get("yaw_deg") is not None else facing(goal)
        goal_pose = [goal[0], goal[1], None if yaw is None else math.degrees(yaw)]
        if not inside(goal, self.arena, MARGIN_M):
            self.report("refused", "goal too close to the arena border", goal_pose)
            return
        if boxes and min(math.hypot(b["x"] - goal[0], b["y"] - goal[1]) for b in boxes) < KEEP_OUT_M:
            self.report("refused", "goal too close to a box", goal_pose)
            return
        robots = self.others()
        if boxes:
            path = route(here[:2], goal, centre, boxes, KEEP_OUT_M, self.arena, MARGIN_M,
                         robots=robots, robot_clearance_m=ROBOT_CLEARANCE_M)
        else:
            path = [goal]
        if path is None:
            self.report("blocked", "no path clear of the boxes and the other robots", goal_pose)
            return
        self.report("driving", goal=goal_pose)
        for k, wp in enumerate(path):
            last = k == len(path) - 1
            ok = go_to(self.pose_fn, self.chassis, wp, yaw if last else None, others=self.others,
                       stop_distance_m=STOP_DISTANCE_M, fence=fence, cancel=self.newer, log=self.log)
            if not ok:
                self.report("stopped" if self.newer() else "failed",
                         "" if self.newer() else "stopped on the way (timeout, tracking or fence)")
                return
        self.report("reached")

    def close(self):
        self.quit.set()
        self.join(timeout=3.0)
        stop(self.chassis)
