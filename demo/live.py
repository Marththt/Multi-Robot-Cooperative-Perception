"""Live demo: every method localises the boxes continuously while the robots stream their cameras, and a page
in the browser shows them side by side. Move the robots by hand, move, add or take away boxes, put objects in
the way: the panels follow within a second (the image networks within a few seconds).

Two terminals (the full commands are in docs/commands.md):

    python -m demo.live_robots --robot P0:8 --robot P1:7 ...     # environment of the robots: cameras, OptiTrack
    python -m demo.live                                         # environment with torch: http://localhost:8000

demo/live_robots.py keeps every robot streaming, in data/live/, its camera pose, the markers it detects and its
last frame, and writes the boxes OptiTrack tracks in the arena. This server runs two loops side by side, so the
slow one never holds up the fast one:
  detections   MHT, MHT with sensor model and the graph network on the markers of the last second of every
               robot (a few times per second)
  images       the last frame of every robot segmented (Grounding DINO + SAM) and given to the image network on
               masks alone and on masks + markers (as fast as the GPU allows)
Each panel of the page shows the boxes its method finds now against the boxes tracked by OptiTrack, with the
verdict of the thesis (correct, too few, too many, misplaced), how old its input is and how often it was correct
in the last 30 seconds; above the panels, the difficulty level of the scene as it is now. A robot whose state
stops arriving (its process stopped, or OptiTrack lost it) is left out.

Speed. The robots and the true boxes on the maps are updated ten times per second by a thread of their own,
whatever the methods are doing; the page asks for the state ten times per second and moves the robots smoothly
between two states. The image loop runs in a process of its own (its pre- and post-processing would otherwise
hold up the other loops), segments the frames of all the robots in one batch and, on CUDA, in half precision
(--no-half: full precision; --image-thread: in a thread of this process, as before).

The camera pitch and yaw offset needed by the image networks are fitted continuously on the markers the robots
see (as bevnet/real.py does on a run); --camera-calib loads fixed values, --write-camera-calib saves the fitted
ones every few seconds.

Rehearsal without robots: --replay plays a recorded run as if it were live (one round every --replay-every
seconds), writing the files demo/live_robots.py writes:

    python -m demo.live --replay data/survey_v2/R01_20261109-153345F --replay-every 4
"""

import argparse
import glob
import json
import multiprocessing
import os
import queue
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from demo import methods

LABELS = {"mht": "MHT", "mht_sensor": "MHT with sensor model", "gnn": "Graph network",
          "image": "Image network (masks)", "image_mk": "Image network (masks + markers)"}
INPUT = {"mht": "detections", "mht_sensor": "detections", "gnn": "detections", "image": "camera frames",
         "image_mk": "camera frames + markers"}
PAGE = os.path.join(os.path.dirname(__file__), "map.html")


class Live:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.recent_lock = threading.Lock()
        self.state = dict(status="starting", methods=[dict(key=k, label=LABELS[k], input=INPUT[k]) for k in LABELS],
                          unavailable={}, results={}, robots={}, truth=None, level=None, detections=[], frames={},
                          camera={}, loops={})
        os.makedirs(a.work, exist_ok=True)
        self.classical = self.gnn = self.images = None

    def set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def unavailable(self, keys, error):
        print(f"not available: {', '.join(LABELS[k] for k in keys)}: {error}", flush=True)
        with self.lock:
            for k in keys:
                self.state["unavailable"][k] = f"{type(error).__name__}: {error}"

    def publish(self, results, truth, t_input, seconds):
        """Store the results of one loop, judged against the true boxes they were computed with."""
        out = {}
        for key, est in results.items():
            r = methods.judge(est, [b[:2] for b in truth]) if truth is not None else dict(
                verdict=None, n_est=len(est), n_true=None, n_matched=None, rmse_cm=None, est_ok=[False] * len(est),
                truth_ok=[])
            r.update(boxes=[[round(p[0], 3), round(p[1], 3)] for p in est], t=t_input, seconds=round(seconds, 2))
            out[key] = r
        with self.lock:
            self.state["results"].update(out)

    # ---- loops --------------------------------------------------------------------------------------------

    def scene_info(self, robots, truth, with_level=True):
        """Robots (last state of each) and true boxes as the maps draw them; the level of the scene too."""
        level = self.state.get("level")
        if with_level:
            from tools.scene_levels import scene_level
            level = None
            if robots and truth and len(truth) > 0:
                lv, gap, views = scene_level([b[:2] for b in truth], [tuple(s["pose"]) for s in robots.values()])
                level = dict(level=lv, gap_cm=round(100 * gap) if gap < 9 else None, min_clear_views=views)
        with self.lock:
            self.state["robots"] = {q: dict(pose=[round(v, 3) for v in s["pose"]], n=len(s.get("detections", [])),
                                            age=round(time.time() - s["t"], 2), motion=s.get("motion"))
                                    for q, s in robots.items()}
            self.state["truth"] = [[round(v, 3) for v in b] for b in truth] if truth is not None else None
            self.state["level"] = level

    def sample(self, recent):
        """Ten times per second: the last state of every robot and the true boxes for the maps (the level of
        the scene twice per second), and the states kept for the pooling of the detection loop."""
        last_level = 0.0
        while True:
            t0 = time.time()
            try:
                robots, truth, _t = methods.snapshot(self.a.live_dir, self.a.stale_s)
                with self.recent_lock:
                    recent.add(robots)
                level_now = t0 - last_level >= 0.5
                self.scene_info(robots, truth, with_level=level_now)
                if level_now:
                    last_level = t0
            except Exception:
                traceback.print_exc()
            time.sleep(max(0.0, self.a.state_every_s - (time.time() - t0)))

    def detection_loop(self):
        try:
            self.classical = methods.Classical(self.a.sensor_calib, self.a.sensor_model, self.a.regions)
        except Exception as e:
            self.unavailable(["mht", "mht_sensor"], e)
        try:
            self.gnn = methods.Gnn(self.a.gnn, self.a.sensor_calib)
        except Exception as e:
            self.unavailable(["gnn"], e)
        run_dir = os.path.join(self.a.work, "round")
        recent = methods.Recent(self.a.live_dir)
        threading.Thread(target=self.sample, args=(recent,), daemon=True).start()
        while True:
            t0 = time.time()
            try:
                with self.recent_lock:
                    robots = recent.pooled(self.a.window_s, self.a.stale_s)
                with self.lock:
                    truth = self.state["truth"]
                if not robots:
                    self.set(status=f"waiting for the robots (state files in {self.a.live_dir})")
                else:
                    self.set(status=f"live: {len(robots)} robot(s)" + ("" if truth is not None else
                                                                        ", no boxes from OptiTrack"))
                    methods.write_round(run_dir, robots, truth)
                    t_input = min(s["t"] for s in robots.values())
                    results, detections = {}, []
                    if self.classical:
                        res, detections = self.classical.run(run_dir)
                        results.update(res)
                    if self.gnn:
                        results["gnn"] = self.gnn.run(run_dir)
                    self.publish(results, truth, t_input, time.time() - t0)
                    with self.lock:
                        self.state["detections"] = [[round(d[0], 3), round(d[1], 3), d[2]] for d in detections]
                        self.state["loops"]["detections"] = round(time.time() - t0, 2)
            except Exception as e:
                self.set(status=f"error in the detection loop: {type(e).__name__}: {e}")
                traceback.print_exc()
            time.sleep(max(0.0, self.a.every_s - (time.time() - t0)))

    def image_loop(self):
        """Receives the results of the image worker (a process of its own, or a thread with --image-thread)."""
        if self.a.image_thread:
            q = queue.Queue()
            threading.Thread(target=image_worker, args=(self.a, q), daemon=True).start()
        else:
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()
            ctx.Process(target=image_worker, args=(self.a, q), daemon=True).start()
        while True:
            msg = q.get()
            kind = msg[0]
            if kind == "unavailable":
                self.unavailable(["image", "image_mk"], RuntimeError(msg[1]))
                return
            if kind == "error":
                self.set(status=f"error in the image loop: {msg[1]}")
                continue
            _, results, truth, t_input, seconds, frames, camera = msg
            self.publish(results, truth, t_input, seconds)
            with self.lock:
                self.state["frames"] = {r: f"{f}?t={time.time():.2f}" for r, f in frames.items()}
                self.state["camera"] = {r: [round(v, 1) for v in c] for r, c in camera.items()}
                self.state["loops"]["images"] = round(seconds, 2)


def image_worker(a, out):
    """The image networks, in their own process: the last frame of every robot segmented (one batch) and given
    to both networks, as fast as the GPU allows; every result is sent to the page server through `out`."""
    try:
        try:
            images = methods.ImageNets({"image": a.image, "image_mk": a.image_markers}, a.work, a.device,
                                       half=not a.no_half)
            if a.camera_calib:
                images.calibration.load(a.camera_calib)
        except Exception as e:
            traceback.print_exc()
            out.put(("unavailable", f"{type(e).__name__}: {e}"))
            return
        print(f"image networks ready on {images.device}" + (" (half precision)" if images.seg.half else ""),
              flush=True)
        last_save, last_frames = 0.0, None
        while True:
            t0 = time.time()
            try:
                robots, truth, _t = methods.snapshot(a.live_dir, a.stale_s)
                frames_now = {q: s.get("frame_t") for q, s in robots.items()}
                if not robots or frames_now == last_frames:      # nothing new to segment
                    time.sleep(0.05)
                    continue
                last_frames = frames_now
                t_input = min(s.get("frame_t") or s["t"] for s in robots.values())
                results, frames, camera = images.run(a.live_dir, robots, truth)
                out.put(("result", results, truth, t_input, time.time() - t0, frames, camera))
                if a.write_camera_calib and time.time() - last_save > 5:
                    images.calibration.save(a.write_camera_calib)
                    last_save = time.time()
            except Exception as e:
                traceback.print_exc()
                out.put(("error", f"{type(e).__name__}: {e}"))
                time.sleep(1.0)
    except KeyboardInterrupt:                            # Ctrl+C reaches this process too
        pass


# ====================================================================
# Rehearsal: a recorded run played as if it were live
# ====================================================================

def replay(run_dir, live_dir, every_s):
    """Writes, round after round of a recorded run, the files the robot processes (static_trial --live) and
    demo/boxes.py write: per robot the first frame of the round with its state, and the true boxes."""
    from bevnet import real
    _layout, boxes, frames, cams = real.load_run(run_dir)
    os.makedirs(live_dir, exist_ok=True)
    rounds = sorted(k for k in frames if k[1] == 0 and k[2] == 0)
    print(f"replay: {len(rounds)} rounds of {run_dir}, {every_s:g} s each", flush=True)
    while True:
        try:
            replay_rounds(run_dir, live_dir, every_s, rounds, frames, cams, boxes)
        except Exception:                                # never let the rehearsal stop
            traceback.print_exc()
            time.sleep(1.0)


def replay_rounds(run_dir, live_dir, every_s, rounds, frames, cams, boxes):
    """One pass over the rounds of the run (see replay)."""
    for key in rounds:
        states = {}
        for q, r in frames[key].items():
            K, D = cams[q]
            src = os.path.join(run_dir, r["image"].replace("\\", "/"))
            have = os.path.exists(src)
            if have:
                shutil.copyfile(src, os.path.join(live_dir, f"{q}.jpg.tmp"))
                methods.replace(os.path.join(live_dir, f"{q}.jpg.tmp"), os.path.join(live_dir, f"{q}.jpg"))
            dets = r.get("detections", [])
            states[q] = dict(robot_id=q, pose=r["camera_pose"], detections=dets, p_d=1.0,
                             camera_matrix=K.tolist(), dist_coeffs=D.tolist(),
                             frame=f"{q}.jpg" if have else None,
                             frame_pose=r["camera_pose"], frame_detections=dets,
                             image_size=[int(r.get("width", 1280)), int(r.get("height", 720))])
        end = time.time() + every_s
        while time.time() < end:                     # rewritten as the robot processes do
            now = time.time()
            for q, st in states.items():
                methods.write_json(os.path.join(live_dir, f"{q}.json"), dict(st, t=now, frame_t=now))
            methods.write_json(os.path.join(live_dir, "boxes.json"), dict(t=now, robots={}, boxes=[
                dict(rb_id=None, x=float(b[0]), y=float(b[1]), yaw=0.0) for b in boxes]))
            time.sleep(0.3)


# ====================================================================
# Page
# ====================================================================

def handler(live):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    self.send(f.read(), "text/html; charset=utf-8")
            elif path == "/state":
                with live.lock:
                    body = json.dumps(dict(live.state, now=time.time())).encode()
                self.send(body, "application/json")
            elif path.startswith("/frames/"):
                full = os.path.join(live.a.work, "frames", os.path.basename(path))   # a file name only
                if os.path.exists(full):
                    data = methods.read_bytes(full)           # read and closed before sending
                    if data:
                        self.send(data, "image/jpeg")
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            """A move command for one robot ("*" = stop every robot): the robot process drives it
            (acquisition/live_drive.py)."""
            if self.path.split("?")[0] != "/command":
                self.send_error(404)
                return
            try:
                c = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                self.send_error(400)
                return
            robot, kind = str(c.get("robot", "")), c.get("kind")
            if kind not in ("goto", "turn", "face", "stop") or not (robot == "*" or robot.isalnum()):
                self.send_error(400)
                return
            if robot == "*":
                names = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(live.a.live_dir, "*.json"))
                         if os.path.basename(p) != "boxes.json" and not p.endswith(".cmd.json")]
            else:
                names = [robot]
            seq = int(time.time() * 1000)
            for q in names:
                cmd = dict(seq=seq, kind=kind if robot != "*" else "stop")
                if kind == "goto":
                    cmd.update(x=float(c["x"]), y=float(c["y"]),
                               yaw_deg=None if c.get("yaw_deg") is None else float(c["yaw_deg"]))
                elif kind == "turn":
                    cmd.update(deg=float(c.get("deg", 0.0)))
                methods.write_json(os.path.join(live.a.live_dir, f"{q}.cmd.json"), cmd)
            self.send(json.dumps(dict(ok=True, robots=names, seq=seq)).encode(), "application/json")

        def log_message(self, *args):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-dir", default="data/live/", help="state files of the robots and boxes.json")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--work", default="data/demo_work/", help="working files of the page")
    ap.add_argument("--every-s", type=float, default=0.15, help="period of the detection loop")
    ap.add_argument("--state-every-s", type=float, default=0.05, help="period of the robots and boxes on the maps")
    ap.add_argument("--window-s", type=float, default=0.6,
                    help="detections of the last seconds pooled per object (a capture pools the 5 frames of a stop)")
    ap.add_argument("--stale-s", type=float, default=2.0, help="a robot silent for longer is left out")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--sensor-model", default="config/sensor_model.json")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--gnn", default="results/plan/M4/gnn/detection_full_full_h2",
                    help="variant folder of gnn/train_sim.py (variant.json, model_seed0.pt)")
    ap.add_argument("--image", default="results/bev_mask_wp/masks_wp",
                    help="image network on masks: variant folder of bevnet/train.py (variant.json, model.pt)")
    ap.add_argument("--image-markers", default="results/bev_mask_mk/masks_mk", help="image network on masks + markers")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu (segmentation and image networks)")
    ap.add_argument("--no-images", action="store_true", help="only the methods on the detections")
    ap.add_argument("--no-half", action="store_true", help="segmentation in full precision on CUDA too")
    ap.add_argument("--image-thread", action="store_true",
                    help="the image networks in a thread of this process instead of a process of their own")
    ap.add_argument("--camera-calib", default=None, help="JSON {robot: [pitch deg, yaw offset deg]}: fixed values")
    ap.add_argument("--write-camera-calib", default=None, help="save the fitted camera angles to this JSON")
    ap.add_argument("--replay", default=None, help="a recorded run played as live (rehearsal without robots)")
    ap.add_argument("--replay-every", type=float, default=4.0, help="with --replay: seconds per round")
    a = ap.parse_args()
    if a.replay:
        a.live_dir = os.path.join(a.work, "replay_live")
        threading.Thread(target=replay, args=(a.replay, a.live_dir, a.replay_every), daemon=True).start()
    live = Live(a)
    threading.Thread(target=live.detection_loop, daemon=True).start()
    if a.no_images:
        live.unavailable(["image", "image_mk"], RuntimeError("--no-images"))
    else:
        threading.Thread(target=live.image_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", a.port), handler(live))
    print(f"page: http://localhost:{a.port}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
