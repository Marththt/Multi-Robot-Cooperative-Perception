"""The methods of the live demo, each behind one call on the latest state of the robots.

Input (demo/live.py reads it a few times per second): per robot, the state written by
acquisition/static_trial.py --live --state-out --frame-out (camera pose from OptiTrack, every marker detected
in the last frame with distance, bearing, viewing angle and rectangle, the last saved camera frame with the
pose and the detections of that frame), and the boxes tracked by OptiTrack (demo/boxes.py).

The wrappers reuse the code of the offline evaluation, so the robots' data are processed as in the thesis:
  Classical   MHT and MHT with sensor model    the state is written as one round of a run and read back with
                                               pipeline/build_scans.load_points + pipeline/analyze_rounds.fuse
  Gnn         graph network on the detections  the same round read with gnn/dataset.build_scenes, gnn/model
  ImageNets   image network, masks alone and   frames segmented with bevnet/segment.py, scene built as in
              masks + detected markers         bevnet/real.build_scene, prediction as bevnet/train.py

A capture keeps, for every marker, the reading with the median distance among the 5 frames of a stop; live, the
methods on the detections get the same from the frames of the last second (Recent.pooled), while the image
networks read the last saved frame.

Heavy packages (torch, transformers) are imported inside the wrappers, so a method whose packages or model
files are missing is reported as unavailable and the others keep working.
"""

import argparse
import collections
import json
import math
import os
import pickle
import time
import types

import numpy as np

CLASSICAL = ("mht", "mht_sensor")
LAYOUT = "LIVE"


# ====================================================================
# Input
# ====================================================================

def read_bytes(path):
    """The whole file, read at once and closed at once: on Windows a file open for reading cannot be replaced,
    so the writers (which retry, see replace) are held up as briefly as possible. None if missing or busy."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def read_json(path):
    data = read_bytes(path)
    try:
        return json.loads(data) if data else None
    except ValueError:                          # replaced while being read: next time
        return None


def replace(tmp, path, attempts=20):
    """os.replace that waits for a reader to close the file (Windows refuses to replace an open file).
    Returns False if the file stayed busy: the caller skips this update, the next one will go through."""
    for _ in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.01)
    return False


def write_json(path, data):
    """Atomic write (temporary file + rename), so a reader never sees half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    return replace(tmp, path)


def snapshot(live_dir, stale_s=2.0):
    """Latest state: (robots {name: state}, true boxes [[x, y, yaw], ...] or None, time of the boxes).
    A robot whose state is older than stale_s (its process stopped, or it lost OptiTrack) is left out."""
    now = time.time()
    robots = {}
    for name in sorted(os.listdir(live_dir)) if os.path.isdir(live_dir) else []:
        if not name.endswith(".json") or name == "boxes.json" or name.endswith(".cmd.json"):
            continue
        s = read_json(os.path.join(live_dir, name))
        if s and s.get("pose") is not None and now - float(s.get("t", 0)) <= stale_s:
            robots[s.get("robot_id", name[:-5])] = s
    b = read_json(os.path.join(live_dir, "boxes.json"))
    if b and now - float(b.get("t", 0)) <= stale_s:
        return robots, [[float(x["x"]), float(x["y"]), float(x.get("yaw") or 0.0)] for x in b["boxes"]], b["t"]
    return robots, None, None


class Recent:
    """The last states of every robot (each state file is rewritten several times per second), so that the
    detections of the last moment can be pooled as a capture pools the readings of a stop."""

    GROUP_M = 0.2                       # detections of one robot closer than this are one object (as in the captures)

    def __init__(self, live_dir, keep=40):
        self.dir = live_dir
        self.states = collections.defaultdict(lambda: collections.deque(maxlen=keep))

    def update(self):
        for name in os.listdir(self.dir) if os.path.isdir(self.dir) else []:
            if not name.endswith(".json") or name == "boxes.json" or name.endswith(".cmd.json"):
                continue
            s = read_json(os.path.join(self.dir, name))
            if s and s.get("pose") is not None:
                q = self.states[s.get("robot_id", name[:-5])]
                if not q or q[-1]["t"] != s["t"]:
                    q.append(s)

    def pooled(self, window_s, stale_s):
        """{robot: its last state, with the detections of the last window_s seconds grouped by position and,
        per group, the one with the median distance (as static_trial.capture_trial does with a stop)}."""
        now = time.time()
        out = {}
        for q, states in self.states.items():
            if not states or now - states[-1]["t"] > stale_s:
                continue
            last = states[-1]
            dets = [dict(d, pose=st["pose"]) for st in states if last["t"] - st["t"] <= window_s
                    for d in st.get("detections", [])
                    if d.get("distance") is not None and d.get("bearing_rad") is not None and d.get("position")]
            groups = []
            for d in dets:
                for g in groups:
                    cx = sum(e["position"][0] for e in g) / len(g)
                    cy = sum(e["position"][1] for e in g) / len(g)
                    if math.hypot(d["position"][0] - cx, d["position"][1] - cy) <= self.GROUP_M:
                        g.append(d)
                        break
                else:
                    groups.append([d])
            objects = []
            for g in groups:
                median = float(np.median([e["distance"] for e in g]))
                objects.append(min(g, key=lambda e: abs(e["distance"] - median)))
            out[q] = dict(last, detections=objects)
        return out


def write_round(run_dir, robots, truth):
    """The robots' last detections as round 0 of a run (the format of the captures), for the readers of the
    offline evaluation. A robot that sees nothing still writes its pose: "I looked and saw nothing" counts."""
    os.makedirs(run_dir, exist_ok=True)
    for old in os.listdir(run_dir):
        if old.endswith(".jsonl"):
            os.remove(os.path.join(run_dir, old))
    for q, s in robots.items():
        base = dict(trial_id=f"{LAYOUT}/{q}/000/+0", layout_id=LAYOUT, run_id=LAYOUT, robot_id=q, stop=0,
                    yaw_offset_deg=0.0, radius_m=None, angle_deg=None, round=0, t=s["t"], robot_pose=s["pose"])
        rows = [dict(base, robot_pose=d.get("pose", s["pose"]), class_name=d["class_name"],
                     confidence=d.get("confidence", 1.0), distance=d["distance"], bearing_rad=d["bearing_rad"],
                     viewing_angle_rad=d.get("viewing_angle_rad"), position=d.get("position"))
                for d in s.get("detections", []) if d.get("distance") is not None and d.get("bearing_rad") is not None]
        rows = rows or [dict(base, class_name=None, distance=None, bearing_rad=None)]
        with open(os.path.join(run_dir, f"{q}.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    boxes = truth if truth else [[99.0, 99.0, 0.0]]     # the GNN reader needs a truth; far away = no label
    with open(os.path.join(run_dir, "ground_truth.json"), "w") as f:
        json.dump({LAYOUT: [dict(class_name="box", x=b[0], y=b[1], yaw=b[2]) for b in boxes]}, f)


# ====================================================================
# Methods on the detections
# ====================================================================

class Classical:
    """MHT and MHT with sensor model, with the settings of pipeline/evaluate_runs.py --sensor-calib."""

    def __init__(self, sensor_calib, sensor_model, regions):
        from pipeline import analyze_rounds, evaluate_runs
        from vf.fusion import sensor_aware_mht
        from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_regions, sensor_from_arguments,
                                            zones_from_arguments)
        ap = argparse.ArgumentParser()
        add_sensor_arguments(ap)
        a = ap.parse_args([])
        a.sensor_calib, a.range_fit, a.bearing_fit, a.camera_offset, a.union_threshold = sensor_calib, None, None, 0.0, None
        evaluate_runs.apply_sensor_calib(a)
        sensor_aware_mht.set_model(sensor_model)
        analyze_rounds.BEARING_FIT = tuple(a.bearing_fit) if a.bearing_fit else None
        analyze_rounds.CAMERA_OFFSET = a.camera_offset
        self.range_fit = a.range_fit
        arena = arena_from_regions(regions)
        self.sensor = sensor_from_arguments(a, arena)
        self.zones = zones_from_arguments(a, arena)

    def run(self, run_dir):
        """({method: [[x, y], ...]}, detections [[x, y, robot], ...]) of the round written in run_dir."""
        from pipeline.analyze_rounds import fuse
        from pipeline.build_scans import load_points
        pts = load_points(run_dir, self.range_fit)
        members = sorted(k for k, v in pts.items() if v["round"] == 0)
        out = {m: [] for m in CLASSICAL}
        detections = []
        if members:
            _poses, per_class, results = fuse(members, pts, self.sensor, self.zones, list(CLASSICAL))
            for res in results.values():
                for m in CLASSICAL:
                    out[m] += [[float(p[0]), float(p[1])] for p in res.get(m, {}).get("objects", [])]
            for reports in per_class.values():
                for key, zs in reports.items():
                    detections += [[float(z[0]), float(z[1]), pts[key]["robot"]] for z in zs]
        return out, detections


class Gnn:
    """The graph network on the detections (first training seed of a variant folder of gnn/train_sim.py)."""

    def __init__(self, model_dir, sensor_calib):
        import torch
        from gnn.model import build_model
        with open(os.path.join(model_dir, "variant.json")) as f:
            self.a = types.SimpleNamespace(**json.load(f))
        self.calib = sensor_calib
        self.model = build_model(self.a.graph, self.a.hidden, self.a.hops, self.a.features, self.a.skip)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model_seed0.pt"), map_location="cpu"))
        self.model.eval()

    def run(self, run_dir):
        from gnn.dataset import build_scenes
        from gnn.model import decode, encode_scene
        from gnn.train_sim import predict
        scenes = build_scenes(run_dir, os.path.join(run_dir, "ground_truth.json"), self.calib)
        if not scenes:                                   # no robot detected anything
            return []
        s, a = scenes[0], self.a
        enc = encode_scene(s, features=a.features, edges=a.edges, comm_range=a.comm_range, gate=a.gate,
                           association=a.association, hops=a.hops)
        p_same, p_false = predict(self.model, [enc])[0]
        est, _assign = decode(s["pos"], enc["pairs"], p_same, p_false)
        return [[float(p[0]), float(p[1])] for p in est]


# ====================================================================
# Image networks
# ====================================================================

class CameraCalibration:
    """Pitch and yaw offset of every camera, fitted continuously on the live markers, as bevnet/real.py does
    on a run: for each detected marker near a true box, the angle between where it is seen and where the box
    puts it; the median of the last `keep` of them per robot. Fixed values can be loaded instead (load)."""

    def __init__(self, keep=300):
        self.offsets = collections.defaultdict(lambda: collections.deque(maxlen=keep))
        self.fixed = None

    def load(self, path):
        with open(path) as f:
            self.fixed = {q: tuple(v) for q, v in json.load(f).items()}

    def save(self, path):
        with open(path, "w") as f:
            json.dump({q: [round(v, 3) for v in self.get(q)] for q in sorted(self.offsets)}, f, indent=2)

    def get(self, q):
        """(pitch [deg], yaw offset [deg]) of robot q."""
        from bevnet import real
        if self.fixed and q in self.fixed:
            return self.fixed[q]
        dv = [v for v, _ in self.offsets[q]]
        du = [u for _, u in self.offsets[q]]
        if len(dv) < 5:
            return real.PITCH_DEG.get(q, 11.5), 0.0
        return real.PITCH_DEG.get(q, 11.5) - float(np.median(dv)), float(np.median(du))

    def add(self, q, state, truth):
        """The markers of the last saved frame of robot q, against the true boxes."""
        import cv2
        from bevnet import geometry as geo
        from bevnet import real
        if self.fixed or not truth or state.get("camera_matrix") is None or state.get("frame_pose") is None:
            return
        K = np.array(state["camera_matrix"], float)
        D = np.array(state.get("dist_coeffs") or [0, 0, 0, 0, 0], float)
        x, y, yaw = real.camera_pose(dict(camera_pose=state["frame_pose"]))
        Rm = geo.camera_rotation(yaw, math.radians(real.PITCH_DEG.get(q, 11.5)), 0.0)
        boxes = np.array([b[:2] for b in truth], float)
        for det in state.get("frame_detections", []):
            if det.get("position") is None or det.get("bbox") is None:
                continue
            p = np.asarray(det["position"], float)
            k = int(np.argmin(np.hypot(*(boxes - p).T)))
            if np.hypot(*(boxes[k] - p)) > 0.3:
                continue
            u = (boxes[k] - [x, y]) / np.linalg.norm(boxes[k] - [x, y])
            face = boxes[k] - real.APOTHEM_M * u
            pc = Rm @ (np.array([face[0], face[1], real.MARKER_CENTRE_M]) - [x, y, real.CAMERA_HEIGHT_M])
            bx, by, bw, bh = det["bbox"]
            obs = cv2.undistortPoints(np.array([[[bx + bw / 2, by + bh / 2]]], np.float64), K, D).reshape(2)
            self.offsets[q].append((math.degrees(math.atan(obs[1]) - math.atan(pc[1] / pc[2])),
                                    math.degrees(math.atan(obs[0]) - math.atan(pc[0] / pc[2]))))


class ImageNets:
    """The image network on masks alone ('image') and on masks + detected markers ('image_mk'). The frames
    of the robots are segmented once per update and both networks read the same masks."""

    SCALE = 0.2                                          # network images: 256 x 144

    def __init__(self, models, work_dir, device="auto"):
        import torch
        from bevnet import geometry as geo
        from bevnet.model import BevNet
        from bevnet.segment import BoxSegmenter
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.seg = BoxSegmenter(device=device)
        self.work = work_dir
        self.models = {}
        for key, folder in models.items():
            with open(os.path.join(folder, "variant.json")) as f:
                v = json.load(f)
            net = BevNet(v["channels"], v["message"], len(geo.HEIGHTS_M), v.get("encoder", "small"),
                         pretrained=False).to(self.device)
            net.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location=self.device))
            net.eval()
            self.models[key] = (net, float(v["threshold"]))
        self.calibration = CameraCalibration()

    def frames(self, live_dir, robots):
        """{robot: (image, state)} with the last saved frame of every robot and the state written with it
        (read twice: a frame replaced in between is read again)."""
        import cv2
        out = {}
        for q, s in robots.items():
            for _ in range(3):
                if not s.get("frame") or s.get("frame_pose") is None or s.get("camera_matrix") is None:
                    break
                data = read_bytes(os.path.join(live_dir, s["frame"]))
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
                again = read_json(os.path.join(live_dir, f"{q}.json"))
                if img is not None and again and again.get("frame_t") == s.get("frame_t"):
                    out[q] = (img, s)
                    break
                s = again or s
        return out

    def run(self, live_dir, robots, truth):
        """({key: [[x, y], ...]}, {robot: overlay file}, {robot: [pitch, yaw offset]})."""
        import cv2
        from bevnet import geometry as geo
        from bevnet import real
        from bevnet.data import BevDataset
        from bevnet.train import loader, predict_iter
        frames = self.frames(live_dir, robots)
        if not frames:
            return {k: [] for k in self.models}, {}, {}
        names = sorted(frames)
        for q in names:
            self.calibration.add(q, frames[q][1], truth)
        poses, pitch, Ks, Ds, small, mk = [], [], [], [], {}, {}
        overlays = {}
        for q in names:
            img, s = frames[q]
            H, W = img.shape[:2]
            w, h = int(round(1280 * self.SCALE)), int(round(720 * self.SCALE))
            mask, _rects, _labels = self.seg(img)
            small[q] = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
            mk[q] = np.zeros_like(small[q])
            for det in s.get("frame_detections", []):
                if det.get("bbox") is None:
                    continue
                bx, by, bw, bh = det["bbox"]
                cv2.rectangle(mk[q], (int(round(bx * w / W)), int(round(by * h / H))),
                              (int(round((bx + bw) * w / W)), int(round((by + bh) * h / H))), 255, -1)
            p_deg, yaw_off = self.calibration.get(q)
            x, y, yaw = real.camera_pose(dict(camera_pose=s["frame_pose"]))
            poses.append([x, y, yaw + math.radians(yaw_off)])
            pitch.append(p_deg)
            K = np.array(s["camera_matrix"], float)
            K[0] *= w / W
            K[1] *= h / H
            Ks.append(K.tolist())
            Ds.append(list(s.get("dist_coeffs") or [0, 0, 0, 0, 0]))
            overlays[q] = self.overlay(q, img, mask, s)
        poses = np.array(poses, float)
        centre = poses[:, :2].mean(0)
        poses[:, :2] -= centre
        boxes = (np.array([b[:2] for b in truth], np.float32) - centre) if truth else np.zeros((0, 2), np.float32)
        out = {}
        for key, (net, threshold) in self.models.items():
            markers = key == "image_mk"
            jpgs = [cv2.imencode(".png", np.dstack([small[q], mk[q], small[q]]) if markers else small[q])[1].tobytes()
                    for q in names]
            scene = dict(run=LAYOUT, family="real", boxes=boxes.astype(np.float32), poses=poses.astype(np.float32),
                         pitch=np.radians(pitch).astype(np.float32), roll=np.zeros(len(names), np.float32),
                         height=real.CAMERA_HEIGHT_M, camera="real", jpgs=jpgs, n_robots=len(names),
                         n_boxes=len(boxes), robots=names, K=Ks, D=Ds, centre=centre.tolist())
            folder = os.path.join(self.work, key)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "shard_0000.pkl"), "wb") as f:
                pickle.dump([scene], f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(folder, "meta.json"), "w") as f:
                json.dump(dict(set=key, scenario="live", scenes=1, camera="real", K=Ks[0], D=Ds[0],
                               size=[256, 144], scale=self.SCALE, marker_channel=markers), f)
            heat, off, *_ = next(predict_iter(net, loader(BevDataset([folder]), 1, False, 0), self.device))
            out[key] = [[float(x + centre[0]), float(y + centre[1])] for x, y in geo.decode(heat, off, threshold)]
        return out, overlays, {q: list(self.calibration.get(q)) for q in names}

    def overlay(self, q, img, mask, state):
        """The frame with the box mask tinted green and the detected markers outlined in red (for the page)."""
        import cv2
        tint = img.copy()
        tint[mask > 127] = (60, 200, 60)
        view = cv2.addWeighted(img, 0.55, tint, 0.45, 0)
        for det in state.get("frame_detections", []):
            if det.get("bbox"):
                bx, by, bw, bh = [int(round(v)) for v in det["bbox"]]
                cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (40, 40, 230), 4)
        folder = os.path.join(self.work, "frames")
        os.makedirs(folder, exist_ok=True)
        name = f"{q}.jpg"
        tmp = os.path.join(folder, name + ".tmp")
        cv2.imwrite(tmp + ".jpg", cv2.resize(view, (640, 360), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        replace(tmp + ".jpg", os.path.join(folder, name))
        return name


# ====================================================================
# Verdict
# ====================================================================

def judge(est, truth, cutoff=0.15):
    """Scene verdict as in the evaluation (pipeline/evaluate_runs.scene_metrics) plus, for drawing, which
    estimates and which true boxes are matched (one to one, nearest pairs first, within the cut-off)."""
    from pipeline.evaluate_runs import scene_metrics
    m = scene_metrics([np.asarray(e, float) for e in est], [np.asarray(t, float) for t in truth], cutoff, 2.0)
    pairs = sorted((math.hypot(e[0] - t[0], e[1] - t[1]), i, j) for i, e in enumerate(est) for j, t in enumerate(truth))
    est_ok, truth_ok = [False] * len(est), [False] * len(truth)
    for d, i, j in pairs:
        if d <= cutoff and not est_ok[i] and not truth_ok[j]:
            est_ok[i] = truth_ok[j] = True
    if m["dn"] < 0:
        verdict = "too few"
    elif m["dn"] > 0:
        verdict = "too many"
    else:
        verdict = "correct" if m["n_matched"] == m["n_true"] else "misplaced"
    return dict(verdict=verdict, n_est=m["n_est"], n_true=m["n_true"], n_matched=m["n_matched"],
                rmse_cm=None if math.isnan(m["rmse"]) else round(100 * m["rmse"], 1), est_ok=est_ok, truth_ok=truth_ok)
