"""The real scenes (data/survey_rounds: 8 layouts x 7 rounds = 56 scenes, 4 robots) in the format of
bevnet/data.py, so the image-based network trained in simulation can be tested on the real camera frames.

For every scene (layout, round) and reading k (each robot took 5 frames per stop): the frame of each robot,
resized to the network's image size (--scale of 1280 x 720, as in bevnet/data.py), its camera pose from
OptiTrack (raw/<robot>.jsonl, camera_pose), its own camera matrix and distortion (raw/meta_<layout>_<robot>.json,
scaled to the image size), the downward pitch measured on the real frames (--pitch), and the true boxes
(ground_truth.json). Poses and boxes are shifted so that the centre of the robots is the origin of the floor
grid. One set per reading: data/bev_real/real_k<k>/ (shard_*.pkl, meta.json); run ids <layout>_r<round>.
--calibrate fits, per robot, the camera pitch and a yaw offset from the ArUco detections stored with the
frames (without --marker-channel the detections are used only for this calibration, never as network input), leave one layout
out: the frames of a layout use the values fitted on the other layouts. A low camera (25 cm) makes the floor
geometry very sensitive to the pitch: at 1.5 m, 1 degree moves the middle of a box by about 60 cm along the ray.
--check draws the true box centres and the grid's sampling points into a few frames (fig_REAL-0.png), to see
that poses, pitch and lens put them on the boxes.

    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real/ --readings 0,1,2,3,4 --check
    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real_cal/ --readings 0,1,2,3,4 --calibrate
    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real_masks/ --readings 0,1,2,3,4 --calibrate --masks
      (--masks: the boxes cut out of every frame by bevnet/segment.py, as the boxes-only simulation; the full-size
       masks are cached in <out>/cache/ and an overlay figure is written to <out>/fig_REAL-masks.png)
    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real_cut/ --readings 0,1,2,3,4 --calibrate --cutout \
        --mask-cache data/bev_real_masks/cache
      (--cutout: the frame itself inside the mask, black elsewhere, as bevnet.data --cutout)
    python -m bevnet.real --runs "data/survey_v2/*" --best-run-per-layout --out data/bev_real_v2_masks/ \
        --readings 0,1,2,3,4 --calibrate --masks          # second campaign: one run per layout (R01, R02, ...)
    python -m bevnet.real --runs "data/survey_v2_best/*" --out data/bev_real_v2_mko/ --readings 0,1,2,3,4 --calibrate \
        --calibrate-scope same-run --masks --marker-channel --oracle-markers --mask-cache data/bev_real_v2_masks/cache
      (--oracle-markers: the detected markers plus those of the visible boxes the detector missed)
    python -m bevnet.train --eval-only --out results/bev_obst/ --tests "data/bev_real/*" --results-name real
"""

import argparse
import glob
import json
import math
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np

from bevnet import geometry as geo
from pipeline.build_scans import best_runs

PITCH_DEG = dict(P0=11.5, P1=11.5, P3=11.5, P5=22.0)      # measured on the real frames (imgsim/render.py)
CAMERA_HEIGHT_M = 0.25                  # lens above the floor (measured 24-26 cm)
CAMERA_FORWARD_FIX_M = 0.03             # lens 15 cm ahead of the rigid body; the acquisition moved the pose by 12 cm
MARKER_CENTRE_M = 0.15                  # marker centre above the floor (faces 30 cm high)
BOX_SIDE_M = 0.235                      # face width (measured 23-24 cm)



def camera_pose(rec):
    """(x, y, yaw) of the lens: the recorded camera pose moved forward by CAMERA_FORWARD_FIX_M."""
    x, y, yaw = rec["camera_pose"]
    return (x + CAMERA_FORWARD_FIX_M * math.cos(yaw), y + CAMERA_FORWARD_FIX_M * math.sin(yaw), yaw)


def load_run(run_dir):
    """One run folder of launch_all.py: (layout id, true boxes (B, 2), frames, cameras), with
    frames[(round, reading, yaw offset)][robot] = the raw record of that frame (pose, image path, detections)
    and cameras[robot] = (camera matrix, distortion)."""
    with open(os.path.join(run_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    frames = defaultdict(dict)                     # (round, reading, yaw offset) -> robot -> record
    cams = {}
    layout = None
    for path in sorted(glob.glob(os.path.join(run_dir, "raw", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("round") is None:
                    continue
                layout = r["layout_id"]
                off = int(round(float(r["trial_id"].rsplit("/", 1)[-1])))   # --yaw-offsets-deg captures
                frames[(int(r["round"]), int(r["reading"]), off)][r["robot_id"]] = r
    for path in glob.glob(os.path.join(run_dir, "raw", "meta_*.json")):
        with open(path) as f:
            m = json.load(f)
        cams[m["robot_id"]] = (np.array(m["camera_matrix"], float), np.array(m["dist_coeffs"], float).ravel())
    key = layout if layout in gt else os.path.basename(os.path.normpath(run_dir))
    boxes = np.array([[b["x"], b["y"]] for b in gt[key]], np.float32)
    return layout, boxes, frames, cams


APOTHEM_M = BOX_SIDE_M / (2 * math.sqrt(3))


def angle_offsets(runs, pitch):
    """Per (layout, robot): median offsets [deg] between where the marker centres are seen (undistorted) and
    where the true box puts them (the face towards the camera, at the marker centre height), vertical and horizontal."""
    out = defaultdict(lambda: ([], []))
    for d, layout, boxes, frames, cams in runs:
        for recs in frames.values():
            for q, r in recs.items():
                K, D = cams[q]
                x, y, yaw = camera_pose(r)
                Rm = geo.camera_rotation(yaw, math.radians(pitch.get(q, 11.5)), 0.0)
                for det in r.get("detections", []):
                    p = np.asarray(det["position"], float)
                    k = int(np.argmin(np.hypot(*(boxes - p).T)))
                    if np.hypot(*(boxes[k] - p)) > 0.3:
                        continue
                    u = (boxes[k] - [x, y]) / np.linalg.norm(boxes[k] - [x, y])
                    face = boxes[k] - APOTHEM_M * u
                    pc = Rm @ (np.array([face[0], face[1], MARKER_CENTRE_M]) - [x, y, CAMERA_HEIGHT_M])
                    bx, by, bw, bh = det["bbox"]
                    obs = cv2.undistortPoints(np.array([[[bx + bw / 2, by + bh / 2]]], np.float64), K, D).reshape(2)
                    out[(layout, q)][0].append(math.degrees(math.atan(obs[1]) - math.atan(pc[1] / pc[2])))
                    out[(layout, q)][1].append(math.degrees(math.atan(obs[0]) - math.atan(pc[0] / pc[2])))
    return out


def calibration(runs, pitch, scope="other-layouts", neighbours=2):
    """(layout, robot) -> (pitch [deg], yaw offset [deg]) of each camera, from the marker detections.
    scope "other-layouts": fitted on all the other layouts (the camera mounting is the same throughout).
    scope "neighbours": fitted on the `neighbours` layouts captured just before and just after (capture
    order = order of the run folders), not on the layout itself: for a campaign in which the cameras were
    moved now and then. scope "same-run": fitted on the layout itself (the camera calibrated in the
    session it is used in; the detections only set the camera angles, never the boxes)."""
    off = angle_offsets(runs, pitch)
    order = [layout for _, layout, *_ in runs]
    cal = {}
    for (layout, q) in off:
        if scope == "same-run":
            use = {layout}
        elif scope == "neighbours":
            k = order.index(layout)
            use = set(order[max(0, k - neighbours):k] + order[k + 1:k + 1 + neighbours])
        else:
            use = set(order) - {layout}
        dv = [v for (l2, q2), (vs, _) in off.items() if q2 == q and l2 in use for v in vs]
        du = [v for (l2, q2), (_, us) in off.items() if q2 == q and l2 in use for v in us]
        if dv:
            cal[(layout, q)] = (pitch.get(q, 11.5) - float(np.median(dv)), float(np.median(du)))
    return cal


MARKER_SIDE_M = 0.077                   # ArUco marker side


def true_markers(boxes, pose, pitch, cam, size, mask):
    """The marker a perfect detector would find on each true box in one frame: the centre of the face that looks
    at the camera (the box centre moved by the apothem towards the camera, at the marker height), projected with
    the camera pose, pitch and lens. A marker is kept if it falls inside the image, on the segmentation mask
    (the box is not hidden by a dark object) and is not behind another box. Returns (box index, full-size
    rectangle (u0, v0, u1, v1)), square, of the marker's apparent size."""
    K, D = cam
    x, y = pose[0], pose[1]
    out = []
    for k, b in enumerate(boxes):
        d = np.hypot(b[0] - x, b[1] - y)
        if d < 1e-6:
            continue
        u = (b - [x, y]) / d
        face = b - APOTHEM_M * u
        hidden = False
        for j, o in enumerate(boxes):              # another box closer to the camera, across the line of sight
            if j == k:
                continue
            t = float(np.dot(o - [x, y], u))
            across = abs(u[0] * (o[1] - y) - u[1] * (o[0] - x))
            if 0 < t < d - APOTHEM_M and across < 0.8 * BOX_SIDE_M / math.sqrt(3):
                hidden = True
        if hidden:
            continue
        uv, ok = geo.project(np.array([[face[0], face[1], MARKER_CENTRE_M]]), pose, pitch, 0.0, CAMERA_HEIGHT_M,
                             K, D, size)
        if not ok[0]:
            continue
        uc, vc = uv[0]
        if mask is not None:
            mh, mw = mask.shape[:2]
            if mask[min(int(vc * mh / size[1]), mh - 1), min(int(uc * mw / size[0]), mw - 1)] < 128:
                continue
        half = 0.5 * K[0, 0] * MARKER_SIDE_M / max(d - APOTHEM_M, 0.1)
        out.append((k, (uc - half, vc - half, uc + half, vc + half)))
    return out


def build_scene(run_dir, layout, boxes, recs, cams, scale, quality, pitch, cal=None, seg=None, cache=None,
                overlays=None, cutout=False, instances=False, marker_channel=False, oracle_markers=False):
    """One scene (the frames `recs` of the robots at one stop) in the format of bevnet/data.py: the images
    (frames, masks, cut-outs or masks + marker channel), the camera poses relative to the centre of the robots,
    the pitch (calibrated with `cal`), each camera's matrix and distortion, and the true boxes. With `seg`
    the frames are segmented (masks read from / written to `cache`)."""
    robots = sorted(recs)
    poses = np.array([camera_pose(recs[q]) for q in robots], np.float64)
    pitch_deg = [pitch.get(q, 11.5) for q in robots]
    if cal:
        for n, q in enumerate(robots):
            if (layout, q) in cal:
                pitch_deg[n] = cal[(layout, q)][0]
                poses[n, 2] += math.radians(cal[(layout, q)][1])
    world = poses.copy()                                    # camera poses in the OptiTrack frame
    centre = poses[:, :2].mean(0)
    poses[:, :2] -= centre
    jpgs, Ks, Ds = [], [], []
    for n, q in enumerate(robots):
        img = cv2.imread(os.path.join(run_dir, recs[q]["image"].replace("\\", "/")))
        if img is None:
            return None
        H, W = img.shape[:2]
        w, h = int(round(1280 * scale)), int(round(720 * scale))
        if seg is not None:                                 # boxes only
            key = os.path.join(cache, os.path.basename(os.path.normpath(run_dir)),
                               os.path.basename(recs[q]["image"].replace("\\", "/"))[:-4] + ".png")
            key_ids = key[:-4] + "_ids.png"                 # one label per box (--instances)
            mask = cv2.imread(key, cv2.IMREAD_GRAYSCALE) if os.path.exists(key) else None
            labels = cv2.imread(key_ids, cv2.IMREAD_GRAYSCALE) if os.path.exists(key_ids) else None
            if mask is None or (instances and labels is None):
                mask, rects, labels = seg(img)
                os.makedirs(os.path.dirname(key), exist_ok=True)
                cv2.imwrite(key, mask)
                cv2.imwrite(key_ids, labels)
                if overlays is not None and len(overlays) < 16:
                    from bevnet.segment import overlay
                    overlays.append(cv2.resize(overlay(img, mask, rects, labels if instances else None),
                                               (640, 360)))
            if instances:                                   # boxes that touch in the image stay apart
                from imgsim.render import separate_instances
                small = separate_instances(cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST))
            else:
                small = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
            if cutout:                                      # the frame inside the mask, black elsewhere
                frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                frame[small == 0] = 0
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif marker_channel:                            # mask, markers, mask (B, G, R), as imgsim
                mk = np.zeros_like(small)
                dets = recs[q].get("detections", [])
                for det in dets:                            # the ArUco markers detected in this frame
                    bx, by, bw, bh = det["bbox"]
                    cv2.rectangle(mk, (int(round(bx * w / W)), int(round(by * h / H))),
                                  (int(round((bx + bw) * w / W)), int(round((by + bh) * h / H))), 255, -1)
                if oracle_markers:                          # plus the visible boxes the detector missed
                    found = {int(np.argmin(np.hypot(*(boxes - np.asarray(dt["position"], float)).T)))
                             for dt in dets if "position" in dt
                             and np.min(np.hypot(*(boxes - np.asarray(dt["position"], float)).T)) <= 0.3}
                    for k, (u0, v0, u1, v1) in true_markers(boxes, world[n], math.radians(pitch_deg[n]), cams[q],
                                                            (W, H), mask):
                        if k not in found:
                            cv2.rectangle(mk, (int(round(u0 * w / W)), int(round(v0 * h / H))),
                                          (int(round(u1 * w / W)), int(round(v1 * h / H))), 255, -1)
                ok, buf = cv2.imencode(".png", np.dstack([small, mk, small]))
            else:
                ok, buf = cv2.imencode(".png", small)
        else:
            small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpgs.append(buf.tobytes())
        K, D = cams[q]
        K = K.copy()
        K[0] *= w / W
        K[1] *= h / H
        Ks.append(K.tolist())
        Ds.append(D.tolist())
    rnd = int(recs[robots[0]]["round"])
    return dict(run=f"{layout}_r{rnd}", family="real", boxes=(boxes - centre).astype(np.float32),
                poses=poses.astype(np.float32), pitch=np.radians(pitch_deg).astype(np.float32),
                roll=np.zeros(len(robots), np.float32), height=CAMERA_HEIGHT_M, camera="real", jpgs=jpgs,
                n_robots=len(robots), n_boxes=len(boxes), robots=robots, K=Ks, D=Ds, centre=centre.tolist())


def check_figure(scenes, out_path, n=4):
    """True box centres (green, at the heights of bevnet/geometry.py) and the grid cells within 15 cm of them (red dots)."""
    cells = geo.cell_centres()
    tiles = []
    for s in scenes[:n]:
        row = []
        for k, jpg in enumerate(s["jpgs"]):
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            K, D = np.array(s["K"][k]), np.array(s["D"][k])
            size = (img.shape[1] // 2, img.shape[0] // 2)
            for b in s["boxes"]:
                near = np.hypot(cells[..., 0] - b[0], cells[..., 1] - b[1]) < 0.15
                P = np.concatenate([cells[near], np.full((near.sum(), 1), MARKER_CENTRE_M)], -1)
                uv, ok = geo.project(P, s["poses"][k], float(s["pitch"][k]), 0.0, s["height"], K, D, size)
                for (u, v), o in zip(uv, ok):
                    if o:
                        cv2.circle(img, (int(2 * u), int(2 * v)), 1, (0, 0, 255), -1)
                for hgt in geo.HEIGHTS_M:
                    uv, ok = geo.project(np.array([[b[0], b[1], hgt]]), s["poses"][k], float(s["pitch"][k]), 0.0,
                                         s["height"], K, D, size)
                    if ok[0]:
                        cv2.circle(img, (int(2 * uv[0, 0]), int(2 * uv[0, 1])), 4, (0, 220, 0), -1)
            cv2.putText(img, f"{s['run']} {s['robots'][k]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            row.append(img)
        tiles.append(np.hstack(row))
    cv2.imwrite(out_path, np.vstack(tiles))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["data/survey_rounds/*"])
    ap.add_argument("--out", default="data/bev_real/")
    ap.add_argument("--readings", default="0", help="comma list of readings (0-4): one set each")
    ap.add_argument("--best-run-per-layout", action="store_true",
                    help="several runs of one layout (repeated captures, data/survey_v2): keep one per layout, "
                         "the complete run whose folder name ends in F, otherwise the latest complete one")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--pitch", default=None, help="per robot pitch in degrees, e.g. P0=11,P1=12,P3=11,P5=22")
    ap.add_argument("--check", action="store_true", help="write fig_REAL-0.png with the true boxes drawn in")
    ap.add_argument("--masks", action="store_true", help="boxes only: segment every frame (bevnet/segment.py)")
    ap.add_argument("--instances", action="store_true",
                    help="with --masks: a 1-pixel black line where two segmented boxes touch (as imgsim "
                         "instance_borders); needs the per-box labels, so frames cached without them are "
                         "segmented again")
    ap.add_argument("--marker-channel", action="store_true",
                    help="with --masks: second image channel = the ArUco markers detected in the frame (filled "
                         "rectangles), as imgsim marker_channel")
    ap.add_argument("--oracle-markers", action="store_true",
                    help="with --marker-channel: the detected markers plus, for every visible box the detector "
                         "missed in that frame, the marker a perfect detector would find (true_markers): what the "
                         "network does when no visible marker is missed")
    ap.add_argument("--cutout", action="store_true",
                    help="boxes only, as seen: the frame inside the segmentation mask, black elsewhere")
    ap.add_argument("--mask-cache", default=None,
                    help="folder of the segmentation masks (default <out>/cache; reuse the one of a --masks run)")
    ap.add_argument("--seg-device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--seg-threshold", type=float, default=0.30, help="detector box threshold")
    ap.add_argument("--seg-max-area", type=float, default=0.70,
                    help="largest detector rectangle kept, as a share of the frame (near boxes fill much of it)")
    ap.add_argument("--calibrate", action="store_true",
                    help="pitch and yaw offset per robot from the stored detections, leave one layout out")
    ap.add_argument("--calibrate-scope", choices=["other-layouts", "neighbours", "same-run"],
                    default="other-layouts",
                    help="which detections calibrate the camera of a layout (see calibration()): the other "
                         "layouts, the 2 layouts captured before and after, or the layout itself")
    a = ap.parse_args()
    pitch = dict(PITCH_DEG)
    if a.pitch:
        pitch.update({k: float(v) for k, v in (p.split("=") for p in a.pitch.split(","))})
    run_dirs = sorted({d for p in a.runs for d in glob.glob(p) if os.path.isdir(os.path.join(d, "raw"))})
    if a.best_run_per_layout:
        run_dirs = best_runs(run_dirs)
        print(f"  {len(run_dirs)} layouts: " + " ".join(os.path.basename(os.path.normpath(d)) for d in run_dirs),
              flush=True)
    runs = [(d, *load_run(d)) for d in run_dirs]
    cal = calibration(runs, pitch, a.calibrate_scope) if a.calibrate else None
    seg, overlays, cache = None, [], a.mask_cache or os.path.join(a.out, "cache")
    if a.masks or a.cutout:
        from bevnet.segment import BoxSegmenter
        seg = BoxSegmenter(device=a.seg_device, box_threshold=a.seg_threshold, max_area=a.seg_max_area)
    if cal:
        for q in sorted({q for _, q in cal}):
            v = np.array([cal[k] for k in cal if k[1] == q])
            print(f"  {q}: pitch {v[:, 0].mean():.2f} deg (assumed {pitch.get(q, 11.5):.1f}), "
                  f"yaw offset {v[:, 1].mean():+.2f} deg, over the {len(v)} leave-one-out fits")
    for k in [int(v) for v in a.readings.split(",")]:
        scenes = []
        for d, layout, boxes, frames, cams in runs:
            for (rnd, reading, off), recs in sorted(frames.items()):
                if reading == k:
                    s = build_scene(d, layout, boxes, recs, cams, a.scale, a.jpeg_quality, pitch, cal, seg, cache,
                                    overlays, a.cutout, a.instances, a.marker_channel,
                                    a.oracle_markers)
                    if s is not None:
                        s["round"] = rnd
                        if off:                     # the cameras turned by `off` degrees at every stop
                            s["run"] += f"_y{off:+d}"
                        scenes.append(s)
        name = f"real_k{k}"
        out = os.path.join(a.out, name)
        os.makedirs(out, exist_ok=True)
        for old in glob.glob(os.path.join(out, "shard_*.pkl")):
            os.replace(old, old + ".old")
        from bevnet.data import SHARD
        for j, start in enumerate(range(0, len(scenes), SHARD)):   # the loader reads SHARD scenes per file
            with open(os.path.join(out, f"shard_{j:04d}.pkl"), "wb") as f:
                pickle.dump(scenes[start:start + SHARD], f, protocol=pickle.HIGHEST_PROTOCOL)
        K0 = scenes[0]["K"][0]
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(dict(set=name, scenario="real", level=f"k{k}", scenes=len(scenes), camera="real", K=K0,
                           marker_channel=bool(a.marker_channel and a.masks),
                           marker_source="detections + missed visible" if a.oracle_markers else "detections",
                           D=scenes[0]["D"][0], size=[int(round(1280 * a.scale)), int(round(720 * a.scale))],
                           scale=a.scale, pitch_deg=pitch, runs=[os.path.basename(r[0]) for r in runs],
                           calibration={f"{l}/{q}": v for (l, q), v in cal.items()} if cal else None), f, indent=2)
        print(f"  {name}: {len(scenes)} scenes, {sum(s['n_robots'] for s in scenes)} images -> {out}", flush=True)
        if overlays and k == int(a.readings.split(",")[0]):
            rows = [np.hstack(overlays[r:r + 4]) for r in range(0, len(overlays) - len(overlays) % 4, 4)]
            if rows:
                cv2.imwrite(os.path.join(a.out, "fig_REAL-masks.png"), np.vstack(rows))
                print(f"  mask figure: {os.path.join(a.out, 'fig_REAL-masks.png')}")
        if a.check and k == int(a.readings.split(",")[0]):
            check_figure(scenes, os.path.join(a.out, "fig_REAL-0.png"))
            print(f"  check figure: {os.path.join(a.out, 'fig_REAL-0.png')}")


if __name__ == "__main__":
    main()
