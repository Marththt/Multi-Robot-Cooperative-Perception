"""Figures of the image-network report: what the pretrained segmentation extracts from the real frames of one
scene, what the network receives, and the map of box centres it returns, on masks alone and on masks + markers.

For every --scene LAYOUT:ROUND (the run kept for that layout in --runs, first reading of the stop):
  <out>/<layout>_r<round>_frames.png   top: the four frames, brightened for display, with the rectangles found by
                                       Grounding DINO (red), the SAM mask (green) and the detected ArUco markers
                                       (yellow); bottom: the 256 x 144 input of each robot (mask white, markers red)
  <out>/<layout>_r<round>_maps.png     the floor map of box-centre probability of the network on masks alone and
                                       of the network on masks + markers, with the true boxes (OptiTrack), the
                                       boxes found and the four robots (their maps, boxes and robots are kept in
                                       <layout>_r<round>_maps.npz)
Runs in the environment of the demo (torch, transformers, a GPU helps):

    python -m tools.fig_image_report --scene R81:0 --scene R96:0 --out results/figures/image_report/
"""

import argparse
import glob
import json
import math
import os
import pickle
import shutil

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from bevnet import geometry as geo
from bevnet import real
from pipeline.build_scans import best_runs
from tools.draw_robot import draw_robot

SCALE = 0.2


def brighten(img, gain=2.2):
    """The laboratory frames are dark: a gamma lift for display only (the network never sees the frame)."""
    lut = np.clip(255 * (np.arange(256) / 255) ** (1 / gain), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def frame_overlay(img, mask, rects, dets):
    """The frame brightened, the mask tinted green with its outline, DINO rectangles red, markers yellow."""
    view = brighten(img)
    tint = view.copy()
    tint[mask > 127] = (80, 200, 80)
    view = cv2.addWeighted(view, 0.65, tint, 0.35, 0)
    cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, cnts, -1, (60, 230, 60), 3)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(view, (int(x0), int(y0)), (int(x1), int(y1)), (40, 40, 235), 3)
    for d in dets:
        if d.get("bbox"):
            bx, by, bw, bh = [int(round(v)) for v in d["bbox"]]
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (0, 220, 255), 4)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def network_input(mask, dets, W, H):
    """What the network receives: the mask at 256 x 144 (white) and the marker rectangles (red)."""
    w, h = int(round(1280 * SCALE)), int(round(720 * SCALE))
    small = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
    mk = np.zeros_like(small)
    for d in dets:
        if d.get("bbox"):
            bx, by, bw, bh = d["bbox"]
            cv2.rectangle(mk, (int(round(bx * w / W)), int(round(by * h / H))),
                          (int(round((bx + bw) * w / W)), int(round((by + bh) * h / H))), 255, -1)
    rgb = np.dstack([small, small, small]).astype(np.uint8)
    rgb[mk > 0] = (230, 50, 50)
    return rgb


def predict(net, threshold, scene, folder, device, markers):
    """(probability map, estimates (K, 2) in the frame of the scene) of one scene."""
    from bevnet.data import BevDataset
    from bevnet.train import loader, predict_iter
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    with open(os.path.join(folder, "shard_0000.pkl"), "wb") as f:
        pickle.dump([scene], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(dict(set=os.path.basename(folder), scenario="real", scenes=1, camera="real", K=scene["K"][0],
                       D=scene["D"][0], size=[256, 144], scale=SCALE, marker_channel=markers), f)
    heat, off, *_ = next(predict_iter(net, loader(BevDataset([folder]), 1, False, 0), device))
    return heat, geo.decode(heat, off, threshold)


def match(est, boxes, cutoff=0.15):
    pairs = sorted((math.hypot(e[0] - b[0], e[1] - b[1]), i, j) for i, e in enumerate(est) for j, b in enumerate(boxes))
    e_ok, b_ok = [False] * len(est), [False] * len(boxes)
    for d, i, j in pairs:
        if d <= cutoff and not e_ok[i] and not b_ok[j]:
            e_ok[i] = b_ok[j] = True
    return e_ok, b_ok


def verdict(est, boxes):
    e_ok, b_ok = match(est, boxes)
    if len(est) < len(boxes):
        return f"{len(est)} found, {len(boxes) - len(est)} missing"
    if len(est) > len(boxes):
        return f"{len(est)} found, {len(est) - len(boxes)} too many"
    return "all correct" if all(b_ok) else "right count, one misplaced"


def draw_map(ax, heat, est, boxes, yaws, cams, title, half):
    G = heat.shape[0]
    ext = [-geo.GRID_HALF_M, geo.GRID_HALF_M, -geo.GRID_HALF_M, geo.GRID_HALF_M]
    ax.imshow(heat, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="bilinear")
    r = real.BOX_SIDE_M / math.sqrt(3)
    e_ok, b_ok = match(est, boxes)
    for (bx, by), yaw, ok in zip(boxes, yaws, b_ok):
        pts = [(bx + r * math.cos(yaw + k * 2 * math.pi / 3), by + r * math.sin(yaw + k * 2 * math.pi / 3))
               for k in range(3)]
        ax.add_patch(Polygon(pts, closed=True, fill=False, ec="#111" if ok else "#c0392b", lw=1.8,
                             ls="-" if ok else "--"))
    for (x, y), ok in zip(est, e_ok):
        if ok:
            ax.plot(x, y, "o", ms=9, mfc="#1e8449", mec="white", mew=1.5)
        else:
            ax.plot(x, y, "X", ms=11, mfc="#c0392b", mec="white", mew=1.2)
    for x, y, yaw in cams:
        draw_robot(ax, x, y, yaw)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", required=True, metavar="LAYOUT:ROUND")
    ap.add_argument("--runs", default="data/survey_v2/*")
    ap.add_argument("--image", default="results/bev_mask_wp/masks_wp")
    ap.add_argument("--image-markers", default="results/bev_mask_mk/masks_mk")
    ap.add_argument("--out", default="results/figures/image_report/")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import torch
    from bevnet.model import BevNet
    from bevnet.segment import BoxSegmenter
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    nets = {}
    for key, folder in (("masks", a.image), ("markers", a.image_markers)):
        with open(os.path.join(folder, "variant.json")) as f:
            v = json.load(f)
        net = BevNet(v["channels"], v["message"], len(geo.HEIGHTS_M), v.get("encoder", "small"),
                     pretrained=False).to(device)
        net.load_state_dict(torch.load(os.path.join(folder, "model.pt"), map_location=device))
        net.eval()
        nets[key] = (net, float(v["threshold"]))
    seg = BoxSegmenter(device=a.device)
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "work")
    runs_all = sorted(d for d in glob.glob(a.runs) if os.path.isdir(os.path.join(d, "raw")))
    for spec in a.scene:
        layout, rnd = spec.split(":")
        rnd = int(rnd)
        cand = [d for d in runs_all if os.path.basename(os.path.normpath(d)).split("_")[0] == layout]
        run_dir = best_runs(cand)[0]
        runs = [(run_dir, *real.load_run(run_dir))]
        _d, lay_id, boxes, frames, cams = runs[0]
        cal = real.calibration(runs, real.PITCH_DEG, "same-run")
        recs = frames[(rnd, 0, 0)]
        with open(os.path.join(run_dir, "ground_truth.json")) as f:
            gt = json.load(f)
        gt_key = lay_id if lay_id in gt else os.path.basename(os.path.normpath(run_dir))   # as real.load_run
        yaws = [b.get("yaw") or 0.0 for b in gt[gt_key]]
        cache = os.path.join(work, "cache")
        tops, bottoms = [], []
        for q in sorted(recs):                                        # segment once, cached for build_scene
            img = cv2.imread(os.path.join(run_dir, recs[q]["image"].replace("\\", "/")))
            mask, rects, labels = seg(img)
            key = os.path.join(cache, os.path.basename(os.path.normpath(run_dir)),
                               os.path.basename(recs[q]["image"].replace("\\", "/"))[:-4] + ".png")
            os.makedirs(os.path.dirname(key), exist_ok=True)
            cv2.imwrite(key, mask)
            cv2.imwrite(key[:-4] + "_ids.png", labels)
            dets = recs[q].get("detections", [])
            tops.append((q, frame_overlay(img, mask, rects, dets)))
            bottoms.append(network_input(mask, dets, img.shape[1], img.shape[0]))
        fig, axes = plt.subplots(2, len(tops), figsize=(4.2 * len(tops), 4.9),
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12, wspace=0.04))
        for k, ((q, top), bottom) in enumerate(zip(tops, bottoms)):
            axes[0, k].imshow(top)
            axes[0, k].set_title(f"robot {q}", fontsize=12)
            axes[1, k].imshow(bottom)
            for ax in (axes[0, k], axes[1, k]):
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_ylabel("camera frame", fontsize=12)
        axes[1, 0].set_ylabel("network input", fontsize=12)
        name = f"{layout}_r{rnd}"
        fig.savefig(os.path.join(a.out, f"{name}_frames.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
        results = {}
        for key, markers in (("masks", False), ("markers", True)):
            scene = real.build_scene(run_dir, lay_id, boxes, recs, cams, SCALE, 92, real.PITCH_DEG, cal, seg,
                                     cache, None, False, False, markers, False)
            net, thr = nets[key]
            heat, est = predict(net, thr, scene, os.path.join(work, key), device, markers)
            results[key] = (heat, est, scene)
        scene = results["masks"][2]
        centre = np.array(scene["centre"])
        bx = np.asarray(boxes, float) - centre
        cams_rel = [(p[0], p[1], p[2]) for p in np.asarray(scene["poses"], float)]
        half = max(1.2, float(np.max(np.abs(np.vstack([bx, np.asarray(cams_rel)[:, :2]])))) + 0.55)
        np.savez_compressed(os.path.join(a.out, f"{name}_maps.npz"), boxes=bx, yaws=np.asarray(yaws),
                            cams=np.asarray(cams_rel), half=half,
                            **{f"{k}_{f}": v for k in results for f, v in (("heat", results[k][0]),
                                                                            ("est", np.asarray(results[k][1])))})
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        for ax, key, label in ((axes[0], "masks", "masks only"), (axes[1], "markers", "masks + markers")):
            heat, est, _s = results[key]
            draw_map(ax, heat, est, bx, yaws, cams_rel, f"{label}: {verdict(est, bx)}", half)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{name}_maps.png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"{name} ({os.path.basename(run_dir)}): " + ", ".join(
            f"{k}: {verdict(v[1], bx)}" for k, v in results.items()), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"figures in {a.out}")


if __name__ == "__main__":
    main()
