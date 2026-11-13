"""Which text prompt of the open-vocabulary detector (bevnet/segment.py) finds the boxes and nothing else.

On a few real frames (round 0, first reading, every robot) of chosen layouts, every prompt is run through the
segmentation, and its rectangles are compared with the boxes the robot can see: the marker centre of every
box not hidden by another box, projected into the frame with the camera pose (bevnet/real.true_markers). A
rectangle with a projected box centre in it is a box found; a rectangle with none is a false box; a visible box
inside no rectangle is a missed box. Prints a table per prompt and per block of layouts, and writes a figure
with the detector rectangles of every prompt (one row per frame, one column per prompt).

The default layouts are two per block of the capture plan: A base, C white pyramids, D white objects, E light,
plus one layout of the first captures.

    python -m tools.prompt_check --out results/figures/prompt_check.png
    python -m tools.prompt_check --prompts "a white box." "a white box with blue stripes." --layouts R61 R69
"""

import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np

from bevnet import real
from bevnet.segment import BoxSegmenter

PROMPTS = ["a white box.", "a white box with blue stripes.", "a white box with blue tape.",
           "a white triangular box with blue stripes."]
LAYOUTS = {"R03": "A", "R20": "A", "R61": "C", "R65": "C", "R69": "D", "R96": "D", "R73": "E", "R101": "E",
           "L4": "first"}


def find_run(layout, patterns):
    runs = sorted(d for p in patterns for d in glob.glob(p)
                  if os.path.isdir(os.path.join(d, "raw")) and os.path.basename(os.path.normpath(d)).split("_")[0] == layout)
    return real.best_runs(runs)[0] if runs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", default=PROMPTS)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    ap.add_argument("--runs", nargs="+", default=["data/survey_v2_best/*", "data/survey_rounds/*"])
    ap.add_argument("--threshold", type=float, default=0.30, help="detector box threshold, as bevnet.real")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/figures/prompt_check.png")
    a = ap.parse_args()
    segs = {p: BoxSegmenter(device=a.device, text=p, box_threshold=a.threshold) for p in a.prompts[:1]}
    base = segs[a.prompts[0]]
    for p in a.prompts[1:]:                       # the same models, only the text changes
        s = object.__new__(BoxSegmenter)
        s.__dict__.update(base.__dict__)
        s.text = p
        segs[p] = s
    rows, tiles = [], []
    for layout in a.layouts:
        run = find_run(layout, a.runs)
        if run is None:
            print(f"{layout}: no run found")
            continue
        _layout, boxes, frames, cams = real.load_run(run)
        key = min(k for k in frames if k[1] == 0 and k[2] == 0)            # first round, first reading
        for q, rec in sorted(frames[key].items()):
            img = cv2.imread(os.path.join(run, rec["image"].replace("\\", "/")))
            if img is None:
                continue
            H, W = img.shape[:2]
            pose = real.camera_pose(rec)
            visible = real.true_markers(boxes, pose, math.radians(real.PITCH_DEG.get(q, 11.5)), cams[q], (W, H), None)
            centres = [((u0 + u1) / 2, (v0 + v1) / 2) for _, (u0, v0, u1, v1) in visible]
            row = []
            for p, seg in segs.items():
                mask, rects, _labels = seg(img)
                inside = [[x0 <= u <= x1 and y0 <= v <= y1 for (u, v) in centres] for x0, y0, x1, y1 in rects]
                found = sum(any(r[j] for r in inside) for j in range(len(centres)))
                false = sum(not any(r) for r in inside)
                rows.append(dict(prompt=p, layout=layout, block=LAYOUTS.get(layout, ""), robot=q,
                                 visible=len(centres), found=found, missed=len(centres) - found, false=false))
                view = cv2.resize(real_overlay(img, mask, rects, centres), (480, 270))
                row.append(view)
            tiles.append(np.hstack(row))
            print(f"{layout} {q}: " + "  ".join(f"[{p}] found {r['found']}/{r['visible']} false {r['false']}"
                                                for p, r in zip(segs, rows[-len(segs):])), flush=True)
    print("\nprompt                                        frames  visible  found  missed  false   (by block: false)")
    for p in segs:
        r = [x for x in rows if x["prompt"] == p]
        by = {b: sum(x["false"] for x in r if x["block"] == b) for b in sorted({x["block"] for x in r})}
        print(f"{p:45s} {len(r):6d} {sum(x['visible'] for x in r):8d} {sum(x['found'] for x in r):6d} "
              f"{sum(x['missed'] for x in r):7d} {sum(x['false'] for x in r):6d}   " +
              " ".join(f"{b}:{v}" for b, v in by.items()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    header = np.full((40, tiles[0].shape[1], 3), 255, np.uint8)
    for k, p in enumerate(segs):
        cv2.putText(header, p, (10 + 480 * k, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, np.vstack([header] + tiles))
    print(f"\nwritten {a.out} and {os.path.splitext(a.out)[0]}.csv")


def real_overlay(img, mask, rects, centres):
    """The frame darkened outside the mask, detector rectangles in red, visible true box centres in green."""
    out = (img * 0.35).astype(np.uint8)
    out[mask > 0] = img[mask > 0]
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 4)
    for u, v in centres:
        cv2.circle(out, (int(u), int(v)), 10, (0, 255, 0), -1)
    return out


if __name__ == "__main__":
    main()
