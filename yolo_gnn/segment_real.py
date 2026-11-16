"""Segments every real frame (all rounds, readings 0-4, no yaw offset) with the YOLO-seg model and caches one
label image per frame: <cache>/<run folder>/<frame name>_ids.png (0 background, 1..n one index per box).
Frames already in the cache are skipped. A sheet of overlays (16 frames) is written next to the cache.

    python -m yolo_gnn.segment_real --runs "data/survey_all/*" --weights results/yolo_seg/box_seg/weights/best.pt \\
        --cache data/yolo_real/cache
"""

import argparse
import os

import cv2
import numpy as np

from yolo_gnn.camera import load_run, run_dirs
from yolo_gnn.segment import YoloSegmenter, overlay


def cache_path(cache, run_dir, rec):
    name = os.path.basename(rec["image"].replace("\\", "/"))[:-4]
    return os.path.join(cache, os.path.basename(os.path.normpath(run_dir)), name + "_ids.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="glob patterns of the run folders")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", default="data/yolo_real/cache")
    ap.add_argument("--readings", default="0,1,2,3,4")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    readings = {int(k) for k in a.readings.split(",")}

    seg = YoloSegmenter(a.weights, device=a.device, conf=a.conf)
    dirs = run_dirs(a.runs)
    sheet, done, skipped = [], 0, 0
    for n, d in enumerate(dirs):
        _, _, frames, _ = load_run(d)
        todo = []
        for (rnd, k, off), recs in sorted(frames.items()):
            if off != 0 or k not in readings:
                continue
            for q, rec in sorted(recs.items()):
                out = cache_path(a.cache, d, rec)
                if os.path.exists(out):
                    skipped += 1
                    continue
                todo.append((out, os.path.join(d, rec["image"].replace("\\", "/"))))
        for start in range(0, len(todo), a.batch):
            chunk = todo[start:start + a.batch]
            imgs = [cv2.imread(p) for _, p in chunk]
            ok = [(o, im) for (o, _), im in zip(chunk, imgs) if im is not None]
            for (out, img), (mask, rects, labels) in zip(ok, seg.many([im for _, im in ok])):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                cv2.imwrite(out, labels)
                done += 1
                if len(sheet) < 16 and done % 97 == 1:
                    sheet.append(cv2.resize(overlay(img, labels, rects), (640, 360)))
        print(f"  {n + 1}/{len(dirs)} {os.path.basename(os.path.normpath(d))}: {len(todo)} frames segmented",
              flush=True)
    print(f"{done} frames segmented, {skipped} already in {a.cache}")
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]
        fig = os.path.join(os.path.dirname(os.path.normpath(a.cache)), "fig_yolo_real.jpg")
        cv2.imwrite(fig, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"overlays: {fig}")


if __name__ == "__main__":
    main()
