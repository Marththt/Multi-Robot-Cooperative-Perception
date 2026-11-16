"""YOLO-seg box segmentation of a real frame.

YoloSegmenter(weights)(bgr) -> (mask, rectangles, labels): the union of the box masks (0/255), the rectangle of
every kept box, and a label image in which every box has its own index (1, 2, ...; larger masks first, so a
nearer box drawn later stays on top). Real lab frames are darker than the simulated ones: a fixed gamma lift is
applied before the network. The gripper at the bottom of the frame is blanked.
"""

import os

import cv2
import numpy as np

GRIPPER = os.path.join(os.path.dirname(__file__), "..", "imgsim", "assets", "gripper_mask.png")


class YoloSegmenter:
    def __init__(self, weights, device="auto", conf=0.20, iou=0.5, imgsz=960, max_area=0.70, max_aspect=2.5,
                 gamma=1.8):
        import torch
        from ultralytics import YOLO
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = str(device)
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.max_area, self.max_aspect = max_area, max_aspect
        self.gripper = cv2.imread(GRIPPER, cv2.IMREAD_GRAYSCALE)
        self.lut = (np.clip(255 * (np.arange(256) / 255.0) ** (1.0 / gamma), 0, 255).astype(np.uint8)
                    if gamma != 1.0 else None)

    def __call__(self, bgr):
        return self.many([bgr])[0]

    def many(self, bgrs):
        if not bgrs:
            return []
        inp = [cv2.LUT(b, self.lut) for b in bgrs] if self.lut is not None else list(bgrs)
        preds = self.model.predict(source=inp, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                   device=self.device, retina_masks=True, verbose=False)
        results = []
        for bgr, res in zip(bgrs, preds):
            H, W = bgr.shape[:2]
            mask = np.zeros((H, W), np.uint8)
            labels = np.zeros((H, W), np.uint8)
            rects, per_box = [], []
            if res.masks is not None and res.boxes is not None:
                for (x0, y0, x1, y1), m in zip(res.boxes.xyxy.cpu().numpy(), res.masks.data.cpu().numpy()):
                    w, h = x1 - x0, y1 - y0
                    if w <= 2 or h <= 2 or w * h > self.max_area * W * H or w > self.max_aspect * h:
                        continue
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                    rects.append([float(x0), float(y0), float(x1), float(y1)])
                    per_box.append(m > 0.5)
            if per_box:
                per_box = np.asarray(per_box, bool)
                mask = (per_box.any(0) * 255).astype(np.uint8)
                for n, k in enumerate(np.argsort([-p.sum() for p in per_box])):
                    labels[per_box[k]] = n + 1
            if self.gripper is not None:
                g = cv2.resize(self.gripper, (W, H)) > 127
                mask[g] = 0
                labels[g] = 0
            results.append((mask, rects, labels))
        return results


def overlay(bgr, labels, rects=()):
    """The frame darkened outside the boxes, the contour of every box in its own colour, rectangles in red."""
    out = (bgr * 0.35).astype(np.uint8)
    out[labels > 0] = bgr[labels > 0]
    colours = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 128, 255)]
    for k in range(1, int(labels.max()) + 1):
        cnts, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colours[(k - 1) % len(colours)], 2)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    return out
