"""Box masks of real camera frames: everything but the boxes removed, as in the boxes-only simulation
(imgsim/render.py, Appearance.mask_only).

Two pre-trained models, no training: Grounding DINO (open-vocabulary detector, text prompt "a white box.")
gives the box rectangles, SAM (Segment Anything) turns each rectangle into a pixel mask; the masks are joined.
Rectangles that are too large (walls, desks) or much wider than tall are dropped, and the own gripper at the
bottom of the frame is blanked. The models are downloaded from the Hugging Face hub the first time
(pip install transformers).

    from bevnet.segment import BoxSegmenter
    seg = BoxSegmenter()                  # device auto: mps / cuda / cpu
    mask, rects = seg(bgr_image)          # uint8 0 / 255, same size as the image
"""

import os

import cv2
import numpy as np

DETECTOR = "IDEA-Research/grounding-dino-tiny"
SEGMENTER = "facebook/sam-vit-base"
GRIPPER = os.path.join(os.path.dirname(__file__), "..", "imgsim", "assets", "gripper_mask.png")


class BoxSegmenter:
    """Mask of the boxes in a real frame: Grounding DINO finds the rectangles that match `text`, SAM draws the
    mask of the object in each of them, and the gripper at the bottom of the frame is blanked. Calling it
    returns (mask, rectangles, labels): labels gives every box its own index, for the instance borders."""

    def __init__(self, device="auto", text="a white box.", box_threshold=0.30, text_threshold=0.25,
                 max_area=0.70, max_aspect=2.5):
        import torch
        from PIL import Image  # noqa: F401  (needed by the processors)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.torch, self.device = torch, torch.device(device)
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(SEGMENTER)
        self.sam = SamModel.from_pretrained(SEGMENTER).to(self.device).eval()
        self.text, self.box_threshold, self.text_threshold = text, box_threshold, text_threshold
        self.max_area, self.max_aspect = max_area, max_aspect
        g = cv2.imread(GRIPPER, cv2.IMREAD_GRAYSCALE)
        self.gripper = g

    def rectangles(self, pil):
        torch = self.torch
        inputs = self.det_proc(images=pil, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.det(**inputs)
        size = [pil.size[::-1]]
        try:
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], box_threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=size)[0]
        except TypeError:                                   # newer transformers: threshold instead of box_threshold
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=size)[0]
        W, H = pil.size
        keep = []
        for b in res["boxes"].float().cpu().numpy():
            x0, y0, x1, y1 = b
            w, h = x1 - x0, y1 - y0
            if w <= 2 or h <= 2 or w * h > self.max_area * W * H or w > self.max_aspect * h:
                continue
            keep.append([float(x0), float(y0), float(x1), float(y1)])
        return keep

    def __call__(self, bgr):
        from PIL import Image
        torch = self.torch
        H, W = bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rects = self.rectangles(pil)
        mask = np.zeros((H, W), np.uint8)
        labels = np.zeros((H, W), np.uint8)                 # 0 = background, k = box k (see real.py)
        if rects:
            inputs = self.sam_proc(pil, input_boxes=[rects], return_tensors="pt")
            inputs = {k: (v.to(self.device, dtype=torch.float32) if v.dtype == torch.float64 else v.to(self.device))
                      for k, v in inputs.items()}
            with torch.no_grad():
                out = self.sam(**inputs, multimask_output=False)
            masks = self.sam_proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
            per_box = masks.numpy().reshape(len(rects), H, W)
            mask = (per_box.any(0) * 255).astype(np.uint8)
            # one label per box; where two masks overlap the smaller one wins: an overlap is mostly the
            # mask of one box spilling onto the box next to it, and the smaller mask is the tighter one
            for k in np.argsort([-m.sum() for m in per_box]):
                labels[per_box[k]] = k + 1
        if self.gripper is not None:
            g = cv2.resize(self.gripper, (W, H)) > 127
            mask[g] = 0
            labels[g] = 0
        return mask, rects, labels


def overlay(bgr, mask, rects=(), labels=None):
    """The frame darkened outside the mask, mask contours in green (with labels: the contour of every box
    in its own colour), detector rectangles in red."""
    out = (bgr * 0.35).astype(np.uint8)
    out[mask > 0] = bgr[mask > 0]
    if labels is None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    else:
        colours = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 128, 255)]
        for k in range(1, int(labels.max()) + 1):
            cnts, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, colours[(k - 1) % len(colours)], 2)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    return out
