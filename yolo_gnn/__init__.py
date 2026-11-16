"""YOLO + marker network: the proposed method.

The ArUco detections of all the robots go to the marker network (gnn/), trained in simulation only. A YOLO
instance-segmentation model, also trained in simulation only (imgsim/), segments every box in the real frames;
each segmented box gives a bearing ray. The rays add the boxes whose marker no robot detected (triangulation
of rays from different robots) and forbid the merging of two detections that the rays show to be different
boxes (cannot-link).

  camera.py        real runs, camera poses, pitch and yaw calibration from the markers
  train_seg.py     simulated segmentation data set and YOLO-seg fine-tuning
  segment.py       YoloSegmenter: (mask, rectangles, per-box labels) of a frame
  segment_real.py  segments every real frame once (cache of per-box label images)
  hybrid.py        rays, triangulation, cannot-link
  evaluate.py      all the real scenes, two-fold cross-validation over the layouts for the thresholds
"""
