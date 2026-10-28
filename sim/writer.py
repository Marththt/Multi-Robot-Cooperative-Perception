"""Writes a simulated run in the format of the real runs:
  <out>/<run_id>/P<k>.jsonl      one feature vector per line, same fields as survey.py
  <out>/<run_id>/ground_truth.json
With folder= several runs share one folder (<out>/<folder>/P<k>.jsonl, appended, no per-run ground truth):
every record carries its run_id, so the readers keep the runs apart; large sets then need far fewer files.
Every record has two fields the real records do not have: sim_box (index of the true box, -1 = false detection) and
sim_face. They are ignored by every evaluation script and only serve diagnostics.
"""

import json
import math
import os
from collections import defaultdict


def write_run(out_dir, scene, detections, model, t0=1.0e9, folder=None):
    """detections: {capture index: list of detection dicts from sensor.capture}. Returns the
    ground-truth entries {layout_id: boxes, run_id: boxes}."""
    run_dir = os.path.join(out_dir, folder or scene.run_id)
    os.makedirs(run_dir, exist_ok=True)
    off = model.camera_offset_m
    per_robot = defaultdict(list)
    t = t0
    for ci, cap in enumerate(scene.captures):
        x, y, yaw = cap.pose
        body = (x - off * math.cos(yaw), y - off * math.sin(yaw), yaw)
        for det in detections.get(ci, []):
            t += 0.05
            d, b = det["distance"], det["bearing"]
            per_robot[cap.robot].append(dict(
                class_name="box", confidence=1.0,
                position=[x + d * math.cos(yaw - b), y + d * math.sin(yaw - b)],
                distance=d, bearing_rad=b, viewing_angle_rad=det["view"], robot_pose=list(body), t=t,
                trial_id=f"{scene.layout_id}/{cap.robot}/{cap.round:03d}/+0", robot_id=cap.robot,
                layout_id=scene.layout_id, stop=cap.round, radius_m=cap.radius, angle_deg=cap.angle_deg,
                yaw_offset_deg=0.0, fault_injected=False, dropped=False, zone="unknown", round=cap.round,
                run_id=scene.run_id, sim_box=int(det["box"]), sim_face=int(det["face"])))
        t += 3.0
    for robot, recs in per_robot.items():
        with open(os.path.join(run_dir, f"{robot}.jsonl"), "a" if folder else "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    boxes = [dict(class_name="box", x=round(float(c[0]), 4), y=round(float(c[1]), 4), rb_id=11 + j)
             for j, c in enumerate(scene.boxes)]
    truth = {scene.layout_id: boxes, scene.run_id: boxes}
    if not folder:
        with open(os.path.join(run_dir, "ground_truth.json"), "w") as f:
            json.dump(truth, f, indent=2)
    return truth