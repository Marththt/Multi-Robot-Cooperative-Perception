"""Compares the image-level reports (imgsim/generate.py) with the sensor file fitted on the real runs, i.e. checks
the measurement-level simulator against the images.

On one set of imgsim/generate.py:
  1. range and bearing calibration of the set, as pipeline/calibrate.py does for the real runs
     (d_meas = A d_true + B, b_meas = k b_true + b0 on the reports of a true box; residual spread; P_D of the
     boxes in range and field of view), written as <set>/sensor_calib.json and compared with the real one
     (config/sensor_calib.json)
  2. with that calibration, the checks of module M1 (sim/plan_check.py) with the image reports in place of the
     simulated ones: position error along and across the line of sight per distance, detection probability per
     viewing angle and per distance, two-face rate, false detections per capture, each against the sensor file
     (config/sensor_model.json, fitted on 607 real reports)
  3. real frames (data/survey_rounds/*/frames) next to rendered ones
Outputs in --out: fig_IMG-1_position_error.png, fig_IMG-2_detection.png, fig_IMG-3_frames.png, table_IMG.csv.

    python -m imgsim.compare --set-dir data/imgsim/G0_nominal_circle/ --out results/imgsim/G0_nominal_circle/
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

from gnn.dataset import build_scenes, read_records
from sim.model import SensorModel
from sim.plan_check import binned, detection_pairs, mean_se, plot_detection, plot_position, position_errors


def fit_calibration(set_dir, hfov_deg=88.0, r_max=3.0):
    """Range and bearing fit of the raw reports of a true box, as pipeline/calibrate.py (camera offset 0)."""
    with open(os.path.join(set_dir, "ground_truth.json")) as f:
        truth = json.load(f)
    d_t, d_m, b_t, b_m = [], [], [], []
    for r in read_records(set_dir):
        j = r.get("sim_box")
        if j is None or j < 0:
            continue
        x, y, yaw = r["robot_pose"]
        bx = truth[r["run_id"]][j]
        dx, dy = bx["x"] - x, bx["y"] - y
        d_t.append(math.hypot(dx, dy))
        b_t.append((yaw - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi)
        d_m.append(float(r["distance"]))
        b_m.append(float(r["bearing_rad"]))
    d_t, d_m, b_t, b_m = map(np.array, (d_t, d_m, b_t, b_m))
    A, B = np.polyfit(d_t, d_m, 1)
    k, b0 = np.polyfit(b_t, b_m, 1)
    sd = float(np.std(d_m - (A * d_t + B)))
    sl = float(np.std((b_m - (k * b_t + b0)) * d_t))
    return dict(range_fit=[round(float(A), 4), round(float(B), 4)], bearing_fit=[round(float(k), 4), round(float(b0), 5)],
                sigma_pos=round(math.sqrt((sd ** 2 + sl ** 2) / 2), 4), camera_offset=0.0,
                residual_distance_m=round(sd, 4), residual_lateral_m=round(sl, 4), detections=int(len(d_t)),
                source=os.path.abspath(set_dir))


def frames_figure(set_dir, out, plt, rng):
    import cv2
    real = sorted(glob.glob("data/survey_rounds/L*/frames/*/*.jpg"))
    synth = sorted(glob.glob(os.path.join(set_dir, "images", "*.jpg")))
    if not real or not synth:
        return False
    picks = [real[i] for i in rng.choice(len(real), 3, replace=False)] + \
            [synth[i] for i in rng.choice(len(synth), 3, replace=False)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 5.6))
    for a, p, title in zip(ax.flat, picks, ["real"] * 3 + ["rendered"] * 3):
        a.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        a.set_title(f"{title}: {os.path.basename(p)}", fontsize=8)
        a.axis("off")
    fig.suptitle("Fig. IMG-3: real robot frames (top) and rendered frames (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_IMG-3_frames.png"), dpi=100)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--model", default="config/sensor_model.json", help="sensor file fitted on the real runs")
    ap.add_argument("--real-calib", default="config/sensor_calib.json")
    ap.add_argument("--detection-scenes", type=int, default=2000)
    ap.add_argument("--out", default=None, help="default results/imgsim/<set>/")
    ap.add_argument("--save-calib", default=None,
                    help="also write the calibration here (e.g. config/sensor_calib_imgsim.json, for every image set "
                         "rendered with the same camera)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = os.path.basename(os.path.normpath(a.set_dir))
    out = a.out or os.path.join("results", "imgsim", name)
    os.makedirs(out, exist_ok=True)
    model = SensorModel.load(a.model)

    calib = fit_calibration(a.set_dir, model.hfov_deg, model.r_max_m)
    calib_path = os.path.join(a.set_dir, "sensor_calib.json")
    for path in [calib_path] + ([a.save_calib] if a.save_calib else []):
        with open(path, "w") as f:
            json.dump(calib, f, indent=2)
    with open(a.real_calib) as f:
        real = json.load(f)
    rows = [dict(quantity="range fit A (d_meas = A d_true + B)", real=real["range_fit"][0], images=calib["range_fit"][0]),
            dict(quantity="range fit B [m]", real=real["range_fit"][1], images=calib["range_fit"][1]),
            dict(quantity="bearing fit k", real=real["bearing_fit"][0], images=calib["bearing_fit"][0]),
            dict(quantity="bearing fit b0 [rad]", real=real["bearing_fit"][1], images=calib["bearing_fit"][1]),
            dict(quantity="residual distance sd [m]", real=real.get("residual_distance_m"),
                 images=calib["residual_distance_m"]),
            dict(quantity="residual lateral sd [m]", real=real.get("residual_lateral_m"),
                 images=calib["residual_lateral_m"])]

    with open(os.path.join(a.set_dir, "geometry.jsonl")) as f:
        geometry = {g["run"]: g for g in map(json.loads, f)}
    scenes = build_scenes(a.set_dir, os.path.join(a.set_dir, "ground_truth.json"), calib_path)
    err = position_errors(scenes)
    plot_position(err, model, os.path.join(out, "fig_IMG-1_position_error.png"), plt, points="images",
                  title="Fig. IMG-1: position error of the reports detected in the rendered images, against the "
                        "sensor file fitted on the real runs")
    pairs = detection_pairs(scenes[:a.detection_scenes], geometry, model)
    plot_detection(pairs, os.path.join(out, "fig_IMG-2_detection.png"), plt, points="images",
                   title="Fig. IMG-2: detection probability in the rendered images, against the sensor file fitted "
                         "on the real runs")
    captures = sum(len(geometry[s["run"]]["poses"]) for s in scenes)
    false = sum(int((s["label"] < 0).sum()) for s in scenes)
    rows += [dict(quantity="P_D, (robot, box) pairs in view", real=round(float(pairs[:, 2].mean()), 4),
                  images=round(float(pairs[:, 4].mean()), 4)),
             dict(quantity="two-face rate, per pair in view", real=round(float(pairs[:, 3].mean()), 4),
                  images=round(float(pairs[:, 5].mean()), 4)),
             dict(quantity="false detections per capture", real=model.false_rate, images=round(false / captures, 4)),
             dict(quantity="reports per scene", real="", images=round(float(np.mean([len(s["pos"]) for s in scenes])), 2))]
    with open(os.path.join(out, "table_IMG.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "real", "images"])
        w.writeheader()
        w.writerows(rows)
    frames_figure(a.set_dir, out, plt, np.random.default_rng(0))
    print(f"{len(scenes)} scenes, {sum(len(s['pos']) for s in scenes)} reports; calibration written to {calib_path}")
    print(f"  {'quantity':<42}{'real / sensor file':>20}{'images':>12}")
    for r in rows:
        print(f"  {r['quantity']:<42}{str(r['real']):>20}{str(r['images']):>12}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
