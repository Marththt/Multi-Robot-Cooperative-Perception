"""Sensor calibration from real runs with OptiTrack ground truth.

Input : processed captures (<survey_dir>/L*_*/P*.jsonl + ground_truth.json, one feature vector per box).
Output: the measured characteristics of the sensor, printed and (with --out) written to a JSON file
        that evaluate_runs.py reads with --sensor-calib:

        range_fit    d_meas = A * d_true + B            (systematic part of the distance)
        bearing_fit  b_meas = k * b_true + b0           (systematic part of the bearing)
        sigma_pos    residual position error after the two fits [m]: RMS of the distance and of the
                     lateral residual, i.e. the isotropic error the fusion algorithms should assume
        p_d          share of the boxes inside a robot's field of view that produce a feature vector

Usage : python calibrate.py <survey_dir> [--camera-offset 0.12] [--out config/sensor_calib.json]

The camera offset is the distance of the camera ahead of the body pivot. Runs saved with
survey.py --camera-offset-forward-m 0.12 already store the camera pose: use --camera-offset 0 (default
scan finds ~0 for them). Without --camera-offset the scan picks the offset where the bearing slope k is
closest to 1.

Only boxes and runs with ground truth are used. A detection farther than 0.4 m from every true box is
treated as spurious and left out of the fits (it is counted in the report at the end).
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('dir')
ap.add_argument('--camera-offset', type=float, default=None, help="camera ahead of the body pivot [m]")
ap.add_argument('--hfov-deg', type=float, default=88.0, help="horizontal field of view used for p_d")
ap.add_argument('--r-max', type=float, default=3.0, help="maximum detection distance used for p_d [m]")
ap.add_argument('--out', default=None, help="write the calibration to this JSON file")
a = ap.parse_args()

rows, captures, spurious, total = [], {}, 0, 0
for run in sorted(glob.glob(os.path.join(a.dir, 'L*_*'))):
    gt = json.load(open(os.path.join(run, 'ground_truth.json')))
    key = [k for k in gt if '_' in k][0]
    B = np.array([[b['x'], b['y']] for b in gt[key]])
    run_name = os.path.basename(run)
    for f in glob.glob(os.path.join(run, 'P*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if not r.get('class_name') or r.get('distance') is None or r.get('dropped'):
                continue
            x, y, yaw = r['robot_pose']
            cap = captures.setdefault((run_name, r.get('round'), r['robot_id']),
                                      dict(pose=(x, y, yaw), boxes=B, positions=[]))
            cap['positions'].append(r['position'])
            total += 1
            j = int(np.hypot(*(B - np.array(r['position'])).T).argmin())
            if np.hypot(*(B[j] - np.array(r['position']))) > 0.4:
                spurious += 1
                continue
            rows.append(dict(run=run_name, robot=r['robot_id'], x=x, y=y, yaw=yaw,
                             bx=B[j][0], by=B[j][1], d=r['distance'], b=r['bearing_rad']))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"no detection with ground truth found in {a.dir}")


def geometry(delta):
    """True distance and bearing from the camera = body pose moved `delta` forward."""
    cx, cy = df.x + delta * np.cos(df.yaw), df.y + delta * np.sin(df.yaw)
    dx, dy = df.bx - cx, df.by - cy
    return np.hypot(dx, dy), (df.yaw - np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi


def fit(delta):
    dt, bt = geometry(delta)
    A, B = np.polyfit(dt, df.d, 1)
    k, b0 = np.polyfit(bt, df.b, 1)
    res_d = df.d - (A * dt + B)
    res_l = (df.b - (k * bt + b0)) * dt
    return A, B, k, b0, float(res_d.std()), float(res_l.std())


print(f'detections: {total} ({spurious} spurious, farther than 0.4 m from every box)')
print('camera offset scan   A      B   |   k     b0   | residual distance [cm], lateral [cm]')
for dl in np.arange(0.0, 0.2001, 0.02):
    A, B, k, b0, sd, sl = fit(dl)
    print(f'  {dl:.2f} m         {A:.3f} {B:+.3f} | {k:.3f} {b0:+.4f} | {sd*100:.1f} {sl*100:.1f}')
delta = a.camera_offset if a.camera_offset is not None else min(
    np.arange(0.0, 0.2001, 0.005), key=lambda dl: abs(fit(dl)[2] - 1.0))
A, B, k, b0, sd, sl = fit(delta)
sigma_pos = math.sqrt((sd ** 2 + sl ** 2) / 2)

# detection probability: boxes inside the field of view of a robot that produced at least one
# feature vector in that round, detected if a feature vector lies within 0.3 m of the box
half_fov = math.radians(a.hfov_deg) / 2
in_view = seen = 0
for cap in captures.values():
    x, y, yaw = cap['pose']
    cx, cy = x + delta * math.cos(yaw), y + delta * math.sin(yaw)
    P = np.array(cap['positions'])
    for bx, by in cap['boxes']:
        dist = math.hypot(bx - cx, by - cy)
        ang = abs((math.atan2(by - cy, bx - cx) - yaw + math.pi) % (2 * math.pi) - math.pi)
        if dist > a.r_max or ang > half_fov:
            continue
        in_view += 1
        seen += bool(len(P)) and float(np.hypot(*(P - [bx, by]).T).min()) < 0.3
p_d = seen / in_view if in_view else float('nan')

print(f'\ncamera offset used: {delta:.3f} m ahead of the body pivot')
print(f'range   : d_meas = {A:.3f} * d_true {B:+.3f}   -> d = (d_meas {-B:+.3f}) / {A:.3f}')
print(f'bearing : b_meas = {k:.3f} * b_true {b0:+.4f}  -> b = (b_meas {-b0:+.4f}) / {k:.3f}')
print(f'residual after calibration: distance std {sd*100:.1f} cm, lateral std {sl*100:.1f} cm')
print(f'sigma_pos (RMS of the two): {sigma_pos*100:.1f} cm')
print(f'p_d: {seen} of {in_view} boxes in the field of view detected = {p_d:.2f}')
dt, bt = geometry(delta)
print('per robot (A, B, k, b0):')
for r, g in df.groupby('robot'):
    pa, pb = np.polyfit(dt[g.index], g.d, 1)
    pk, p0 = np.polyfit(bt[g.index], g.b, 1)
    print(f'  {r}: {pa:.3f} {pb:+.3f} | {pk:.3f} {p0:+.4f}   (n={len(g)})')

if a.out:
    calib = dict(range_fit=[round(A, 4), round(B, 4)], bearing_fit=[round(k, 4), round(b0, 5)],
                 sigma_pos=round(sigma_pos, 4), p_d=round(p_d, 3), camera_offset=round(float(delta), 3),
                 residual_distance_m=round(sd, 4), residual_lateral_m=round(sl, 4),
                 detections=int(len(df)), spurious=int(spurious), source=os.path.abspath(a.dir))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'\nwritten {a.out}')