"""One capture of one robot: which boxes are reported, where, and the lone false detections.

Per box in the field of view and range:
  1. the faces whose marker looks towards the camera are candidates; for each, the viewing angle a,
     the distance d and whether the line of sight to its marker crosses another box or an obstacle
  2. the box is reported with probability P(best face) = sigmoid(w . [1, a, a^2, d, occluded]),
     the best face being the most detectable one; with probability two_face_q * P(second face) the
     second face is reported too (the same box twice, as observed on the real setup)
  3. the reported point = box centre + error along the line of sight (mean depending on d and on the
     face, since the marker is on the face, plus a per-robot range-scale bias and noise growing with d)
     + error across it (per-robot bearing offset + noise)
Then a Poisson number of lone false detections is added at raw distances and bearings drawn as measured.
All values are returned as RAW distance / bearing, i.e. before the correction of calibrate.py.
"""

import math

import numpy as np

from .geometry import blocked, faces_seen, marker_point, viewing_angle, wrap


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def robot_bias(model, rng):
    """Systematic error of one robot in one run: range scale error and bearing offset, i.e. errors along
    and across the line of sight that grow linearly with the distance [m per m]."""
    return (float(rng.normal(0, model.bias_sd_along_rel)), float(rng.normal(0, model.bias_sd_across_rel)))


def to_raw(model, d, b):
    A, B = model.range_fit
    k, b0 = model.bearing_fit
    return A * d + B, k * b + b0


def capture(scene, cap, model, rng, bias):
    """Detections of one capture: list of dicts (distance, bearing, view = raw values; box, face = the
    true source, -1 for a false detection)."""
    x, y, yaw = cap.pose
    cam = np.array([x, y])
    polys = scene.box_polygons()
    half_fov = math.radians(model.hfov_deg) / 2
    out = []
    for j, (c, theta) in enumerate(zip(scene.boxes, scene.thetas)):
        d = float(math.hypot(c[0] - x, c[1] - y))
        b = float(wrap(yaw - math.atan2(c[1] - y, c[0] - x)))
        if d < model.r_min_m or d > model.r_max_m or abs(b) > half_fov:
            continue
        others = [p for k, p in enumerate(polys) if k != j] + list(scene.obstacles)
        cands = []
        for f in faces_seen(cam, c, theta):
            a = viewing_angle(yaw, theta, f)
            occ = 1.0 if blocked(cam, marker_point(c, theta, f), others) else 0.0
            w = model.detect_w
            p = sigmoid(w[0] + w[1] * a + w[2] * a * a + w[3] * d + w[4] * occ)
            cands.append((p, a, f))
        if not cands:
            continue
        cands.sort(reverse=True)
        reported = []
        if rng.random() < cands[0][0]:
            reported.append(cands[0])
            if len(cands) > 1 and rng.random() < model.two_face_q * cands[1][0]:
                reported.append(cands[1])
        for p, a, f in reported:
            out.append(measure(model, rng, cam, yaw, c, d, a, f, j, bias))
    out += false_detections(model, rng)
    rng.shuffle(out)
    return out


def measure(model, rng, cam, yaw, c, d, a, face, box, bias):
    u = (np.asarray(c) - cam) / d                      # along the line of sight, away from the camera
    n = np.array([-u[1], u[0]])
    m0, m1, m2 = model.along_mean
    s0, s1 = model.along_sd
    t0, t1 = model.across_sd
    along = m0 + m1 * d + m2 * math.cos(a) + bias[0] * d + rng.normal(0, max(1e-4, s0 + s1 * d))
    across = bias[1] * d + rng.normal(0, max(1e-4, t0 + t1 * d))
    p = np.asarray(c) + along * u + across * n
    d_cal = float(np.hypot(*(p - cam)))
    b_cal = float(wrap(yaw - math.atan2(p[1] - cam[1], p[0] - cam[0])))
    d_raw, b_raw = to_raw(model, d_cal, b_cal)
    view = abs(a + rng.normal(0, model.view_noise_sd_rad))
    return dict(distance=d_raw, bearing=b_raw, view=view, box=box, face=face)


def false_detections(model, rng):
    out = []
    half = model.hfov_deg / 2
    for _ in range(int(rng.poisson(model.false_rate))):
        lo, hi = model.false_near_range_m if rng.random() < model.false_near_share else model.false_far_range_m
        d = rng.uniform(lo, hi)
        if rng.random() < model.false_edge_share:
            b = math.radians(rng.uniform(*model.false_edge_band_deg)) * (1 if rng.random() < 0.5 else -1)
        else:
            b = math.radians(rng.uniform(-half, half))
        out.append(dict(distance=d, bearing=b, view=rng.uniform(0.0, 1.6), box=-1, face=-1))
    return out