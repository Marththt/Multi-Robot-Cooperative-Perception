"""Fits the sensor model of the simulator on real runs and writes it as JSON.

Only the layouts given with --layouts are used (default L5, L7, L10: one 2x2 block, one line, one
scattered layout), so the other layouts stay free for the fidelity check (sim_fidelity.py).

What is fitted, from the real detections and the OptiTrack ground truth:
  box orientation    per box, from the viewing angles reported by the robots (scene state, not noise)
  detection          P(box reported) as a logistic function of the viewing angle of its best face, the
                     distance and whether that face is occluded by another box
  two faces          rate at which the second face of a box is reported too
  position error     along the line of sight: mean c0 + c1 d + c2 cos(a), noise s0 + s1 d;
                     across: noise t0 + t1 d; plus, per robot and run, a range-scale error and a
                     bearing offset (errors growing linearly with d)
  viewing angle      noise of the reported angle
  false detections   rate per capture, raw distance (near / far groups) and bearing (share at the edge
                     of the field of view)

False detections and two-face reports are rare (a few per layout). --rare-events all fits them on every
layout instead; the output records which choice was made.

    python sim_calibrate.py --survey-dir survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --sensor-calib config/sensor_calib.json --layouts L5,L7,L10 --out config/sensor_model.json
"""

import argparse
import math
from collections import defaultdict

import numpy as np

from gnn_dataset import build_scenes, load_calibration
from sim.geometry import blocked, box_polygon, faces_seen, marker_point, viewing_angle, wrap
from sim.model import SensorModel
from sim.scene import estimate_orientations

FALSE_DET_MIN_M = 0.25        # a detection farther than this from every box is a false detection
NEAR_RAW_MAX_M = 3.5          # raw distance separating near and far false detections
EDGE_DEG = 38.0               # bearings beyond this are at the edge of the field of view


def fit_logistic(X, y, l2=1e-3, iters=200):
    """Logistic regression by Newton's method (small ridge for stability)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def orientations_by_run(scenes):
    by_run = defaultdict(list)
    for s in scenes:
        by_run[s["run"]].append(s)
    out = {}
    for run, rs in by_run.items():
        obs = [(int(l), float(s["poses"][s["det_robot"][i]][2]), float(s["view"][i]))
               for s in rs for i, l in enumerate(s["label"]) if l >= 0]
        th = estimate_orientations(len(rs[0]["truth"]), obs)
        out[run] = np.array([t if t is not None else 0.0 for t in th])
    return out


def candidate_faces(cam, yaw, j, truth, thetas):
    """(viewing angle, occluded) of every face of box j that looks towards the camera."""
    c = truth[j]
    others = [box_polygon(truth[k], thetas[k]) for k in range(len(truth)) if k != j]
    out = []
    for f in faces_seen(cam, c, thetas[j]):
        occ = 1.0 if blocked(cam, marker_point(c, thetas[j], f), others) else 0.0
        out.append((viewing_angle(yaw, thetas[j], f), occ, f))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--sensor-calib", required=True, help="calibrate.py --out of the same runs")
    ap.add_argument("--layouts", default="L5,L7,L10")
    ap.add_argument("--rare-events", choices=["calibration", "all"], default="calibration",
                    help="layouts used for false detections and two-face reports")
    ap.add_argument("--hfov-deg", type=float, default=88.0)
    ap.add_argument("--r-max", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    layouts = [l.strip() for l in a.layouts.split(",") if l.strip()]
    scenes = build_scenes(a.survey_dir, a.ground_truth, a.sensor_calib)
    cal = [s for s in scenes if s["layout"] in layouts]
    rare = cal if a.rare_events == "calibration" else scenes
    if not cal:
        raise SystemExit(f"no scene of {layouts} in {a.survey_dir}")
    thetas = orientations_by_run(scenes)
    half_fov = math.radians(a.hfov_deg) / 2
    calib = load_calibration(a.sensor_calib)

    # ---- detection, two faces, position error
    X, y = [], []
    two_events, two_expect = 0, []
    err = []          # (run, robot, d, a, along, across)
    view_err = []
    for s in cal:
        th = thetas[s["run"]]
        truth = s["truth"]
        gaps = np.linalg.norm(s["pos"][:, None] - truth[None], axis=2)
        for r, robot in enumerate(s["robots"]):
            x0, y0, yaw = s["poses"][r]
            cam = np.array([x0, y0])
            mine = np.where(s["det_robot"] == r)[0]
            for j, c in enumerate(truth):
                d = float(np.hypot(*(c - cam)))
                b = float(wrap(yaw - math.atan2(c[1] - y0, c[0] - x0)))
                if d > a.r_max or abs(b) > half_fov:
                    continue
                cands = candidate_faces(cam, yaw, j, truth, th)
                if not cands:
                    continue
                cands.sort(key=lambda q: (q[1], q[0]))           # unoccluded first, then smallest angle
                best = cands[0]
                hits = [i for i in mine if s["label"][i] == j and gaps[i, j] <= FALSE_DET_MIN_M]
                X.append([1.0, best[0], best[0] ** 2, d, best[1]])
                y.append(1.0 if hits else 0.0)
                if hits and len(cands) > 1:
                    two_expect.append((cands[1][0], cands[1][1], d))
                    two_events += len(hits) > 1
                for k, i in enumerate(hits):
                    # viewing-angle noise: against the face whose angle is closest to the reported one
                    allf = [viewing_angle(yaw, th[j], f) for f in range(3)]
                    view_err.append(s["view"][i] - min(allf, key=lambda q: abs(q - s["view"][i])))
                    # position error: against the face the simulator reports (best, then second)
                    ang = cands[min(k, len(cands) - 1)][0]
                    u = (c - cam) / d
                    e = s["pos"][i] - c
                    err.append((s["run"], robot, d, ang, float(e @ u), float(u[0] * e[1] - u[1] * e[0])))
    X, y = np.array(X), np.array(y)
    w = fit_logistic(X, y)
    p_second = [1 / (1 + math.exp(-(w[0] + w[1] * q[0] + w[2] * q[0] ** 2 + w[3] * q[2] + w[4] * q[1])))
                for q in two_expect]
    two_q = two_events / max(sum(p_second), 1e-9)

    E = np.array([(e[2], e[3], e[4], e[5]) for e in err])
    d_, ang_, along, across = E.T
    A = np.column_stack([np.ones(len(E)), d_, np.cos(ang_)])
    cm, *_ = np.linalg.lstsq(A, along, rcond=None)
    res = along - A @ cm
    # per robot and run bias: a range-scale error and a bearing offset, i.e. errors growing linearly
    # with the distance (beta * d), separated from the noise
    groups = defaultdict(list)
    for k, e in enumerate(err):
        groups[(e[0], e[1])].append(k)

    def relative_bias(values):
        beta = {g: float(values[idx] @ d_[idx] / (d_[idx] @ d_[idx])) for g, idx in groups.items()}
        noise = [float(np.var(values[idx] - beta[g] * d_[idx], ddof=1)) / float(d_[idx] @ d_[idx])
                 for g, idx in groups.items() if len(idx) > 2]
        spread = np.var(list(beta.values()), ddof=1) - np.mean(noise)
        white = np.array([values[k] - beta[(err[k][0], err[k][1])] * d_[k] for k in range(len(err))])
        return float(math.sqrt(max(0.0, spread))), white

    sd_bias_a, white_a = relative_bias(res)
    sd_bias_c, white_c = relative_bias(across)
    Ad = np.column_stack([np.ones(len(E)), d_])
    k_half_normal = math.sqrt(math.pi / 2)            # E|x| = sd * sqrt(2/pi)
    sa, *_ = np.linalg.lstsq(Ad, np.abs(white_a) * k_half_normal, rcond=None)
    sc_, *_ = np.linalg.lstsq(Ad, np.abs(white_c) * k_half_normal, rcond=None)
    view_sd = float(1.4826 * np.median(np.abs(np.array(view_err) - np.median(view_err))))

    # ---- false detections (raw distance and bearing)
    A_r, B_r = calib["range_fit"] or (1.0, 0.0)
    K_b, B0_b = calib["bearing_fit"] or (1.0, 0.0)
    fd, captures = [], 0
    for s in rare:
        captures += len(s["robots"])
        gaps = np.linalg.norm(s["pos"][:, None] - s["truth"][None], axis=2).min(axis=1)
        for i in np.where(gaps > FALSE_DET_MIN_M)[0]:
            fd.append((A_r * s["distance"][i] + B_r, K_b * s["bearing"][i] + B0_b))
    fd = np.array(fd) if fd else np.zeros((0, 2))
    near = fd[fd[:, 0] < NEAR_RAW_MAX_M] if len(fd) else fd
    far = fd[fd[:, 0] >= NEAR_RAW_MAX_M] if len(fd) else fd

    m = SensorModel(
        hfov_deg=a.hfov_deg, r_max_m=a.r_max,
        range_fit=list(calib["range_fit"] or [1.0, 0.0]), bearing_fit=list(calib["bearing_fit"] or [1.0, 0.0]),
        camera_offset_m=calib["camera_offset"],
        detect_w=[round(float(v), 4) for v in w], two_face_q=round(float(two_q), 4),
        view_noise_sd_rad=round(view_sd, 4),
        along_mean=[round(float(v), 4) for v in cm], along_sd=[round(float(v), 4) for v in sa],
        across_sd=[round(float(v), 4) for v in sc_],
        bias_sd_along_rel=round(sd_bias_a, 4), bias_sd_across_rel=round(sd_bias_c, 4),
        false_rate=round(len(fd) / max(captures, 1), 4),
        false_near_share=round(len(near) / len(fd), 3) if len(fd) else 0.0,
        false_near_range_m=[round(float(near[:, 0].min()), 2), round(float(near[:, 0].max()), 2)] if len(near)
        else [1.6, 3.0],
        false_far_range_m=[round(float(far[:, 0].min()), 2), round(float(far[:, 0].max()), 2)] if len(far)
        else [4.4, 7.4],
        false_edge_share=round(float(np.mean(np.degrees(np.abs(fd[:, 1])) >= EDGE_DEG)), 3) if len(fd) else 0.5,
        false_edge_band_deg=[EDGE_DEG, a.hfov_deg / 2],
        source=dict(layouts=layouts, rare_events=a.rare_events, survey_dir=a.survey_dir,
                    box_captures=int(len(y)), detections=len(err), two_face_events=int(two_events),
                    false_detections=int(len(fd)), false_captures=int(captures)))
    m.save(a.out)

    pd_hat = 1 / (1 + np.exp(-X @ w))
    print(f"calibration layouts: {', '.join(layouts)} ({len(cal)} scenes); rare events from {a.rare_events}")
    print(f"detection: {int(y.sum())} of {len(y)} boxes in view reported ({y.mean():.2f}); "
          f"model mean {pd_hat.mean():.2f}; w = {np.round(w, 3).tolist()}")
    for lo, hi in [(0, 30), (30, 45), (45, 60), (60, 75), (75, 120)]:
        msk = (np.degrees(X[:, 1]) >= lo) & (np.degrees(X[:, 1]) < hi) & (X[:, 4] == 0)
        if msk.any():
            print(f"   best face {lo:>3}-{hi:<3} deg, unoccluded: real {y[msk].mean():.2f}  model "
                  f"{pd_hat[msk].mean():.2f}  (n={msk.sum()})")
    print(f"two faces: {two_events} events, q = {two_q:.3f}")
    print(f"error along: mean {cm[0]:+.3f} {cm[1]:+.3f}*d {cm[2]:+.3f}*cos(a); noise {sa[0]:.3f} + {sa[1]:.3f}*d; "
          f"robot range-scale bias sd {sd_bias_a:.3f} m/m")
    print(f"error across: noise {sc_[0]:.3f} + {sc_[1]:.3f}*d; robot bearing-offset sd {sd_bias_c:.3f} rad; "
          f"viewing-angle noise {math.degrees(view_sd):.1f} deg")
    print(f"false detections: {len(fd)} in {captures} captures ({len(fd) / max(captures, 1):.3f} per capture), "
          f"{len(near)} near (< {NEAR_RAW_MAX_M} m raw)")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()