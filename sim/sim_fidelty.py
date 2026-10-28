"""Fidelity check of the simulator: real layouts against their simulated replays (sim_generate.py replay).

1. Detection level, per layout (real vs simulated, same loader and same calibration):
     detections per capture, share of boxes in view that are reported, error along / across the line
     of sight per distance band, false detections per capture (all, and those inside 3 m), two-face rate
2. Fusion level, per layout and method, from the scenes.csv of evaluate_runs.py on both sets:
     the real exact-count rate (7 scenes) must lie inside the 5-95% range of the rate of 7 simulated
     scenes (one per round, drawn at random from the replays), and the count errors must have the
     same sign (too many / too few boxes)

    python sim_fidelity.py --real-survey-dir survey_rounds/ --real-ground-truth config/ground_truth_positions.json \\
        --sim-dir sim_replay/ --sensor-calib config/sensor_calib.json --layouts L4,L6,L8,L9,L11 \\
        --real-scenes evaluation_calib/scenes.csv --sim-scenes evaluation_sim/scenes.csv --out fidelity/
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

from gnn_dataset import build_scenes

MATCH_M = 0.30              # a box in view counts as reported if a detection lies this close
OWN_M = 0.25                # a detection within this distance belongs to the box (else: false detection)
NEAR_FALSE_M = 3.0          # false detections closer than this (calibrated distance) land in the arena
BANDS = [(0.0, 1.0), (1.0, 1.5), (1.5, 3.0)]
METHODS = ["zone_class", "union", "phd", "cphd", "mht"]
LABELS = dict(zone_class="Zone+class", union="Union", phd="PHD", cphd="CPHD", mht="MHT")


def detection_stats(scenes, hfov_deg=88.0, r_max=3.0):
    half = math.radians(hfov_deg) / 2
    n_cap = n_det = in_view = reported = 0
    false_all = false_near = 0
    robot_box = twice = 0
    along = defaultdict(list)
    across = defaultdict(list)
    for s in scenes:
        truth = s["truth"]
        gaps = np.linalg.norm(s["pos"][:, None] - truth[None], axis=2) if len(s["pos"]) else np.zeros((0, len(truth)))
        for r in range(len(s["robots"])):
            n_cap += 1
            x, y, yaw = s["poses"][r]
            mine = np.where(s["det_robot"] == r)[0]
            n_det += len(mine)
            for j, c in enumerate(truth):
                d = math.hypot(c[0] - x, c[1] - y)
                b = (yaw - math.atan2(c[1] - y, c[0] - x) + math.pi) % (2 * math.pi) - math.pi
                own = [i for i in mine if gaps[i, j] <= OWN_M and gaps[i].argmin() == j]
                if own:
                    robot_box += 1
                    twice += len(own) > 1
                    u = np.array([c[0] - x, c[1] - y]) / d
                    for i in own:
                        e = s["pos"][i] - c
                        for lo, hi in BANDS:
                            if lo <= d < hi:
                                along[(lo, hi)].append(float(e @ u))
                                across[(lo, hi)].append(float(u[0] * e[1] - u[1] * e[0]))
                if d <= r_max and abs(b) <= half:
                    in_view += 1
                    reported += any(gaps[i, j] <= MATCH_M for i in mine)
            for i in mine:
                if gaps[i].min() > OWN_M:
                    false_all += 1
                    false_near += s["distance"][i] < NEAR_FALSE_M
    out = dict(detections_per_capture=n_det / max(n_cap, 1), reported_in_view=reported / max(in_view, 1),
               false_per_capture=false_all / max(n_cap, 1), near_false_per_capture=false_near / max(n_cap, 1),
               two_face_rate=twice / max(robot_box, 1))
    for lo, hi in BANDS:
        a, c = np.array(along[(lo, hi)]), np.array(across[(lo, hi)])
        out[f"along_sd_{lo:g}-{hi:g}m_cm"] = 100 * a.std() if len(a) > 2 else math.nan
        out[f"along_mean_{lo:g}-{hi:g}m_cm"] = 100 * a.mean() if len(a) > 2 else math.nan
        out[f"across_sd_{lo:g}-{hi:g}m_cm"] = 100 * c.std() if len(c) > 2 else math.nan
    return out


def read_scenes(path, layouts):
    rows = defaultdict(list)          # (layout, method) -> list of (round, dn)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["k"]) != int(r["n_robots"]) or r["layout"] not in layouts:
                continue
            rows[(r["layout"], r["method"])].append((int(r["round"]), int(r["dn"])))
    return rows


def seven_scene_range(sim, rounds, rng, draws=4000):
    """5-95% range of the exact-count rate of one simulated scene per real round."""
    by_round = defaultdict(list)
    for rnd, dn in sim:
        by_round[rnd].append(dn == 0)
    pools = [np.array(by_round[r]) for r in rounds if by_round[r]]
    rates = np.mean([rng.choice(p, draws) for p in pools], axis=0)
    return np.percentile(100 * rates, 5), np.percentile(100 * rates, 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-survey-dir", required=True)
    ap.add_argument("--real-ground-truth", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--sensor-calib", required=True)
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--real-scenes", default=None, help="evaluate_runs.py scenes.csv of the real runs")
    ap.add_argument("--sim-scenes", default=None, help="evaluate_runs.py scenes.csv of the simulated runs")
    ap.add_argument("--out", default="fidelity/")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layouts = [l.strip() for l in a.layouts.split(",") if l.strip()]
    rng = np.random.default_rng(a.seed)

    real = [s for s in build_scenes(a.real_survey_dir, a.real_ground_truth, a.sensor_calib) if s["layout"] in layouts]
    sim = [s for s in build_scenes(a.sim_dir, os.path.join(a.sim_dir, "ground_truth.json"), a.sensor_calib)
           if s["layout"] in layouts]

    # ---- 1. detection level
    det_rows = []
    print("1. DETECTION LEVEL (real / simulated)")
    for group in layouts + ["all"]:
        rs = [s for s in real if group == "all" or s["layout"] == group]
        ss = [s for s in sim if group == "all" or s["layout"] == group]
        sr, sm = detection_stats(rs), detection_stats(ss)
        for k in sr:
            det_rows.append(dict(layout=group, metric=k, real=round(sr[k], 4), sim=round(sm[k], 4)))
        if group == "all":
            for k in sr:
                print(f"  {k:<28}{sr[k]:>9.3f}{sm[k]:>9.3f}")
    with open(os.path.join(a.out, "fidelity_detections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0]))
        w.writeheader()
        w.writerows(det_rows)

    # ---- 2. fusion level
    if not (a.real_scenes and a.sim_scenes):
        return
    R, S = read_scenes(a.real_scenes, layouts), read_scenes(a.sim_scenes, layouts)
    fus_rows = []
    print("\n2. FUSION LEVEL: exact count %, real vs simulated 5-95% range of 7 scenes; count-error sign")
    print("  " + f"{'layout':<8}" + "".join(f"{LABELS[m]:>22}" for m in METHODS))
    n_pass = n_all = 0
    for lay in layouts:
        line = f"  {lay:<8}"
        for m in METHODS:
            r, s = R.get((lay, m), []), S.get((lay, m), [])
            if not r or not s:
                line += f"{'-':>22}"
                continue
            real_rate = 100 * np.mean([dn == 0 for _, dn in r])
            lo, hi = seven_scene_range(s, sorted({rnd for rnd, _ in r}), rng)
            ok = lo - 1e-9 <= real_rate <= hi + 1e-9
            sign_r = np.sign(sum(dn for _, dn in r))
            sign_s = np.sign(np.mean([dn for _, dn in s]))
            same_sign = sign_r == sign_s or real_rate == 100
            n_pass += ok and same_sign
            n_all += 1
            mark = "ok" if ok and same_sign else ("sign" if ok else "OUT")
            line += f"{real_rate:>6.0f} [{lo:>3.0f}-{hi:>3.0f}] {mark:<4}"
            fus_rows.append(dict(layout=lay, method=m, real_exact_pct=round(real_rate, 1),
                                 sim_exact_pct=round(100 * np.mean([dn == 0 for _, dn in s]), 1),
                                 sim_p5=round(lo, 1), sim_p95=round(hi, 1), real_mean_dn=round(np.mean([dn for _, dn in r]), 3),
                                 sim_mean_dn=round(float(np.mean([dn for _, dn in s])), 3), inside=ok, same_sign=same_sign))
        print(line)
    print("\n  overall (all check layouts): real exact % / simulated exact %")
    for m in METHODS:
        r = [dn for lay in layouts for _, dn in R.get((lay, m), [])]
        s = [dn for lay in layouts for _, dn in S.get((lay, m), [])]
        if r and s:
            print(f"  {LABELS[m]:<12}{100 * np.mean([d == 0 for d in r]):>6.0f}{100 * np.mean([d == 0 for d in s]):>8.0f}"
                  f"   mean dn {np.mean(r):+.2f} / {np.mean(s):+.2f}")
    print(f"\n  {n_pass} of {n_all} layout-method cells pass (rate inside the range and same error sign)")
    with open(os.path.join(a.out, "fidelity_fusion.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fus_rows[0]))
        w.writeheader()
        w.writerows(fus_rows)
    print(f"\nwritten in {a.out}: fidelity_detections.csv, fidelity_fusion.csv")


if __name__ == "__main__":
    main()