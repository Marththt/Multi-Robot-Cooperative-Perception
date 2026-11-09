"""Module M1 of the research plan: does the simulator reproduce the sensor file (A1), and do the gate-input
features of a region depend on the layout (A2)? Also draws the example scenes of module M0.

On one set of sim/plan.py (by default the gate-training set, nominal sensor, both families):
  M1-1  position error along and across the line of sight per distance bin: simulated mean and standard
        deviation against the sensor file (noise, per-robot bias and, along, the spread of the
        viewing-angle term of the mean, added in quadrature)
  M1-2  detection probability per viewing-angle bin and per distance bin: share of (robot, box) pairs in
        range and field of view that are reported, against the mean of the specified sigmoid
  M1-3  histograms of the 8 gate-input features of Section 2.2, split by family, by N in {4, 8, 16} and by
        B in {4, 8, 16}
  M1-4  reports per region and regions per scene
  Table M1  specified vs simulated two-face rate, false-detection rate and P_D, with 3-standard-error
        bounds; largest region; region purity (regions with 0, 1, >= 2 true boxes); boxes split over regions

A region links two reports of any robots when their box-centre estimates are at most --link m apart
and is a connected component of that graph. The plan starts from 0.35 m; on the gate-training set this gives
regions of up to 101 reports (M_max = 64) and 27% of the regions hold two or more boxes, also in the nominal
scene with boxes 30 cm apart, because single linkage chains the noisy reports of neighbouring boxes. 0.20 m
keeps the largest region at 54 reports, 7% of the regions with two or more boxes (3% in G0) and splits 0.4% of
the boxes over two regions (below the 2% of the plan), so it is the default.
The plan computes the histograms on the regions MHT gets right: with --mht-estimates (MHT's boxes on the same
set, from pipeline.plan_mht) only those regions are used, otherwise every region; the table says which.

    python -m sim.plan_check --set-dir data/plan/gate/ --out results/plan/M1/
    python -m sim.plan_check --set-dir data/plan/gate/ --out results/plan/M1/ \
        --mht-estimates results/plan/M2/gate/estimates_H300_o0.jsonl
    python -m sim.plan_check --examples --out results/plan/M0/        # Figure of module M0 only
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from gnn.dataset import build_scenes
from pipeline.plan_regions import LINK_M, region_results, regions
from sim.geometry import blocked, faces_seen, marker_point, viewing_angle
from sim.model import SensorModel
from sim.sensor import sigmoid

FEATURES = ["offset x", "offset y", "distance", "cos view", "sin view", "line of sight x", "line of sight y",
            "same-robot count"]


# ====================================================================
# Regions and gate-input features (Section 2.1-2.2)
# ====================================================================

def gate_features(scene, idx):
    """(len(idx), 8) gate input of the reports idx of one region: offset to the region mean (2), distance,
    cos and sin of the viewing angle, unit line of sight robot -> report (2), other reports of the same robot."""
    pos = scene["pos"][idx]
    cams = scene["poses"][scene["det_robot"][idx], :2]
    los = pos - cams
    los /= np.linalg.norm(los, axis=1, keepdims=True) + 1e-12
    robots = scene["det_robot"][idx]
    same = np.array([(robots == r).sum() - 1 for r in robots], float)
    view = scene["view"][idx]
    return np.column_stack([pos - pos.mean(axis=0), scene["distance"][idx], np.cos(view), np.sin(view), los, same])


# ====================================================================
# Checks
# ====================================================================

def position_errors(scenes):
    """Per true report: distance to the true centre, along and across error, cos of the reported view."""
    rows = []
    for s in scenes:
        ok = s["label"] >= 0
        if not ok.any():
            continue
        c = s["truth"][s["label"][ok]]
        cam = s["poses"][s["det_robot"][ok], :2]
        u = c - cam
        d = np.linalg.norm(u, axis=1)
        u /= d[:, None]
        e = s["pos"][ok] - c
        along = (e * u).sum(axis=1)
        across = u[:, 0] * e[:, 1] - u[:, 1] * e[:, 0]
        rows.append(np.column_stack([d, along, across, np.cos(s["view"][ok])]))
    return np.concatenate(rows)


def detection_pairs(scenes, geometry, model):
    """Per (robot, box) pair in range and field of view: viewing angle and distance of the best face, the
    specified P(reported), P(second face also reported), whether it was reported and how many times."""
    half = math.radians(model.hfov_deg) / 2
    out = []
    for s in scenes:
        g = geometry[s["run"]]
        boxes, yaws = np.array(g["boxes"]), np.array(g["yaws"])
        obstacles = [np.array(o) for o in g["obstacles"]]
        polys = [np.array(p) for p in box_polygons(boxes, yaws)]
        count = defaultdict(int)
        for r, lab in zip(s["det_robot"], s["label"]):
            if lab >= 0:
                count[(s["robots"][r], int(lab))] += 1
        for k, (x, y, yaw) in enumerate(g["poses"]):
            cam = np.array([x, y])
            for j, (c, th) in enumerate(zip(boxes, yaws)):
                d = math.hypot(c[0] - x, c[1] - y)
                b = (yaw - math.atan2(c[1] - y, c[0] - x) + math.pi) % (2 * math.pi) - math.pi
                if d < model.r_min_m or d > model.r_max_m or abs(b) > half:
                    continue
                others = polys[:j] + polys[j + 1:] + obstacles
                cands = []
                for f in faces_seen(cam, c, th):
                    a = viewing_angle(yaw, th, f)
                    occ = 1.0 if blocked(cam, marker_point(c, th, f), others) else 0.0
                    w = model.detect_w
                    cands.append((sigmoid(w[0] + w[1] * a + w[2] * a * a + w[3] * d + w[4] * occ), a))
                if not cands:
                    continue
                cands.sort(reverse=True)
                p2 = model.two_face_q * cands[1][0] if len(cands) > 1 else 0.0
                n = count.get((f"P{k}", j), 0)
                out.append((cands[0][1], d, cands[0][0], cands[0][0] * p2, n > 0, n > 1))
    return np.array(out, float)


def box_polygons(boxes, yaws):
    from sim.geometry import box_polygon
    return [box_polygon(c, t) for c, t in zip(boxes, yaws)]


def binned(x, y, edges, fn):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        out.append(fn(y[m]) if m.sum() >= 20 else (np.nan, np.nan, 0))
    return out


def mean_se(v):
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))), len(v)


def sd_se(v):
    sd = float(v.std(ddof=1))
    return sd, sd / math.sqrt(2 * (len(v) - 1)), len(v)


# ====================================================================
# Figures
# ====================================================================

def plot_position(err, model, path, plt, points="simulated", title="Fig. M1-1: position error of the simulated reports against the sensor file"):
    edges = np.arange(0.25, 3.01, 0.25)
    mids = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    m_along = binned(err[:, 0], err[:, 1], edges, mean_se)
    cosv = binned(err[:, 0], err[:, 3], edges, mean_se)
    m0, m1, m2 = model.along_mean
    spec_mean = [m0 + m1 * d + m2 * cv[0] for d, cv in zip(mids, cosv)]
    ax[0].errorbar(mids, [100 * v[0] for v in m_along], [300 * v[1] for v in m_along], fmt="o", label=points)
    ax[0].plot(mids, [100 * v for v in spec_mean], "-", label="sensor file")
    ax[0].set(xlabel="distance to the box centre [m]", ylabel="mean error along the line of sight [cm]",
              title="Mean along (bars: 3 standard errors)")
    # along: the mean depends on the viewing angle (m2 cos a), so its spread over the reports of a bin adds
    # to the noise and the bias
    var_cos = np.array([v[0] ** 2 if np.isfinite(v[0]) else np.nan for v in binned(err[:, 0], err[:, 3], edges, sd_se)])
    for col, name, (s0, s1), bias, extra, colour in [
            (1, "along", model.along_sd, model.bias_sd_along_rel, m2 ** 2 * var_cos, "C0"),
            (2, "across", model.across_sd, model.bias_sd_across_rel, 0.0, "C1")]:
        sims = binned(err[:, 0], err[:, col], edges, sd_se)
        spec = np.sqrt((s0 + s1 * mids) ** 2 + (bias * mids) ** 2 + extra)
        ax[1].errorbar(mids, [100 * v[0] for v in sims], [300 * v[1] for v in sims], fmt="o", color=colour,
                       label=f"{name}, {points}")
        ax[1].plot(mids, 100 * spec, "-", color=colour, label=f"{name}, sensor file")
    ax[1].set(xlabel="distance to the box centre [m]", ylabel="standard deviation [cm]",
              title="Spread (noise and per-robot bias)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_detection(pairs, path, plt, points="simulated", title="Fig. M1-2: detection probability of the (robot, box) pairs in range and field of view"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (col, edges, label) in enumerate([(0, np.radians(np.arange(0, 91, 10)), "viewing angle of the best face [deg]"),
                                             (1, np.arange(0.0, 3.01, 0.25), "distance [m]")]):
        x = pairs[:, col]
        sim = binned(x, pairs[:, 4], edges, mean_se)
        spec = binned(x, pairs[:, 2], edges, mean_se)
        mids = (edges[:-1] + edges[1:]) / 2
        mids = np.degrees(mids) if col == 0 else mids
        ax[k].errorbar(mids, [v[0] for v in sim], [3 * v[1] for v in sim], fmt="o", label=points)
        ax[k].plot(mids, [v[0] for v in spec], "-", label="sensor file (mean of the sigmoid)")
        ax[k].set(xlabel=label, ylabel="P(box reported)", ylim=(0, 1.02))
        ax[k].grid(alpha=0.3)
        ax[k].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_features(feats, groups, path, plt, which="every region"):
    fig, ax = plt.subplots(len(groups), 8, figsize=(22, 3.2 * len(groups)))
    for r, (gname, members) in enumerate(groups):
        for c in range(8):
            a = ax[r, c]
            allv = np.concatenate([feats[m][:, c] for m in members if len(feats[m])])
            lo, hi = np.percentile(allv, [0.5, 99.5])
            bins = np.arange(-0.5, 6.5, 1.0) if c == 7 else np.linspace(lo, hi, 40)
            for m in members:
                if len(feats[m]):
                    a.hist(feats[m][:, c], bins=bins, density=True, histtype="step", lw=1.3,
                           label=f"{m} ({len(feats[m])})")
            if r == 0:
                a.set_title(FEATURES[c], fontsize=10)
            if c == 0:
                a.set_ylabel(gname)
            a.tick_params(labelsize=7)
        ax[r, 0].legend(fontsize=7)
    fig.suptitle(f"Fig. M1-3: gate-input features of the reports of {which} (nominal sensor); "
                 "density per group, report counts in brackets", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_regions(sizes, per_scene, path, plt):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(sizes, bins=np.arange(0.5, max(sizes) + 1.5, 1), log=True)
    ax[0].set(xlabel="reports in the region", ylabel="regions", title=f"Reports per region (largest {max(sizes)})")
    ax[1].hist(per_scene, bins=np.arange(-0.5, max(per_scene) + 1.5, 1))
    ax[1].set(xlabel="regions in the scene", ylabel="scenes", title="Regions per scene")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Fig. M1-4: size and number of the regions")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_scene(ax, scene, title, model):
    from matplotlib.patches import Polygon, Wedge
    half = model.hfov_deg / 2
    for c in scene.captures:
        x, y, yaw = c.pose
        ax.add_patch(Wedge((x, y), model.r_max_m, math.degrees(yaw) - half, math.degrees(yaw) + half,
                           color="tab:blue", alpha=0.05))
        ax.plot(x, y, marker=(3, 0, math.degrees(yaw) - 90), ms=10, color="tab:blue")
    for poly in scene.box_polygons():
        ax.add_patch(Polygon(poly, closed=True, fc="none", ec="k", lw=1.2))
    for o in scene.obstacles:
        ax.add_patch(Polygon(o, closed=True, fc="tab:orange", ec="tab:orange", alpha=0.8))
    ax.add_patch(__import__("matplotlib.pyplot").pyplot.Rectangle((-3, -3), 6, 6, fill=False, ec="gray", ls="--"))
    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), aspect="equal", title=title, xlabel="x [m]", ylabel="y [m]")
    ax.grid(alpha=0.3)


def examples(model, out, plt):
    from sim.plan import NOMINAL, plan_scene, run_model
    from dataclasses import replace
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    cases = [("circle", NOMINAL, "Circle family, nominal scene"), ("random", NOMINAL, "Random family, nominal scene"),
             ("circle", replace(NOMINAL, hidden_robots=2), "E3, h = 2 (obstacles in orange)")]
    for a, (family, spec, title) in zip(ax, cases):
        scene, meta = plan_scene(spec, family, np.random.default_rng(7), "example", "example", run_model(model, spec))
        draw_scene(a, scene, title, model)
    fig.suptitle("M0: example scenes in the 6 m x 6 m arena (dashed); robots: triangles, shaded: field of view "
                 f"{model.hfov_deg:.0f} deg, {model.r_max_m:.0f} m; boxes: 30 cm prisms")
    fig.tight_layout()
    path = os.path.join(out, "fig_M0_examples.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path}")


# ====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", default="data/plan/gate/")
    ap.add_argument("--model", default="config/sensor_model.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--out", default="results/plan/M1/")
    ap.add_argument("--link", type=float, default=LINK_M,
                    help="region link distance g_r [m] (plan default 0.35; 0.20 keeps regions within M_max and "
                         "boxes 30 cm apart in different regions, see M1)")
    ap.add_argument("--max-region", type=int, default=64, help="M_max of the plan")
    ap.add_argument("--detection-scenes", type=int, default=3000,
                    help="scenes used for the detection-probability check (ray casting is slow)")
    ap.add_argument("--examples", action="store_true", help="only draw the example scenes of module M0")
    ap.add_argument("--mht-estimates", default=None,
                    help="estimates_H<H>_o<k>.jsonl of pipeline.plan_mht on the same set: the features (M1-3) are then "
                         "computed on the regions MHT gets right only, as the plan asks"),
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    model = SensorModel.load(a.model)
    if a.examples:
        examples(model, a.out, plt)
        return

    with open(os.path.join(a.set_dir, "runs.csv")) as f:
        meta = {r["run"]: r for r in csv.DictReader(f)}
    with open(os.path.join(a.set_dir, "geometry.jsonl")) as f:
        geometry = {g["run"]: g for g in map(json.loads, f)}
    scenes = build_scenes(a.set_dir, os.path.join(a.set_dir, "ground_truth.json"), a.sensor_calib)
    print(f"{len(scenes)} scenes, {sum(len(s['pos']) for s in scenes)} reports", flush=True)
    rows = []

    # M1-1
    err = position_errors(scenes)
    plot_position(err, model, os.path.join(a.out, "fig_M1-1_position_error.png"), plt)

    # M1-2 and the rates
    pairs = detection_pairs(scenes[:a.detection_scenes], geometry, model)
    plot_detection(pairs, os.path.join(a.out, "fig_M1-2_detection.png"), plt)
    n = len(pairs)
    for name, sim, spec in [("P_D, (robot, box) pairs in view", pairs[:, 4].mean(), pairs[:, 2].mean()),
                            ("two-face rate, per pair in view", pairs[:, 5].mean(), pairs[:, 3].mean())]:
        se = math.sqrt(max(spec * (1 - spec), 1e-12) / n)
        rows.append(dict(quantity=name, specified=spec, simulated=sim, bound=3 * se, n=n,
                         passed=abs(sim - spec) <= 3 * se))
    captures = sum(len(geometry[s["run"]]["poses"]) for s in scenes)
    false = sum(int((s["label"] < 0).sum()) for s in scenes)
    spec = model.false_rate * float(np.mean([float(meta[s["run"]]["false_factor"]) for s in scenes]))
    rows.append(dict(quantity="false detections per capture", specified=spec, simulated=false / captures,
                     bound=3 * math.sqrt(spec / captures), n=captures,
                     passed=abs(false / captures - spec) <= 3 * math.sqrt(spec / captures)))

    # regions: M1-3, M1-4, purity
    estimates = None
    if a.mht_estimates:
        with open(a.mht_estimates) as f:
            estimates = {e["run"]: e for e in map(json.loads, f)}
    feats = defaultdict(list)
    sizes, per_scene, purity, split, boxes_seen = [], [], np.zeros(3), 0, 0
    kept, total_regions = 0, 0
    for s in scenes:
        m = meta[s["run"]]
        if estimates is not None and s["run"] in estimates:
            regs, res = region_results(s, estimates[s["run"]], a.link)
            ok = [bool(r["correct"]) for r in res]
        else:                                # no MHT result for this scene: its regions give no features
            regs = regions(s["pos"], a.link)
            ok = [True] * len(regs) if estimates is None else None
        per_scene.append(len(regs))
        where = defaultdict(set)
        for k, idx in enumerate(regs):
            sizes.append(len(idx))
            true = {int(l) for l in s["label"][idx] if l >= 0}
            purity[min(len(true), 2)] += 1
            for l in s["label"][idx]:
                if l >= 0:
                    where[int(l)].add(k)
            if ok is None:
                continue
            total_regions += 1
            if not ok[k]:
                continue
            kept += 1
            f = gate_features(s, idx)
            feats[m["family"]].append(f)
            for key, val in (("N", int(m["n_robots"])), ("B", int(m["n_boxes"]))):
                if val in (4, 8, 16):
                    feats[f"{key} = {val}"].append(f)
        boxes_seen += len(where)
        split += sum(1 for v in where.values() if len(v) > 1)
    feats = {k: np.concatenate(v) if v else np.zeros((0, 8)) for k, v in feats.items()}
    groups = [("family", ["circle", "random"]), ("robots", ["N = 4", "N = 8", "N = 16"]),
              ("boxes", ["B = 4", "B = 8", "B = 16"])]
    plot_features(feats, groups, os.path.join(a.out, "fig_M1-3_features.png"), plt,
                  "the regions MHT gets right" if estimates is not None else "every region")
    plot_regions(sizes, per_scene, os.path.join(a.out, "fig_M1-4_regions.png"), plt)
    total = purity.sum()
    rows += [dict(quantity="largest region [reports]", specified=a.max_region, simulated=max(sizes), bound="",
                  n=len(sizes), passed=max(sizes) <= a.max_region),
             dict(quantity="regions with no true box", specified="", simulated=purity[0] / total, bound="", n=total, passed=""),
             dict(quantity="regions with one true box", specified="", simulated=purity[1] / total, bound="", n=total, passed=""),
             dict(quantity="regions with two or more true boxes", specified="", simulated=purity[2] / total, bound="",
                  n=total, passed=""),
             dict(quantity="boxes whose reports fall in two or more regions", specified=0.02,
                  simulated=split / boxes_seen, bound="", n=boxes_seen, passed=split / boxes_seen < 0.02),
             dict(quantity="regions used for the features (MHT correct)" if estimates is not None else
                  "regions used for the features (all; no --mht-estimates)", specified="",
                  simulated=kept / total_regions, bound="", n=total_regions, passed="")]
    path = os.path.join(a.out, "table_M1.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "specified", "simulated", "bound", "n", "passed"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print("  " + "  ".join(f"{k}={(f'{v:.4f}' if isinstance(v, float) else v)}" for k, v in r.items()))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
