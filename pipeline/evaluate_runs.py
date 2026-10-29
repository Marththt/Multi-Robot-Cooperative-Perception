"""Evaluation of the round-mode runs with the agreed metrics.

A SCENE is one round of one run: the captures of the robots standing at
their positions of that round. For every scene, every subset of the robots
(1, 2, ... all of them) is fused with each of the five methods, and the
estimate is compared with the true box positions of that run (OptiTrack):

  matching    every estimate is paired with at most one true box (optimal
              assignment); pairs farther than the cut-off c are not matched
  e_j         distance between a matched estimate and its true box
  P_D         detection probability: matched true boxes / all true boxes
  RMSE*       over ALL true boxes, a missed box counting as an error of c
  false       estimates matching no true box, per scene
  RMSE        sqrt(mean e_j^2) over the matched boxes of the scene
  dn          n_est - n_true (count error: < 0 missed, > 0 invented)
  OSPA_c      position and count together (order p, cut-off c)

Outputs (in --out):
  scenes.csv              one row per scene, robot subset and method
  summary.csv             per method (and per layout): median / 90th
                          percentile RMSE, exact count, mean |dn|, OSPA
  cdf_rmse.png            CDF of the per-scene RMSE (all robots)
  rmse_by_layout.png      RMSE per layout and method (mean and spread)
  count_error.png         share of scenes with too few / exact / too many
  rmse_vs_robots.png      OSPA and exact count against the number of robots
  rmse_per_scene.png      every scene, mean +- std over the subsets of
                          (all - 1) robots
  layouts.png             the box layouts of the runs

    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --regions config/optitrack_regions.json --camera-offset 0.12 --range-fit 1.064 -0.080 \\
        --out results/evaluation/ --names L1=Line L2=Square L3=Pairs

--camera-offset M moves the saved body pose forward to the camera pose. --range-fit A B and --bearing-fit K B0 undo the systematic part of the measurements
(measured = A * true + B, measured = K * true + B0); pipeline/calibrate.py estimates them.

Sensor calibration (recommended): --sensor-calib config/sensor_calib.json, written by
    python -m pipeline.calibrate survey_rounds/ --camera-offset 0 --out config/sensor_calib.json
sets, for ALL methods at once, the measured characteristics of the sensor:
  range_fit, bearing_fit   distance and bearing correction (unless --range-fit / --bearing-fit are given)
  sigma_pos                position error assumed by MHT, PHD, CPHD (unless --sigma-pos is given)
  p_d                      detection probability assumed by MHT, PHD, CPHD (unless --p-d is given)
  camera_offset            camera ahead of the body pivot, the one the fits were made with
                           (unless --camera-offset is given)
  union threshold          3 * sigma_pos (unless --union-threshold is given)
Without --sensor-calib every value keeps the code default (sigma 0.10 m, p_d 1, union 0.5 m).
The settings actually used are printed at the start.

    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json \\
        --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --out results/evaluation/
"""

import argparse
import csv
import functools
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np

from pipeline import analyze_rounds
from pipeline.analyze_rounds import METHOD_LABELS, fuse
from pipeline.build_scans import ARENA_DEFAULT, bind_run_truth, load_points, load_truth
from vf.fusion.localization_methods import METHODS
from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_regions, sensor_from_arguments,
                                    zones_from_arguments)

# categorical palette (fixed order, validated for adjacent pairs), one per method
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = {}          # layout name -> sort key (its run id), so the figures follow L1, L2, ...


def by_order(names):
    return sorted(names, key=lambda n: (ORDER.get(n, n), n))


# ====================================================================
# Metrics
# ====================================================================

def match(est, truth, c):
    """Optimal one-to-one assignment estimate <-> true box minimising the sum
    of the distances; pairs farther than c are dropped. Returns the list of
    distances of the matched pairs."""
    est = [np.asarray(e, float) for e in est]
    truth = [np.asarray(t, float) for t in truth]
    if not est or not truth:
        return []
    D = np.array([[float(np.hypot(*(e - t))) for t in truth] for e in est])
    Dc = np.minimum(D, c)                       # a pair beyond c is as bad as no pair
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(Dc)
        pairs = list(zip(rows, cols))
    except ImportError:
        small, big = (est, truth) if len(est) <= len(truth) else (truth, est)
        best = None
        for perm in permutations(range(len(big)), len(small)):
            cost = sum(Dc[i, j] if small is est else Dc[j, i] for i, j in enumerate(perm))
            if best is None or cost < best[0]:
                best = (cost, perm)
        pairs = [(i, j) if small is est else (j, i) for i, j in enumerate(best[1])]
    return [D[i, j] for i, j in pairs if D[i, j] <= c]


def detection_metrics(rows, c=0.5):
    """The three summary metrics over a set of scenes (rows of scene_metrics):
      pd_pct           detection probability: true boxes matched by an estimate / all true boxes [%]
      rmse_all_cm      RMSE over ALL true boxes, a box with no estimate within c counting as an error of c [cm]
      false_per_scene  estimates that match no true box, per scene
    Detection probability and RMSE alone reward a method that reports every detection as a box (it finds
    every box, with small errors); the false estimates per scene measure that cost."""
    n_true = sum(int(r["n_true"]) for r in rows)
    n_match = sum(int(r["n_matched"]) for r in rows)
    sq = 0.0
    for r in rows:
        m = int(r["n_matched"])
        e = float(r["rmse"]) if m and r["rmse"] not in ("", None) else 0.0
        sq += m * (e * e if math.isfinite(e) else 0.0) + (int(r["n_true"]) - m) * c * c
    n_false = sum(int(r["n_est"]) - int(r["n_matched"]) for r in rows)
    return dict(pd_pct=100.0 * n_match / n_true if n_true else math.nan,
                rmse_all_cm=100.0 * math.sqrt(sq / n_true) if n_true else math.nan,
                false_per_scene=n_false / len(rows) if rows else math.nan)


def scene_metrics(est, truth, c, p):
    errs = match(est, truth, c)
    n, m = len(truth), len(est)
    rmse = float(math.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    big = max(n, m)
    if big == 0:
        ospa = 0.0
    else:
        ospa = ((sum(e ** p for e in errs) + c ** p * (big - len(errs))) / big) ** (1.0 / p)
    return dict(n_true=n, n_est=m, dn=m - n, n_matched=len(errs), rmse=rmse, ospa=float(ospa))


# ====================================================================
# Scenes
# ====================================================================

def evaluate(points, truth, sensor, zones, methods, c, p, max_k=None, full_only=False, quiet=False):
    """One row per (run, round, robot subset, method); with full_only only the subset of all robots."""
    by_scene = defaultdict(list)
    for k, v in points.items():
        if v["round"] is None or v["yaw_offset"] != 0:
            continue
        by_scene[(v["layout"], v["round"])].append(k)
    rows = []
    for (layout, rnd), keys in sorted(by_scene.items()):
        if layout not in truth:
            continue
        true_xy = [b["xy"] for b in truth[layout]]
        keys = sorted(keys, key=lambda q: points[q]["robot"])
        n_rob = len(keys)
        name = points[keys[0]].get("layout_name", layout)
        for kk in ([n_rob] if full_only else range(1, (max_k or n_rob) + 1)):
            for subset in combinations(keys, kk):
                _poses, _per_class, results = fuse(list(subset), points, sensor, zones, methods)
                for m in methods:
                    est = []
                    for cls_res in results.values():
                        est += list(cls_res.get(m, {}).get("objects", []))
                    r = scene_metrics(est, true_xy, c, p)
                    r.update(run=layout, layout=name, round=rnd, k=kk, n_robots=n_rob, method=m,
                             robots=" ".join(points[q]["robot"] for q in subset))
                    rows.append(r)
        if not quiet:
            print(f"  {layout} round {rnd}: {n_rob} robot(s)", flush=True)
    return rows


# ====================================================================
# Figures
# ====================================================================

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_cdf(rows, methods, path, names):
    import matplotlib.pyplot as plt
    full = [r for r in rows if r["k"] == r["n_robots"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xmax = 0.0
    for i, m in enumerate(methods):
        v = np.sort([r["rmse"] for r in full if r["method"] == m and not math.isnan(r["rmse"])])
        if not len(v):
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        xs = np.concatenate([[0.0], np.repeat(v, 2)])
        ys = np.concatenate([[0.0], [0.0], np.repeat(y, 2)[:-1]])
        ax.plot(xs, ys, color=COLORS[i], lw=2, label=METHOD_LABELS[m])
        xmax = max(xmax, v[-1])
    for q, lab in ((0.5, "median"), (0.9, "90th percentile")):
        ax.axhline(q, color=MUTED, lw=0.8, ls=":")
        ax.text(0.0, q + 0.015, lab, color=MUTED, fontsize=11)
    ax.set_xlim(0, xmax * 1.05 + 1e-3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("RMSE of one scene [m]")
    ax.set_ylabel("fraction of scenes")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def grouped_bars(ax, groups, methods, values, errs=None):
    """values[g][m] -> bar; one cluster per group, one bar per method."""
    n = len(methods)
    w = 0.8 / n
    for i, m in enumerate(methods):
        xs = np.arange(len(groups)) + (i - (n - 1) / 2) * w
        vals = [values[g].get(m, np.nan) for g in groups]
        e = None if errs is None else [errs[g].get(m, 0.0) for g in groups]
        ax.bar(xs, vals, width=w * 0.88, color=COLORS[i], label=METHOD_LABELS[m],
               yerr=e, error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=2))
    ax.set_xticks(np.arange(len(groups)))


def fig_rmse_by_layout(rows, methods, path):
    import matplotlib.pyplot as plt
    full = [r for r in rows if r["k"] == r["n_robots"]]
    layouts = by_order({r["layout"] for r in full})
    mean, std = defaultdict(dict), defaultdict(dict)
    for L in layouts:
        for m in methods:
            v = [r["rmse"] for r in full if r["layout"] == L and r["method"] == m and not math.isnan(r["rmse"])]
            mean[L][m] = np.mean(v) if v else np.nan
            std[L][m] = np.std(v) if v else 0.0
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    grouped_bars(ax, layouts, methods, mean, std)
    ax.set_xticklabels(layouts, fontsize=11, color=TEXT)
    ax.set_ylabel("RMSE per scene [m]\n(mean, std over rounds)")
    ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_count(rows, methods, path):
    import matplotlib.pyplot as plt
    full = [r for r in rows if r["k"] == r["n_robots"]]
    cats = [("too few", lambda d: d < 0, "#c3c2b7"), ("exact", lambda d: d == 0, "#2a78d6"),
            ("too many", lambda d: d > 0, "#52514e")]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    y = np.arange(len(methods))[::-1]
    left = np.zeros(len(methods))
    for lab, f, col in cats:
        share = np.array([100.0 * np.mean([f(r["dn"]) for r in full if r["method"] == m]) for m in methods])
        ax.barh(y, share, left=left, color=col, height=0.6, label=lab, edgecolor="white", linewidth=2)
        for yi, l, s in zip(y, left, share):
            if s >= 8:
                ax.text(l + s / 2, yi, f"{s:.0f}%", ha="center", va="center", fontsize=11,
                        color="white" if col != "#c3c2b7" else TEXT)
        left += share
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=11, color=TEXT)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of scenes [%]")
    ax.legend(frameon=False, fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    style(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_vs_robots(rows, methods, path):
    """OSPA (position and count together) and exact count against the number
    of robots fused. Not the RMSE: it only covers the matched boxes, so with
    few robots (few boxes seen, only the easy ones matched) it looks better
    than it is."""
    import matplotlib.pyplot as plt
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for i, m in enumerate(methods):
        os_, ex = [], []
        for k in ks:
            rr = [r for r in rows if r["k"] == k and r["method"] == m]
            os_.append(np.mean([r["ospa"] for r in rr]) if rr else np.nan)
            ex.append(100.0 * np.mean([r["dn"] == 0 for r in rr]) if rr else np.nan)
        axes[0].plot(ks, os_, color=COLORS[i], marker=MARKERS[i], ms=7, lw=2, label=METHOD_LABELS[m])
        axes[1].plot(ks, ex, color=COLORS[i], marker=MARKERS[i], ms=7, lw=2, label=METHOD_LABELS[m])
    axes[0].set_ylabel("mean OSPA per scene [m]")
    axes[0].set_ylim(0, None)
    axes[1].set_ylabel("scenes with the exact count [%]")
    axes[1].set_ylim(0, 102)
    for ax in axes:
        ax.set_xlabel("robots fused")
        ax.set_xticks(ks)
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=12, ncol=len(methods), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_per_scene(rows, methods, path):
    import matplotlib.pyplot as plt
    sub = [r for r in rows if r["k"] == r["n_robots"] - 1]
    scenes = sorted({(r["layout"], r["round"]) for r in sub}, key=lambda q: (ORDER.get(q[0], q[0]), q[1]))
    fig, ax = plt.subplots(figsize=(11, 4.0))
    n = len(methods)
    w = 0.8 / n
    for i, m in enumerate(methods):
        mu, sd = [], []
        for s in scenes:
            v = [r["rmse"] for r in sub if (r["layout"], r["round"]) == s and r["method"] == m
                 and not math.isnan(r["rmse"])]
            mu.append(np.mean(v) if v else np.nan)
            sd.append(np.std(v) if v else 0.0)
        xs = np.arange(len(scenes)) + (i - (n - 1) / 2) * w
        ax.errorbar(xs, mu, yerr=sd, fmt=MARKERS[i], color=COLORS[i], ms=5, elinewidth=1, capsize=0,
                    label=METHOD_LABELS[m])
    # layout bands
    layouts = []
    for j, (L, _r) in enumerate(scenes):
        if not layouts or layouts[-1][0] != L:
            layouts.append([L, j, j])
        layouts[-1][2] = j
    for b, (L, a, z) in enumerate(layouts):
        if b % 2 == 0:
            ax.axvspan(a - 0.5, z + 0.5, color="#f2f1ee", zorder=0)
        ax.text((a + z) / 2, 1.0, L, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=11, color=TEXT)
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels([f"r{r}" for _L, r in scenes], fontsize=11)
    ax.set_xlim(-0.6, len(scenes) - 0.4)
    ax.set_ylabel("RMSE [m] (mean, std over\nsubsets of all-but-one robot)")
    ax.legend(frameon=False, fontsize=11, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_layouts(truth, runs, path, arena):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.4))
    axes = np.atleast_1d(axes)
    allxy = np.array([b["xy"] for run, _ in runs for b in truth[run]])
    cx, cy = allxy.mean(axis=0)
    half = max(1.0, np.abs(allxy - [cx, cy]).max() + 0.4)
    for ax, (run, name) in zip(axes, runs):
        for b in truth[run]:
            ax.add_patch(Rectangle((b["xy"][0] - 0.1, b["xy"][1] - 0.1), 0.2, 0.2, color="#8c564b"))
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=12, color=TEXT)
        x0, y0 = cx - half + 0.15, cy - half + 0.15
        ax.plot([x0, x0 + 0.5], [y0, y0], color=MUTED, lw=2)
        ax.text(x0 + 0.25, y0 + 0.06, "0.5 m", ha="center", fontsize=11, color=MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ====================================================================

def summarize(rows, methods, key, c=0.5):
    out = []
    full = [r for r in rows if r["k"] == r["n_robots"]]
    groups = by_order({r[key] for r in full}) if key else ["all"]
    for g in groups:
        for m in methods:
            rr = [r for r in full if r["method"] == m and (not key or r[key] == g)]
            v = np.array([r["rmse"] for r in rr if not math.isnan(r["rmse"])])
            out.append(dict(group=g, method=m, scenes=len(rr), **detection_metrics(rr, c),
                            rmse_median=float(np.median(v)) if len(v) else np.nan,
                            rmse_p90=float(np.percentile(v, 90)) if len(v) else np.nan,
                            rmse_mean=float(np.mean(v)) if len(v) else np.nan,
                            exact_pct=100.0 * np.mean([r["dn"] == 0 for r in rr]),
                            mean_abs_dn=float(np.mean([abs(r["dn"]) for r in rr])),
                            ospa_mean=float(np.mean([r["ospa"] for r in rr]))))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


UNION = METHODS["union"]


def given(flag):
    """True if the option was written on the command line (also as --flag=value)."""
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def apply_sensor_calib(a):
    """Fill the sensor settings from --sensor-calib where the command line did not set them.
    Returns the union clustering threshold [m]."""
    if not a.sensor_calib:
        return a.union_threshold if a.union_threshold is not None else 0.5
    with open(a.sensor_calib) as f:
        calib = json.load(f)
    if a.range_fit is None and calib.get("range_fit"):
        a.range_fit = list(calib["range_fit"])
    if a.bearing_fit is None and calib.get("bearing_fit"):
        a.bearing_fit = list(calib["bearing_fit"])
    if not given("--sigma-pos") and calib.get("sigma_pos"):
        a.sigma_pos = float(calib["sigma_pos"])
    if not given("--p-d") and calib.get("p_d"):
        a.p_d = float(calib["p_d"])
    if not given("--camera-offset") and calib.get("camera_offset") is not None:
        a.camera_offset = float(calib["camera_offset"])   # the fits above hold for this camera offset
    return a.union_threshold if a.union_threshold is not None else 3.0 * a.sigma_pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--regions", default=None)
    ap.add_argument("--out", default="results/evaluation/")
    ap.add_argument("--range-fit", type=float, nargs=2, default=None, metavar=("A", "B"),
                    help="distance correction measured = A * true + B")
    ap.add_argument("--bearing-fit", type=float, nargs=2, default=None, metavar=("K", "B0"),
                    help="bearing correction measured = K * true + B0 (see pipeline/calibrate.py)")
    ap.add_argument("--camera-offset", type=float, default=0.0, metavar="M",
                    help="camera ahead of the body pivot [m] (see pipeline/calibrate.py); only for runs saved with "
                         "survey.py --camera-offset-forward-m 0")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--cutoff", type=float, default=0.5, help="matching / OSPA cut-off c [m]")
    ap.add_argument("--ospa-p", type=float, default=2.0, help="OSPA order p")
    ap.add_argument("--full-only", action="store_true",
                    help="fuse only all the robots of each scene (no subsets, no robot-count figures); "
                         "for large simulated sets")
    ap.add_argument("--quiet", action="store_true", help="no line per scene")
    ap.add_argument("--names", nargs="*", default=[], metavar="LAYOUT=NAME",
                    help="names of the layouts in the figures, e.g. L1=Line L2=Square")
    ap.add_argument("--sensor-calib", default=None, metavar="JSON",
                    help="sensor calibration written by pipeline/calibrate.py --out (range and bearing fit, "
                         "sigma_pos, p_d); explicit command-line values win")
    ap.add_argument("--union-threshold", type=float, default=None, metavar="M",
                    help="clustering distance of union [m] (default: 3 * sigma_pos with --sensor-calib, "
                         "else 0.5)")
    add_sensor_arguments(ap)
    a = ap.parse_args()
    union_threshold = apply_sensor_calib(a)
    METHODS["union"] = functools.partial(UNION, threshold_m=union_threshold)
    print(f"sensor settings: range fit {a.range_fit}, bearing fit {a.bearing_fit}, sigma_pos {a.sigma_pos:.3f} m, "
          f"p_d {a.p_d:.2f}, union threshold {union_threshold:.3f} m, camera offset {a.camera_offset:.3f} m")
    analyze_rounds.BEARING_FIT = tuple(a.bearing_fit) if a.bearing_fit else None
    analyze_rounds.CAMERA_OFFSET = a.camera_offset
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13})

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    arena = arena_from_regions(a.regions) if a.regions else ARENA_DEFAULT
    sensor = sensor_from_arguments(a, arena)
    zones = zones_from_arguments(a, arena)
    truth = load_truth(a.ground_truth)
    points = bind_run_truth(load_points(a.survey_dir, a.range_fit), truth)
    points = {k: v for k, v in points.items() if v["round"] is not None}
    names = dict(n.split("=", 1) for n in a.names)
    for v in points.values():
        v["layout_name"] = names.get(v.get("layout_name", v["layout"]), v.get("layout_name", v["layout"]))
    for v in points.values():
        ORDER.setdefault(v["layout_name"], v["layout"])
    os.makedirs(a.out, exist_ok=True)

    rows = evaluate(points, truth, sensor, zones, methods, a.cutoff, a.ospa_p, full_only=a.full_only, quiet=a.quiet)
    write_csv(os.path.join(a.out, "scenes.csv"), rows)
    summ = summarize(rows, methods, None, a.cutoff) + summarize(rows, methods, "layout", a.cutoff)
    write_csv(os.path.join(a.out, "summary.csv"), summ)

    print("\n=== all robots fused, per method (all scenes) ===")
    print(f"  {'method':<14}{'scenes':>7}{'P_D':>8}{'RMSE*':>8}{'false':>8}{'exact':>8}{'OSPA':>7}")
    for s in summ:
        if s["group"] == "all":
            print(f"  {METHOD_LABELS[s['method']]:<14}{s['scenes']:>7}{s['pd_pct']:>7.1f}%{s['rmse_all_cm']:>6.1f}cm"
                  f"{s['false_per_scene']:>8.2f}{s['exact_pct']:>7.0f}%{s['ospa_mean']:>7.3f}")
    print(f"  P_D = true boxes found; RMSE* = over all true boxes, a missed box = {a.cutoff:g} m; "
          f"false = estimates with no true box, per scene")

    fig_cdf(rows, methods, os.path.join(a.out, "cdf_rmse.png"), names)
    fig_rmse_by_layout(rows, methods, os.path.join(a.out, "rmse_by_layout.png"))
    fig_count(rows, methods, os.path.join(a.out, "count_error.png"))
    if not a.full_only:                     # these need the robot subsets / one panel per run
        fig_vs_robots(rows, methods, os.path.join(a.out, "rmse_vs_robots.png"))
        fig_per_scene(rows, methods, os.path.join(a.out, "rmse_per_scene.png"))
        runs = sorted({(v["layout"], v["layout_name"]) for v in points.values()})
        fig_layouts(truth, [(r, n) for r, n in runs if r in truth], os.path.join(a.out, "layouts.png"), arena)
    print(f"\nwritten in {a.out}: scenes.csv, summary.csv and the figures")


if __name__ == "__main__":
    main()