"""Round-by-round analysis of a survey made with launch_all.py.

A ROUND is one formation: every robot standing at its stop of that round.
For every round (and every yaw offset, a separate scene: all the robots
turned by the same angle) the observations of ALL the robots of that
round are fused with each of the five methods, exactly as if the robots
were looking at the scene together:

  - one figure per scene: the robots (pose and field of view), what each
    of them measured, and for every method the estimated objects, their
    number and the OSPA against the true boxes;
  - rounds.csv: one row per scene and method (count, OSPA, ...);
  - a summary per method over all the scenes;
  - finally ONE OVERALL scene with all the captures of all the rounds
    together (every capture as a separate "robot"), with its figure.

    python -m pipeline.analyze_rounds --survey-dir data/survey/ --ground-truth config/ground_truth_positions.json \\
        --regions config/optitrack_regions.json --out results/rounds/

    # only the captures facing the centre, distances and bearings corrected (see pipeline/build_scans.py,
    # pipeline/calibrate.py)
    python -m pipeline.analyze_rounds ... --yaw-offsets 0 --camera-offset 0.12 --range-fit 1.064 -0.080

Data recorded without rounds (survey.py alone): the round of a capture is
its position in the robot's own sequence of stops.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

from pipeline.build_scans import ARENA_DEFAULT, bind_run_truth, load_points, load_truth
from vf.fusion.localization_methods import METHODS, measurement_from_observation, ospa, run_method
from vf.fusion.sensor_model import (add_sensor_arguments, arena_from_regions, sensor_from_arguments,
                                    zones_from_arguments)

METHOD_LABELS = dict(zone_class="zone + class", union="union", mht="MHT", phd="PHD", cphd="CPHD")

# Bearing correction (k, b0): measured = k * true + b0. Set by --bearing-fit (here or by
# pipeline/evaluate_runs.py); None = bearings used as measured. The distance correction (--range-fit)
# is applied by load_points() in pipeline/build_scans.py.
BEARING_FIT = None

# Camera ahead of the robot body pivot [m], set by --camera-offset. Use it for runs recorded with
# survey.py --camera-offset-forward-m 0 (the saved pose is the body pose); for runs recorded with the
# offset the saved pose is already the camera pose, leave it at 0.
CAMERA_OFFSET = 0.0


def camera_pose(pose):
    """Body pose (x, y, yaw) -> camera pose, moved CAMERA_OFFSET forward along the heading."""
    if pose is None or not CAMERA_OFFSET:
        return pose
    x, y, yaw = pose
    return (x + CAMERA_OFFSET * math.cos(yaw), y + CAMERA_OFFSET * math.sin(yaw), yaw)


def corrected_bearing(bearing):
    """Measured bearing -> corrected bearing (identity without --bearing-fit)."""
    if BEARING_FIT is None:
        return bearing
    k, b0 = BEARING_FIT
    return (bearing - b0) / k


def assign_rounds(points):
    """Points without a 'round' field: round = rank of the stop in the
    robot's own sequence (order of capture time)."""
    by_robot = defaultdict(list)
    for p in points.values():
        if p["round"] is None:
            by_robot[(p["layout"], p.get("run", ""), p["robot"])].append(p)
    for pts in by_robot.values():
        order = []
        for p in sorted(pts, key=lambda q: q["t"]):
            if p["stop"] not in order:
                order.append(p["stop"])
        for p in pts:
            p["round"] = order.index(p["stop"])


def travelled_track(points):
    """{robot: [(round, x, y), ...]}: where every robot actually was in
    every round (pose of its first capture of that round)."""
    best = {}
    for v in points.values():
        if v["round"] is None or v["pose"] is None:
            continue
        key = (v["robot"], v["round"])
        if key not in best or v["t"] < best[key]["t"]:
            best[key] = v
    track = {}
    for (rid, rnd), v in best.items():
        track.setdefault(rid, []).append((rnd, v["pose"][0], v["pose"][1]))
    return {rid: sorted(pts) for rid, pts in track.items()}


def fuse(members, points, sensor, zones, methods):
    """members: observation-point keys (one per 'robot' of the scene).
    Returns per_class measurements, poses and {class: {method: result}}."""
    poses, per_class = {}, {}
    for k in members:
        pt = points[k]
        poses[k] = camera_pose(pt["pose"])
        for ob in pt["observations"]:
            z = measurement_from_observation(camera_pose(ob["pose"]) or poses[k], ob["distance"],
                                             corrected_bearing(ob["bearing"]))
            per_class.setdefault(ob["class_name"], {}).setdefault(k, []).append(z)
    order = sorted(members)
    results = {}
    for c, reports in per_class.items():
        results[c] = {m: run_method(m, order, reports, poses, sensor, zones=zones) for m in methods}
    return poses, per_class, results


def round_line(rows, methods):
    """One console line: estimated/true count and OSPA per method."""
    cells = ""
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        cells += (f"{sum(x['n_est'] for x in r):>4}/{sum(x['n_true'] for x in r):<2} "
                  f"{np.mean([x['ospa'] for x in r]):.2f}").rjust(14)
    return cells


def scene_rows(name, layout, rnd, offset, members, points, truth, results, methods):
    rows = []
    classes = sorted({b["class_name"] for b in truth} | set(results))
    for c in classes:
        true_xy = [b["xy"] for b in truth if b["class_name"] == c]
        for m in methods:
            est = results.get(c, {}).get(m, {}).get("objects", [])
            rows.append(dict(scene=name, layout=layout, round=rnd, yaw_offset=offset, class_name=c, method=m,
                             robots=" ".join(sorted({points[k]["robot"] for k in members})),
                             n_views=len(members), n_true=len(true_xy), n_est=len(est),
                             exact=int(len(est) == len(true_xy)), ospa=ospa(est, true_xy)))
    return rows


def draw_scene(path, title, members, points, truth, poses, per_class, results, methods, sensor, arena,
               rows, keep_open=False, track=None, round_now=None):
    """Saves the figure of one scene in `path`; with keep_open the figure
    is returned (still open, e.g. to show it live) instead of closed.
    track: {robot: [(round, x, y), ...]} the whole path of every robot
    (planned or travelled), drawn faint with the round numbers, so that the
    positions of this scene (round_now) can be placed on it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    x0, x1, y0, y1 = arena
    n_panels = 1 + len(methods)
    cols = 3
    rows_n = int(math.ceil(n_panels / float(cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 4.6 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    robots = sorted({points[k]["robot"] for k in members} | set(track or {}))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rcol = {r: colors[i % len(colors)] for i, r in enumerate(robots)}
    many = len(members) > 12          # overall scene: no fields of view, small markers

    def draw_track(ax, labels):
        for rid, pts in (track or {}).items():
            if not pts:
                continue
            col = rcol[rid]
            ax.plot([p[1] for p in pts], [p[2] for p in pts], "-", color=col, lw=0.8, alpha=0.35, zorder=1)
            for rnd, px, py in pts:
                now = round_now is not None and rnd == round_now
                ax.plot(px, py, "o", ms=3, color=col, alpha=0.35 if not now else 0.9, zorder=1)
                if labels:
                    ax.text(px + 0.04, py + 0.04, f"r{rnd}", fontsize=6, color=col,
                            alpha=1.0 if now else 0.6, fontweight="bold" if now else "normal")

    def base(ax):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="0.5", lw=0.8))
        for b in truth:
            ax.add_patch(plt.Rectangle((b["xy"][0] - 0.1, b["xy"][1] - 0.1), 0.2, 0.2,
                                       color="0.75", zorder=1))
        for k in members:
            x, y, yaw = poses[k]
            col = rcol[points[k]["robot"]]
            ax.plot(x, y, "o", color=col, ms=4 if many else 7, zorder=3)
            ax.annotate("", xy=(x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw)), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8 if many else 1.2))
        ax.set_aspect("equal")
        ax.set_xlim(x0 - 0.1, x1 + 0.1)
        ax.set_ylim(y0 - 0.1, y1 + 0.1)
        ax.tick_params(labelsize=7)

    # panel 0: what the robots measured (and where this scene is on the whole path)
    ax = axes[0]
    base(ax)
    draw_track(ax, labels=True)
    for k in members:
        x, y, yaw = poses[k]
        col = rcol[points[k]["robot"]]
        if not many:
            ax.add_patch(Wedge((x, y), sensor.r_max, math.degrees(yaw - sensor.hfov / 2),
                               math.degrees(yaw + sensor.hfov / 2), color=col, alpha=0.07, zorder=0))
        for zs in (per_class[c].get(k, []) for c in per_class):
            for z in zs:
                ax.plot(z[0], z[1], ".", color=col, ms=5 if many else 8, zorder=4)
                if not many:
                    ax.plot([x, z[0]], [y, z[1]], color=col, lw=0.4, alpha=0.6, zorder=2)
    n_meas = sum(len(v) for c in per_class for v in per_class[c].values())
    ax.set_title(f"measurements ({n_meas} from {len(members)} views)"
                 + (f" - round {round_now} on the path" if round_now is not None and track else ""), fontsize=9)
    if not many:
        for r in robots:
            ax.plot([], [], "o", color=rcol[r], label=r)
        ax.legend(fontsize=7, loc="upper right")

    # one panel per method
    by_m = {r["method"]: r for r in rows}
    for i, m in enumerate(methods, start=1):
        ax = axes[i]
        base(ax)
        draw_track(ax, labels=False)
        for c in results:
            for (ox, oy) in results[c][m]["objects"]:
                ax.plot(ox, oy, "x", color="crimson", ms=10, mew=2.2, zorder=5)
        r = by_m.get(m)
        txt = (f"{METHOD_LABELS.get(m, m)}: {r['n_est']} / {r['n_true']} objects, OSPA {r['ospa']:.2f} m"
               if r else METHOD_LABELS.get(m, m))
        ax.set_title(txt, fontsize=9, color="black" if (r and r["exact"]) else "crimson")
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title + "   (grey squares: true boxes, red crosses: estimates)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    if keep_open:
        return fig
    plt.close(fig)


def main():
    global BEARING_FIT, CAMERA_OFFSET
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--survey-dir", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--regions", default=None)
    p.add_argument("--out", default="results/rounds/")
    p.add_argument("--layouts", nargs="*", default=None)
    p.add_argument("--yaw-offsets", type=float, nargs="*", default=None,
                   help="analyse only these yaw offsets (default: all, each one a separate scene)")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--range-fit", type=float, nargs=2, default=None, metavar=("A", "B"),
                   help="distance correction measured = A * true + B (see pipeline/build_scans.py)")
    p.add_argument("--bearing-fit", type=float, nargs=2, default=None, metavar=("K", "B0"),
                   help="bearing correction measured = K * true + B0 (see pipeline/calibrate.py)")
    p.add_argument("--camera-offset", type=float, default=0.0, metavar="M",
                   help="camera ahead of the body pivot [m] (see pipeline/calibrate.py); only for runs saved with "
                        "survey.py --camera-offset-forward-m 0")
    p.add_argument("--no-overall", action="store_true", help="skip the overall scene with all the rounds")
    p.add_argument("--no-figures", action="store_true")
    add_sensor_arguments(p)
    a = p.parse_args()
    BEARING_FIT = tuple(a.bearing_fit) if a.bearing_fit else None
    CAMERA_OFFSET = a.camera_offset
    import matplotlib
    matplotlib.use("Agg")

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    arena = arena_from_regions(a.regions) if a.regions else ARENA_DEFAULT
    sensor = sensor_from_arguments(a, arena)
    zones = zones_from_arguments(a, arena)
    truth = load_truth(a.ground_truth)
    points = bind_run_truth(load_points(a.survey_dir, a.range_fit), truth)
    if any(v["round"] is not None for v in points.values()):
        older = [k for k, v in points.items() if v["round"] is None]
        if older:
            print(f"{len(older)} capture(s) without round (older survey in the same folder): ignored")
            points = {k: v for k, v in points.items() if v["round"] is not None}
    assign_rounds(points)
    if a.yaw_offsets is not None:
        points = {k: v for k, v in points.items() if v["yaw_offset"] in a.yaw_offsets}
    layouts = sorted({v["layout"] for v in points.values()} & set(truth))
    if a.layouts:
        names = {v["layout"]: v.get("layout_name", v["layout"]) for v in points.values()}
        layouts = [l for l in layouts if l in a.layouts or names.get(l) in a.layouts]
    if not layouts:
        sys.exit("no data with a ground truth to analyse")
    os.makedirs(a.out, exist_ok=True)

    all_rows = []
    # every run (launch_all.py) is analysed on its own: its rounds are its scenes
    groups = sorted({(v["layout"], v.get("run", "")) for v in points.values() if v["layout"] in layouts})
    for layout, run in groups:
        lp = {k: v for k, v in points.items() if v["layout"] == layout and v.get("run", "") == run}
        tag = run or layout
        track = travelled_track(lp)
        scenes = defaultdict(list)
        for k, v in lp.items():
            scenes[(v["round"], v["yaw_offset"])].append(k)
        name_ = next(iter(lp.values()))["layout_name"]
        print(f"\n=== layout {name_}" + (f", run {run}" if run else "") + f": {len(truth[layout])} box(es)"
              + (" (read at the start of the run)" if layout == run else "") + ", "
              f"{len(scenes)} scene(s) (round x yaw offset)")
        print(f"  {'scene':<14}{'robots':<16}" + "".join(f"{METHOD_LABELS[m]:>14}" for m in methods))
        for (rnd, off) in sorted(scenes):
            members = sorted(scenes[(rnd, off)])
            name = f"{tag}_r{rnd:02d}_{off:+.0f}"
            poses, per_class, results = fuse(members, lp, sensor, zones, methods)
            rows = scene_rows(name, layout, rnd, off, members, lp, truth[layout], results, methods)
            for r in rows:
                r["run"] = run
            all_rows += rows
            if not a.no_figures:
                draw_scene(os.path.join(a.out, name + ".png"),
                           f"layout {layout}" + (f" ({run})" if run else "") + f", round {rnd}, "
                           f"yaw offset {off:+.0f} deg",
                           members, lp, truth[layout], poses, per_class, results, methods, sensor, arena, rows,
                           track=track, round_now=rnd)
            robots = " ".join(sorted({lp[k]["robot"] for k in members}))
            cells = ""
            for m in methods:
                r = [x for x in rows if x["method"] == m]
                n_est = sum(x["n_est"] for x in r)
                n_true = sum(x["n_true"] for x in r)
                cells += f"{n_est:>4}/{n_true:<2} {np.mean([x['ospa'] for x in r]):.2f}".rjust(14)
            print(f"  r{rnd:02d} {off:+4.0f} deg  {robots:<16}{cells}")

        if not a.no_overall:
            members = sorted(lp)
            name = f"{tag}_overall"
            poses, per_class, results = fuse(members, lp, sensor, zones, methods)
            rows = scene_rows(name, layout, "all", "all", members, lp, truth[layout], results, methods)
            for r in rows:
                r["run"] = run
            all_rows += rows
            if not a.no_figures:
                draw_scene(os.path.join(a.out, name + ".png"),
                           f"layout {layout}, ALL rounds together ({len(members)} captures)",
                           members, lp, truth[layout], poses, per_class, results, methods, sensor, arena, rows,
                           track=track)
            cells = ""
            for m in methods:
                r = [x for x in rows if x["method"] == m]
                cells += (f"{sum(x['n_est'] for x in r):>4}/{sum(x['n_true'] for x in r):<2} "
                          f"{np.mean([x['ospa'] for x in r]):.2f}").rjust(14)
            print(f"  {'ALL':<14}{'(every capture)':<16}{cells}")

    per_scene = [r for r in all_rows if r["round"] != "all"]
    print("\n=== summary over the scenes (each cell: exact count [%], mean OSPA [m]) ===")
    print("  " + "".join(f"{METHOD_LABELS[m]:>16}" for m in methods))
    print("  " + "".join(f"{100 * np.mean([r['exact'] for r in per_scene if r['method'] == m]):>8.0f}% "
                         f"{np.mean([r['ospa'] for r in per_scene if r['method'] == m]):>6.3f}" for m in methods))
    for m in methods:
        rr = [r for r in per_scene if r["method"] == m]
        over = sum(1 for r in rr if r["n_est"] > r["n_true"])
        under = sum(1 for r in rr if r["n_est"] < r["n_true"])
        print(f"  {METHOD_LABELS[m]:<14} too many {100 * over / len(rr):5.1f}%   too few {100 * under / len(rr):5.1f}%")

    path = os.path.join(a.out, "rounds.csv")
    fields = ["scene", "layout", "run", "round", "yaw_offset", "class_name", "method", "robots", "n_views",
              "n_true", "n_est", "exact", "ospa"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwritten {path}" + ("" if a.no_figures else f" and one figure per scene in {a.out}"))


if __name__ == "__main__":
    main()