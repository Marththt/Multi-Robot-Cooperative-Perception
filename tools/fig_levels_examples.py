"""Figure of the reports: one real scene per difficulty level (tools/scene_levels.py), seen from above.

Each box is coloured by the number of robots that have it in clear view (green: two or more, orange: one, red:
none); the dashed lines are the clear views. In a hard scene the cause is marked: the two boxes that almost
touch (red ring).

    python -m tools.fig_levels_examples --scene R03:1:easy --scene R81:0:medium --scene R08:0:hard \\
        --out results/figures/fig_levels_examples.png
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon

from pipeline.build_scans import best_runs
from tools.draw_robot import draw_robot
from tools.fig_scene_rounds import LEVEL_COLOUR, VIEW_COLOUR, seen_by
from tools.scene_levels import CIRCUMRADIUS_M, TOUCH_M, run_scenes, scene_level

WHY = {"easy": "every box seen clearly by 2+ robots,\nboxes at least 25 cm apart",
       "medium": "boxes seen clearly by a single robot\n(orange)",
       "hard": "{cause}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", required=True, metavar="LAYOUT:ROUND:LEVEL")
    ap.add_argument("--runs", default="data/survey_v2/*")
    ap.add_argument("--out", default="results/figures/fig_levels_examples.png")
    ap.add_argument("--half", type=float, default=1.65, help="half side of each panel [m]")
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(os.path.join(d, "raw")))
    fig, axes = plt.subplots(1, len(a.scene), figsize=(4.4 * len(a.scene), 5.6))
    for ax, spec in zip(np.atleast_1d(axes), a.scene):
        layout, rnd, expected = spec.split(":")
        run = best_runs([d for d in runs if os.path.basename(os.path.normpath(d)).split("_")[0] == layout])[0]
        name = os.path.basename(os.path.normpath(run))
        boxes, cams = run_scenes(run, name.rstrip("F"))
        with open(os.path.join(run, "ground_truth.json")) as f:
            gt = json.load(f)
        entries = gt.get(name.rstrip("F")) or next(iter(gt.values()))
        yaws = [b.get("yaw") or 0.0 for b in entries if b["class_name"] == "box"]
        cs = [cams[int(rnd)][q] for q in sorted(cams[int(rnd)])]
        level, gap, mv = scene_level(boxes, cs)
        if level != expected:
            print(f"{spec}: level is {level}, not {expected}")
        views = seen_by(boxes, cs)
        B = np.asarray(boxes, float)
        allxy = np.vstack([B, np.asarray(cs)[:, :2]])
        centre = (allxy.min(0) + allxy.max(0)) / 2
        for k, (bx, by) in enumerate(boxes):
            for n in views[k]:
                ax.plot([cs[n][0], bx], [cs[n][1], by], color="#2a78d6", lw=0.7, ls=(0, (3, 2)), alpha=0.6)
        for k, (bx, by) in enumerate(boxes):
            pts = [(bx + CIRCUMRADIUS_M * math.cos(yaws[k] + i * 2 * math.pi / 3),
                    by + CIRCUMRADIUS_M * math.sin(yaws[k] + i * 2 * math.pi / 3)) for i in range(3)]
            ax.add_patch(Polygon(pts, closed=True, fc=VIEW_COLOUR[min(len(views[k]), 2)], ec="#222", lw=0.9,
                                 zorder=4))
        cause = ""
        if level == "hard":
            if gap < TOUCH_M:
                i, j = min(((i, j) for i in range(len(B)) for j in range(i + 1, len(B))),
                           key=lambda p: np.hypot(*(B[p[0]] - B[p[1]])))
                mid = (B[i] + B[j]) / 2
                ang = math.degrees(math.atan2(*(B[j] - B[i])[::-1]))
                ax.add_patch(Ellipse(mid, gap + 0.42, 0.40, angle=ang, fill=False, ec="#c0392b", lw=2, ls="--",
                                     zorder=6))
                cause = f"two boxes {100 * gap:.0f} cm apart,\nalmost touching (red ring)"
            elif mv == 0:
                cause = "a box (red) seen clearly\nby no robot"
        for x, y, yaw in cs:
            draw_robot(ax, x, y, yaw)
        half = max(a.half, float(np.max(np.abs(allxy - centre))) + 0.35)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ccc")
        ax.set_title(level.capitalize(), loc="left", pad=46, fontsize=15, fontweight="bold",
                     color=LEVEL_COLOUR[level])
        ax.text(0.0, 1.02, WHY[level].format(cause=cause), transform=ax.transAxes, va="bottom", fontsize=11,
                color="#333")
        print(f"{spec}: {name}, level {level}, closest boxes {100 * gap:.0f} cm, fewest clear views {mv}")
    handles = [plt.Line2D([], [], marker="^", ls="", ms=10, mfc=VIEW_COLOUR[v], mec="#222", label=t)
               for v, t in ((2, "box in clear view of 2+ robots"), (1, "of 1 robot only"), (0, "of no robot"))]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=9, mfc="#d6e4f7", mec="#2a78d6", label="robot"))
    handles.append(plt.Line2D([], [], color="#2a78d6", lw=0.8, ls=(0, (3, 2)), label="clear view"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
