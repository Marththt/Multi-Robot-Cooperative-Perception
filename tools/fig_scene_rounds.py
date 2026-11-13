"""Figure of the thesis: one layout and its rounds, each with its difficulty level (tools/scene_levels.py).

The boxes stay where they are for the whole capture; at every round the robots move to new positions, so every
round is a scene of its own and has its own level. Each box is coloured by the number of robots that have it in
clear view; the dashed lines are the clear views.

    python -m tools.fig_scene_rounds --run data/survey_v2_best/R58_20261111-150805 --out fig_scene_rounds.png
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge

from tools.scene_levels import CIRCUMRADIUS_M, HALF_FOV, MAX_RANGE_M, clear_views, run_scenes, scene_level

VIEW_COLOUR = {0: "#c0392b", 1: "#e08a1e", 2: "#2e7d4f"}          # clear views: none, one, two or more
LEVEL_COLOUR = {"easy": "#2e7d4f", "medium": "#e08a1e", "hard": "#c0392b"}


def seen_by(boxes, cams):
    """Per box, the cameras that have it in clear view (indices into cams)."""
    out = []
    for k in range(len(boxes)):
        out.append([n for n, c in enumerate(cams) if clear_views([boxes[k]] + [b for j, b in enumerate(boxes) if j != k],
                                                                 [c])[0]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="fig_scene_rounds.png")
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.run))
    boxes, cams = run_scenes(a.run, name.rstrip("F"))
    with open(os.path.join(a.run, "ground_truth.json")) as f:
        gt = json.load(f)
    yaws = [b.get("yaw") or 0.0 for b in next(iter(gt.values())) if b["class_name"] == "box"]
    rounds = sorted(cams)
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.3 * len(rounds), 3.9))
    for ax, rnd in zip(axes, rounds):
        names = sorted(cams[rnd])
        cs = [cams[rnd][q] for q in names]
        level, gap, mv = scene_level(boxes, cs)
        views = seen_by(boxes, cs)
        for x, y, yaw in cs:                                            # field of view, up to MAX_RANGE_M
            ax.add_patch(Wedge((x, y), MAX_RANGE_M, math.degrees(yaw - HALF_FOV), math.degrees(yaw + HALF_FOV),
                               color="#2a78d6", alpha=0.06, lw=0))
        for k, (bx, by) in enumerate(boxes):
            for n in views[k]:
                ax.plot([cs[n][0], bx], [cs[n][1], by], color="#2a78d6", lw=0.7, ls=(0, (3, 2)), alpha=0.7)
        for k, (bx, by) in enumerate(boxes):
            pts = [(bx + CIRCUMRADIUS_M * math.cos(yaws[k] + i * 2 * math.pi / 3),
                    by + CIRCUMRADIUS_M * math.sin(yaws[k] + i * 2 * math.pi / 3)) for i in range(3)]
            ax.add_patch(Polygon(pts, closed=True, fc=VIEW_COLOUR[min(len(views[k]), 2)], ec="#222", lw=0.8))
        for q, (x, y, yaw) in zip(names, cs):
            ax.plot(x, y, "o", ms=8, mfc="#d6e4f7", mec="#2a78d6", mew=1.4)
            ax.plot([x, x + 0.28 * math.cos(yaw)], [y, y + 0.28 * math.sin(yaw)], color="#2a78d6", lw=1.6)
        ax.set_title(f"Round {rnd + 1}: {level}", color=LEVEL_COLOUR[level], fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-2.1, 1.9)
        ax.set_ylim(-2.4, 1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ccc")
    handles = [plt.Line2D([], [], marker="^", ls="", ms=10, mfc=VIEW_COLOUR[v], mec="#222", label=t)
               for v, t in ((2, "box in clear view of 2+ robots"), (1, "of 1 robot only"), (0, "of no robot"))]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=8, mfc="#d6e4f7", mec="#2a78d6", label="robot"))
    handles.append(plt.Line2D([], [], color="#2a78d6", lw=0.8, ls=(0, (3, 2)), label="clear view"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(a.out, dpi=200)
    print(f"{name}: written {a.out}")


if __name__ == "__main__":
    main()
