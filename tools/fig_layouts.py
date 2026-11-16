"""Figure of the thesis: one real layout per group, as acquired.

Boxes are drawn as their triangular footprint at the position and orientation measured by OptiTrack, robots
(tools/draw_robot.py) at their camera poses of the first round, obstacles and white objects at their planned
positions (they are not tracked).

    python -m tools.fig_layouts --out fig_layouts.png
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from tools.draw_robot import draw_robot
from tools.scene_levels import CIRCUMRADIUS_M, run_scenes

EXAMPLES = [("base", "R05"), ("distance", "R54"), ("obstacles", "R63"),
            ("white objects", "R96"), ("light", "R101"), ("crowded", "R85")]
PLANS = ["docs/capture_plan_v2.json", "docs/capture_plan_v3.json"]


def triangle(x, y, yaw, r=CIRCUMRADIUS_M):
    return [(x + r * math.cos(yaw + i * 2 * math.pi / 3), y + r * math.sin(yaw + i * 2 * math.pi / 3))
            for i in range(3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey-dir", default="data/survey_all")
    ap.add_argument("--out", default="fig_layouts.png")
    a = ap.parse_args()
    plans = {}
    for p in PLANS:
        with open(p) as f:
            plans.update({l["id"]: l for l in json.load(f)})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    for ax, (group, lid) in zip(axes.flat, EXAMPLES):
        run = sorted(glob.glob(os.path.join(a.survey_dir, lid + "_*")))[0]
        _, cams = run_scenes(run, lid)
        with open(os.path.join(run, "ground_truth.json")) as f:
            gt = json.load(f)
        boxes = [b for b in (gt.get(lid) or next(iter(gt.values()))) if b["class_name"] == "box"]
        for kind, r in (("obstacle", 0.12), ("white", 0.08)):
            for o in plans[lid]["objects"]:
                if (kind == "obstacle") == (o[2] == "obstacle"):
                    if kind == "obstacle":
                        ax.add_patch(Polygon(triangle(o[0], o[1], math.pi / 2, 0.17), fc="#f2a65a", ec="#a0522d",
                                             lw=1.0, zorder=3))
                    else:
                        ax.add_patch(Circle((o[0], o[1]), r, fc="#f2a65a", ec="#a0522d", lw=1.0, zorder=3))
        for b in boxes:
            ax.add_patch(Polygon(triangle(b["x"], b["y"], b.get("yaw") or 0.0), fc="#eeeeee", ec="#222",
                                 lw=1.0, zorder=4))
        for x, y, yaw in cams[min(cams)].values():
            draw_robot(ax, x, y, yaw)
        xs = [b["x"] for b in boxes] + [c[0] for c in cams[min(cams)].values()]
        ys = [b["y"] for b in boxes] + [c[1] for c in cams[min(cams)].values()]
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        ax.set_xlim(cx - 1.95, cx + 1.95)
        ax.set_ylim(cy - 1.95, cy + 1.95)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ccc")
        ax.set_title(f"{group}: {len(boxes)} boxes", fontsize=12)
        ax.plot([cx + 0.9, cx + 1.9], [cy - 1.85, cy - 1.85], color="#444", lw=2)
        ax.text(cx + 1.4, cy - 1.78, "1 m", ha="center", va="bottom", fontsize=9, color="#444")
    handles = [plt.Line2D([], [], marker="^", ls="", ms=11, mfc="#eeeeee", mec="#222", label="box"),
               plt.Line2D([], [], marker="s", ls="", ms=10, mfc="#d6e4f7", mec="#2a78d6", label="robot"),
               plt.Line2D([], [], marker="^", ls="", ms=11, mfc="#f2a65a", mec="#a0522d", label="obstacle"),
               plt.Line2D([], [], marker="o", ls="", ms=10, mfc="#f2a65a", mec="#a0522d", label="white object")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(a.out, dpi=200)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
