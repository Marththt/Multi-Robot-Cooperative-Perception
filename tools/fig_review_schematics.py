"""Two schematic figures of the thesis (drawn geometry, not data):

fig_metrics.png            the four scene verdicts with the 15 cm cut-off, and the scene F1 of one example
fig_negative_evidence.png  what a missed detection means for MHT and for the graph network: robots that have a box
                           in view but report nothing near it lower its weight in MHT, while in the graph they have no
                           node within the 1 m gate, so no message reaches the report of the robot that saw it

    python -m tools.fig_review_schematics --out-dir results/figures
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

GOOD, BAD, EST, ROBOT, RAY, MUTE = "#2E7A4F", "#B23A2E", "#1E2D4F", "#2867d6", "#E07A1F", "#6B7683"
CUT = 0.15


def boxes(ax, truth, found):
    for (x, y), ok in zip(truth, found):
        ax.add_patch(Circle((x, y), CUT, fill=False, ec=GOOD if ok else BAD, lw=1.6, ls="-" if ok else "--"))
        ax.plot(x, y, "^", color="#9AA3AD", ms=7)


def ests(ax, pts, ok):
    for (x, y), good in zip(pts, ok):
        ax.plot(x, y, "x", color=EST if good else BAD, ms=10, mew=2.4)


def metrics(path):
    truth = [(0, 0), (0.6, 0.1), (0.3, 0.6), (-0.4, 0.5), (0.9, 0.7)]
    cases = [
        ("correct\n5 estimates, 5 matched", [(0.03, 0.02), (0.58, 0.14), (0.33, 0.57), (-0.42, 0.52), (0.88, 0.73)],
         [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
        ("too few\n4 estimates (two boxes merged)", [(0.03, 0.02), (0.6, 0.12), (-0.05, 0.55), (0.88, 0.73)],
         [1, 1, 0, 1], [1, 1, 0, 0, 1]),
        ("too many\n6 estimates (one false)", [(0.03, 0.02), (0.58, 0.14), (0.33, 0.57), (-0.42, 0.52),
                                              (0.88, 0.73), (1.2, -0.3)], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1]),
        ("misplaced\n5 estimates, one 25 cm off", [(0.03, 0.02), (0.58, 0.14), (0.33, 0.57), (-0.42, 0.52),
                                                 (1.12, 0.6)], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (title, pts, ok, found) in zip(axes, cases):
        boxes(ax, truth, found)
        ests(ax, pts, ok)
        nm, ne, nb = sum(ok), len(pts), len(truth)
        f1 = 2 * nm / (nb + ne)
        ax.set_title(title, fontsize=10)
        ax.text(0.25, -0.62, f"B = {nb}, B̂ = {ne}, matched {nm}\nF1 = 2·{nm}/({nb}+{ne}) = {f1:.2f}",
                ha="center", fontsize=9)
        ax.set_xlim(-0.8, 1.4)
        ax.set_ylim(-0.8, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    h = [plt.Line2D([], [], marker="^", ls="", color="#9AA3AD", label="true box centre"),
         plt.Line2D([], [], marker="o", ls="", mfc="none", mec=GOOD, label="15 cm cut-off, box matched"),
         plt.Line2D([], [], marker="o", ls="--", mfc="none", mec=BAD, color=BAD, label="box not matched"),
         plt.Line2D([], [], marker="x", ls="", color=EST, mew=2, label="estimate matched"),
         plt.Line2D([], [], marker="x", ls="", color=BAD, mew=2, label="estimate matched to no box")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)


def robot(ax, x, y, yaw, name, fov=True):
    if fov:
        ax.add_patch(Wedge((x, y), 3.0, math.degrees(yaw) - 44, math.degrees(yaw) + 44, color="#DCE6F4", alpha=0.45,
                           lw=0))
    ax.plot(x, y, "s", ms=11, mfc="#D9E3F1", mec=ROBOT, mew=1.5)
    ax.plot([x, x + 0.3 * math.cos(yaw)], [y, y + 0.3 * math.sin(yaw)], color=ROBOT, lw=2)
    ax.annotate(name, (x, y), xytext=(-6, -16), textcoords="offset points", fontsize=9, color=ROBOT)


def negative(path):
    R = [(-1.5, 0.0, 0.0, "robot 1"), (0.0, -1.5, math.pi / 2, "robot 2"), (1.5, 0.0, math.pi, "robot 3")]
    box = (0.0, 0.2)
    other = (0.0, 1.6)          # a box that robot 2 does report, far from the first
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for k, ax in enumerate(axes):
        for x, y, yaw, n in R:
            robot(ax, x, y, yaw, n)
        ax.plot(*box, "^", color="#9AA3AD", ms=12)
        ax.plot(*other, "^", color="#9AA3AD", ms=12)
        rep = (box[0] - 0.04, box[1] + 0.03)
        ax.plot(*rep, "o", color=EST, ms=8)
        rep2 = (other[0] + 0.03, other[1] - 0.04)
        ax.plot(*rep2, "o", color=MUTE, ms=8)
        ax.plot([-1.5, rep[0]], [0, rep[1]], color=EST, lw=1, ls=":")
        ax.plot([0, rep2[0]], [-1.5, rep2[1]], color=MUTE, lw=1, ls=":")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.0, 2.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_title("MHT: the silence of robots 2 and 3 is evidence", fontsize=11)
            for x, y, yaw, n in R[1:]:
                ax.annotate("", xy=(box[0] + 0.12 * (x - box[0]) / 1.5, box[1] + 0.12 * (y - box[1]) / 1.5),
                            xytext=(x, y), arrowprops=dict(arrowstyle="->", color=BAD, lw=1.5, ls="--"))
            ax.text(0.12, -1.0, "box in view of robots 2 and 3,\nno report: weight × (1 − p_D) twice",
                    fontsize=9, color=BAD)
        else:
            ax.set_title("Graph network: robots 2 and 3 send nothing", fontsize=11)
            ax.add_patch(Circle(rep, 1.0, fill=False, ec=MUTE, ls="--", lw=1.2))
            ax.text(rep[0] - 0.95, rep[1] + 1.05, "1 m gate: no other report,\nno edge, no message", fontsize=9,
                    color=MUTE)
            ax.text(0.12, -1.05, "a report with no neighbour:\nthe network judges it from its own\nfeatures and a prior learned in simulation",
                    fontsize=9, color=EST)
    h = [plt.Line2D([], [], marker="^", ls="", color="#9AA3AD", ms=9, label="true box"),
         plt.Line2D([], [], marker="o", ls="", color=EST, label="report of robot 1"),
         plt.Line2D([], [], marker="o", ls="", color=MUTE, label="report of robot 2 (another box)"),
         plt.Line2D([], [], color="#C9D7EC", lw=8, label="field of view, 88°, 3 m")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    metrics(os.path.join(a.out_dir, "fig_metrics.png"))
    negative(os.path.join(a.out_dir, "fig_negative_evidence.png"))
    print("figures in", a.out_dir)


if __name__ == "__main__":
    main()
