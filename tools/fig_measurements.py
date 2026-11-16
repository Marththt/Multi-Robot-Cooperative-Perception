"""Figure of the thesis: the real feature vectors against the OptiTrack ground truth, on all the real scenes.

Reads the tables written by `python -m sim.calibrate ... --dump DIR`: the error of every detection along and across
the line of sight, and every box in the field of view of a robot with its distance, the viewing angle of its best
face, whether that face is hidden by another box, and whether the box was detected.

    python -m tools.fig_measurements --dump results/sensor_model_dump --out measurements.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="measurements.png")
    a = ap.parse_args()
    e = pd.read_csv(os.path.join(a.dump, "detection_errors.csv"))
    b = pd.read_csv(os.path.join(a.dump, "boxes_in_view.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))

    edges = np.arange(0.25, 2.51, 0.25)
    mid = (edges[1:] + edges[:-1]) / 2
    for col, name, colour in (("along_m", "along the line of sight", "#2a78d6"),
                              ("across_m", "across the line of sight", "#e08a1e")):
        ax[0].scatter(e.distance_m, e[col] * 100, s=2, alpha=0.15, color=colour, lw=0)
        g = e.groupby(pd.cut(e.distance_m, edges), observed=False)[col]
        sd = g.std().to_numpy() * 100
        ok = g.size().to_numpy() >= 20
        ax[0].plot(mid[ok], sd[ok], "o-", color=colour, label=f"{name}, std")
    ax[0].axhline(0, color="#888", lw=0.8)
    ax[0].set_ylim(-20, 20)
    ax[0].set_xlabel("distance [m]")
    ax[0].set_ylabel("position error [cm]")
    ax[0].legend(fontsize=8.5, loc="lower left")

    free = b[b.occluded == 0]
    ang = np.degrees(free.angle_rad)
    aedges = np.array([0, 15, 30, 45, 60, 75, 90])
    g = free.groupby(pd.cut(ang, aedges), observed=False).detected
    ax[1].bar((aedges[1:] + aedges[:-1]) / 2, g.mean(), width=13, color="#2a78d6", alpha=0.85)
    for x, (m, n) in zip((aedges[1:] + aedges[:-1]) / 2, zip(g.mean(), g.size())):
        ax[1].text(x, m + 0.02, f"{m:.2f}", ha="center", fontsize=8.5)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_xlabel("viewing angle of the best face [deg]")
    ax[1].set_ylabel("share detected")
    ax[1].set_title("boxes in view, best face not hidden", fontsize=10)

    dedges = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0])
    for occ, name, colour in ((0, "best face not hidden", "#2a78d6"), (1, "every face hidden", "#c0392b")):
        s = b[b.occluded == occ]
        g = s.groupby(pd.cut(s.distance_m, dedges), observed=False).detected
        ok = g.size().to_numpy() >= 20
        x = ((dedges[1:] + dedges[:-1]) / 2)[ok]
        ax[2].plot(x, g.mean().to_numpy()[ok], "o-", color=colour, label=name)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlabel("distance [m]")
    ax[2].set_ylabel("share detected")
    ax[2].legend(fontsize=8.5)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
