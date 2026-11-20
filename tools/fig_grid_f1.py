"""Scene F1 over the robots x boxes grid (sim/scaling.py, sets G_n<N>_b<B>): one heat map per method, from the
summary.csv of yolo_gnn/evaluate_sim.py. A second figure gives the difference of every method from the MHT.

    python -m tools.fig_grid_f1 --summary results/scale_grid/summary.csv --out results/scale_grid/
"""

import argparse
import csv
import os
import re

import numpy as np

LABELS = dict(mht="MHT", marker="marker network", inject="+ YOLO boxes", cannot_link="+ YOLO cannot-link",
              full="YOLO + marker network")
SHOWN = [("mht", ""), ("marker", "scale"), ("full", "scale"), ("marker", "plan"), ("full", "plan")]


def read_grid(path, metric):
    """{(method, gnn): {(n_robots, n_boxes): (value, sd)}} of the G sets."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m = re.fullmatch(r"G_n(\d+)_b(\d+)", r["set"])
            if not m or r.get(metric, "") == "":
                continue
            out.setdefault((r["method"], r["gnn"]), {})[(int(m[1]), int(m[2]))] = (
                float(r[metric]), float(r.get(metric + "_sd") or 0.0))
    return out


def heat(ax, cells, robots, boxes, vmin, vmax, cmap, fmt):
    z = np.full((len(robots), len(boxes)), np.nan)
    for i, n in enumerate(robots):
        for j, b in enumerate(boxes):
            if (n, b) in cells:
                z[i, j] = cells[(n, b)]
    im = ax.imshow(z, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(robots)):
        for j in range(len(boxes)):
            if np.isfinite(z[i, j]):
                v = (z[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, fmt.format(z[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if (v < 0.45 if cmap == "viridis" else abs(v - 0.5) > 0.3) else "black")
    ax.set_xticks(range(len(boxes)), boxes)
    ax.set_yticks(range(len(robots)), robots)
    ax.set_xlabel("boxes")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/scale_grid/summary.csv")
    ap.add_argument("--metric", default="f1_mean")
    ap.add_argument("--out", default="results/scale_grid/")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = read_grid(a.summary, a.metric)
    if not grid:
        raise SystemExit(f"no G_n<N>_b<B> row with {a.metric} in {a.summary}")
    shown = [c for c in SHOWN if c in grid] + sorted(c for c in grid if c not in SHOWN and c[0] != "mht")
    shown = [c for c in shown if c in grid][:6]
    robots = sorted({n for c in grid.values() for n, _ in c})
    boxes = sorted({b for c in grid.values() for _, b in c})
    name = lambda c: LABELS.get(c[0], c[0]) + (f" ({c[1]})" if c[1] else "")

    # values
    fig, axes = plt.subplots(1, len(shown), figsize=(3.6 * len(shown) + 0.8, 3.6), squeeze=False)
    for ax, c in zip(axes[0], shown):
        im = heat(ax, {k: v for k, (v, _) in grid[c].items()}, robots, boxes, 0.0, 1.0, "viridis", "{:.2f}")
        ax.set_title(name(c), fontsize=9)
    axes[0][0].set_ylabel("robots")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="scene F1")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"fig_grid_{a.metric}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path)

    # difference from the MHT
    if ("mht", "") in grid:
        others = [c for c in shown if c[0] != "mht"]
        mht = {k: v for k, (v, _) in grid[("mht", "")].items()}
        diffs = {c: {k: v - mht[k] for k, (v, _) in grid[c].items() if k in mht} for c in others}
        lim = max([abs(v) for d in diffs.values() for v in d.values()] + [0.05])
        fig, axes = plt.subplots(1, len(others), figsize=(3.6 * len(others) + 0.8, 3.6), squeeze=False)
        for ax, c in zip(axes[0], others):
            im = heat(ax, diffs[c], robots, boxes, -lim, lim, "RdBu", "{:+.2f}")
            ax.set_title(name(c) + " - MHT", fontsize=9)
        axes[0][0].set_ylabel("robots")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="scene F1 difference")
        path = os.path.join(a.out, f"fig_grid_{a.metric}_vs_mht.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
