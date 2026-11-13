"""One compact figure with the floor maps of several scenes, from the .npz written by tools/fig_image_report.py:
for each scene the map of the network on masks alone and of the network on masks + markers, side by side.

    python -m tools.fig_image_maps --scene results/figures/image_report/R81_r0_maps.npz "medium scene (Figure 3)" \\
        --scene results/figures/image_report/R96_r0_maps.npz "hard scene, white objects (Figure 5)" \\
        --out results/figures/image_report/maps_combined.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.fig_image_report import draw_map, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs=2, action="append", required=True, metavar=("NPZ", "TITLE"))
    ap.add_argument("--out", default="results/figures/image_report/maps_combined.png")
    a = ap.parse_args()
    n = len(a.scene)
    fig, axes = plt.subplots(1, 2 * n, figsize=(4.6 * 2 * n, 5.2))
    for k, (path, title) in enumerate(a.scene):
        d = np.load(path)
        boxes, yaws, cams = d["boxes"], d["yaws"], d["cams"]
        half = max(float(d["half"]), float(np.max(np.abs(cams[:, :2]))) + 0.55)
        for j, (key, label) in enumerate((("masks", "masks only"), ("markers", "masks + markers"))):
            ax = axes[2 * k + j]
            est = d[f"{key}_est"].reshape(-1, 2)
            draw_map(ax, d[f"{key}_heat"], est, boxes, yaws, cams, f"{label}:\n{verdict(est, boxes)}", half)
        axes[2 * k].text(0.0, 1.22, title, transform=axes[2 * k].transAxes, fontsize=15, fontweight="bold",
                         va="bottom")
    fig.tight_layout(w_pad=0.2)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", pad_inches=0.1)
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
