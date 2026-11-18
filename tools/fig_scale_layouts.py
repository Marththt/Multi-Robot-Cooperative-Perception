"""Scene 0 of the test sets SC_n8, SC_n24, SC_n50, regenerated with the seeds of sim/scaling.py (thesis figure).

    python -m tools.fig_scale_layouts [out.png]
"""
import argparse, math, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
from sim import scaling as S
from sim.geometry import box_polygon
from sim.model import SensorModel

a = argparse.Namespace(train_runs=20000, val_runs=2000, tune_runs=300, test_runs=200, seed=1)
table = {row[0]: row for row in S.set_table(a)}
model = SensorModel.load("config/sensor_model.json")
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, name in zip(axes, ["SC_n8", "SC_n24", "SC_n50"]):
    _, spec, _, seed, *_ = table[name]
    seed = a.seed * 1000 + seed
    m = S.run_model(model, spec)
    rng = np.random.default_rng(seed * 1_000_003 + 0)
    scene, meta, covered, side = S.area_scene(spec, m, rng, name, f"{name}_00000")
    half = m.hfov_deg / 2
    for c in scene.captures:
        x, y, yaw = c.pose
        ax.add_patch(Wedge((x, y), m.r_max_m, math.degrees(yaw) - half, math.degrees(yaw) + half,
                           color="tab:blue", alpha=0.06, lw=0))
        ax.add_patch(Polygon(S.robot_footprint(c.pose), closed=True, fc="tab:blue", ec="navy", lw=0.6))
    for j, (b, t) in enumerate(zip(scene.boxes, scene.thetas)):
        cov = j in covered
        ax.add_patch(Polygon(box_polygon(b, t), closed=True, fc="tab:orange" if cov else "white",
                             ec="saddlebrown" if cov else "gray", lw=0.8))
    h = side / 2
    ax.plot([-h, h, h, -h, -h], [-h, -h, h, h, -h], "k-", lw=0.8)
    ax.set_xlim(-h - 0.2, h + 0.2); ax.set_ylim(-h - 0.2, h + 0.2)
    ax.set_title(f"N = B = {meta['n_robots']}: side {side:.1f} m, {meta['n_covered']} of {meta['n_boxes']} boxes covered",
                 fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
fig.tight_layout()
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "results/scale/fig_scale_layouts.png", dpi=200)
print("ok", m.hfov_deg, m.r_max_m)
