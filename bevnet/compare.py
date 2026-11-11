"""Correct scenes of the image-based network next to MHT and the GNN, set by set (same scenes, same metric).

Each --method is NAME=summary.csv: a summary of bevnet/train.py, pipeline/plan_mht.py (the rows with H = --h,
order 0) or gnn/train_sim.py (the GNN rows, mean over seeds). Writes table_BEV.csv (one row per test set, one
column per method) and fig_BEV-1 (correct scenes per scenario level, circle and random).

    python -m bevnet.compare --method "MHT (measurements)=results/plan/M3/summary.csv" \
        "GNN (measurements)=results/plan/M4/gnn/detection_full_full_h2/summary.csv" \
        "MHT (images, ArUco)=results/imgsim/M3/summary.csv" \
        "GNN (images, ArUco)=results/plan/M4/gnn/detection_full_full_h2/summary_imgsim.csv" \
        "Image network=results/bev/bev_c64_m16/summary.csv" --out results/bev/
"""

import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIOS = ("G0", "E3", "S1")


def read(path, h):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "method" in r and r["method"] != "gnn":
                continue
            if "H" in r and (int(float(r["H"])) != h or int(float(r.get("order", 0))) != 0):
                continue
            out[r.get("set") or r.get("test")] = float(r["correct_pct"])
    return out


def level_key(name):
    m = re.match(r"([A-Z]\d)_\D*(\d*)", name)
    return (m.group(1), int(m.group(2) or 0)) if m else (name, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", nargs="+", required=True, help="NAME=path/to/summary.csv")
    ap.add_argument("--h", type=int, default=100, help="MHT hypotheses (rows of plan_mht summaries)")
    ap.add_argument("--out", default="results/bev/")
    a = ap.parse_args()
    methods = {}
    for item in a.method:
        name, path = item.split("=", 1)
        methods[name] = read(path, a.h)
    sets = sorted({s for v in methods.values() for s in v if s.split("_")[0] in SCENARIOS},
                  key=lambda s: (level_key(s), s.endswith("random")))
    os.makedirs(a.out, exist_ok=True)
    rows = [dict(set=s, **{m: methods[m].get(s, "") for m in methods}) for s in sets]
    with open(os.path.join(a.out, "table_BEV.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set"] + list(methods))
        w.writeheader()
        w.writerows(rows)
    print(f"{'set':22s}" + "".join(f"{m[:20]:>22s}" for m in methods))
    for r in rows:
        print(f"{r['set']:22s}" + "".join(f"{r[m]:22.1f}" if r[m] != "" else f"{'-':>22s}" for m in methods))

    scen = [s for s in SCENARIOS if any(x.startswith(s + "_") for x in sets)]
    fig, axes = plt.subplots(2, len(scen), figsize=(4.2 * len(scen), 6.4), squeeze=False, sharey=True)
    for c, sid in enumerate(scen):
        for r, family in enumerate(("circle", "random")):
            ax = axes[r, c]
            levels = [s for s in sets if s.startswith(sid + "_") and s.endswith(family)]
            labels = [s[len(sid) + 1:-len(family) - 1] for s in levels]
            for m in methods:
                xs = [k for k, s in enumerate(levels) if s in methods[m]]
                ax.plot(xs, [methods[m][levels[k]] for k in xs], "o-", label=m)
            ax.set_xticks(range(len(levels)), labels)
            ax.set_title(f"{sid}, {family}")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("correct scenes [%]")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig_BEV-1.{ext}"), dpi=150)
    print(f"written in {a.out}: table_BEV.csv, fig_BEV-1")


if __name__ == "__main__":
    main()
