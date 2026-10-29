"""Collects the results of every GNN variant trained with gnn/train_sim.py into one comparison.

Reads results/gnn_sim/<variant>/summary*.csv and mcnemar*.csv of every variant (the main tests and, if
evaluated, the real scenes with combined rounds) and writes (in --out):
  comparison.csv   one row per test set: correct rate of the classical methods, mean and sd over the
                   seeds of every GNN variant, and McNemar p (largest over the seeds) of every variant vs MHT
  comparison.png   correct rate per test set: MHT, CPHD and the GNN variants (mean +- sd over the seeds)
  viewpoints.png   correct rate on the real scenes against the number of viewpoints (combined rounds)

    python -m gnn.collect --results-dir results/gnn_sim/

--metric exact uses the exact-count rate instead of "correct" (count right and every box matched).
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

CLASSICAL = ["zone_class", "union", "phd", "cphd", "mht"]
LABELS = dict(zone_class="Zone+class", union="Union", phd="PHD", cphd="CPHD", mht="MHT")
TEST_ORDER = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "REAL"] + [f"REALx{r}" for r in range(1, 8)]
TEST_NAMES = dict(T0="T0 nominal", T1="T1 close", T2="T2 dense", T3="T3 occluded", T4="T4 false det.",
                  T5="T5 two faces", T6="T6 unseen count", T7="T7 many robots", REAL="Real (L4-L11)",
                  **{f"REALx{r}": f"Real, {4 * r} views" for r in range(1, 8)})


def variant_label(name):
    """robot_full_ring_h2 -> 'robot / full / ring / 2 hops'."""
    parts = name.split("_")
    if len(parts) == 4 and parts[3].startswith("h"):
        return f"{parts[0]} / {parts[1]} / {parts[2]} / {parts[3][1:]} hop{'s' if parts[3] != 'h1' else ''}"
    return name


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/gnn_sim/")
    ap.add_argument("--out", default=None, help="default: --results-dir")
    ap.add_argument("--metric", choices=["correct", "exact"], default="correct")
    a = ap.parse_args()
    out = a.out or a.results_dir
    os.makedirs(out, exist_ok=True)
    key = f"{a.metric}_pct"

    variants = sorted({os.path.basename(os.path.dirname(p))
                       for p in glob.glob(os.path.join(a.results_dir, "*", "summary*.csv"))})
    if not variants:
        raise SystemExit(f"no */summary.csv in {a.results_dir}")

    classical = defaultdict(dict)               # test -> method -> rate
    gnn = defaultdict(dict)                     # test -> variant -> (mean, sd)
    p_mht = defaultdict(dict)                   # test -> variant -> largest p over the seeds
    viewpoints = {}                             # test -> mean number of viewpoints
    for v in variants:
        for path in sorted(glob.glob(os.path.join(a.results_dir, v, "summary*.csv"))):
            for r in read_csv(path):
                if r["method"] == "gnn":
                    gnn[r["test"]][v] = (num(r[key]), num(r.get(key + "_sd")))
                    if r.get("viewpoints"):
                        viewpoints[r["test"]] = num(r["viewpoints"])
                else:
                    classical[r["test"]].setdefault(r["method"], num(r[key]))
        ps = defaultdict(list)
        for path in sorted(glob.glob(os.path.join(a.results_dir, v, "mcnemar*.csv"))):
            for r in read_csv(path):
                if r["baseline"] == "mht":
                    ps[r["test"]].append(num(r["p_value"]))
        for t, vals in ps.items():
            p_mht[t][v] = max(vals)

    tests = [t for t in TEST_ORDER if t in gnn] + sorted(t for t in gnn if t not in TEST_ORDER)
    rows = []
    for t in tests:
        row = dict(test=t)
        for m in CLASSICAL:
            row[m] = round(classical[t][m], 1) if m in classical[t] else ""
        for v in variants:
            mean, sd = gnn[t].get(v, (math.nan, math.nan))
            row[f"{v}_mean"] = round(mean, 1) if math.isfinite(mean) else ""
            row[f"{v}_sd"] = round(sd, 1) if math.isfinite(sd) else ""
            p = p_mht[t].get(v)
            row[f"{v}_p_vs_mht"] = f"{p:.2g}" if p is not None else ""
        rows.append(row)
    with open(os.path.join(out, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{a.metric} rate [%]: MHT and every GNN variant (mean +- sd over the seeds; * = different from MHT, p < 0.05 for every seed)")
    width = max(18, max(len(variant_label(v)) for v in variants) + 2)
    print(f"  {'test':<16}{'MHT':>8}" + "".join(f"{variant_label(v):>{width}}" for v in variants))
    for t in tests:
        line = f"  {TEST_NAMES.get(t, t):<16}{classical[t].get('mht', math.nan):>8.1f}"
        for v in variants:
            mean, sd = gnn[t].get(v, (math.nan, math.nan))
            p = p_mht[t].get(v)
            star = "*" if p is not None and p < 0.05 else " "
            line += f"{f'{mean:.1f} +- {sd:.1f}{star}':>{width}}"
        print(line)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    series = [("MHT", "#d1495b", [(classical[t].get("mht", math.nan), 0.0) for t in tests]),
              ("CPHD", "#2bb3a3", [(classical[t].get("cphd", math.nan), 0.0) for t in tests])]
    blues = ["#1f4e9c", "#4a78c9", "#7fa3e0", "#b3c9f0", "#6a4c9c", "#9a7cc9", "#c6b3e6", "#3d3d3d"]
    for k, v in enumerate(variants):
        series.append((f"GNN {variant_label(v)}", blues[k % len(blues)],
                       [gnn[t].get(v, (math.nan, math.nan)) for t in tests]))
    bar_tests = [t for t in tests if not t.startswith("REALx")]
    keep = [tests.index(t) for t in bar_tests]
    series = [(label, color, [vals[i] for i in keep]) for label, color, vals in series]
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(bar_tests) + 0.25 * n * len(bar_tests)), 5.6))
    for k, (label, color, vals) in enumerate(series):
        x = [i - 0.4 + w * (k + 0.5) for i in range(len(bar_tests))]
        sds = [s if math.isfinite(s) else 0.0 for _, s in vals]
        ax.bar(x, [m for m, _ in vals], w, yerr=sds if any(sds) else None,
               color=color, label=label, capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(range(len(bar_tests)))
    ax.set_xticklabels([TEST_NAMES.get(t, t) for t in bar_tests], rotation=20, ha="right")
    ax.set_ylabel(f"{a.metric} [%]")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison.png"), dpi=150)
    plt.close(fig)
    written = "comparison.csv, comparison.png"

    combined = [t for t in tests if t.startswith("REALx")]
    if combined:
        x = [viewpoints.get(t, 4 * int(t[5:])) for t in combined]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for m, color in (("mht", "#d1495b"), ("cphd", "#2bb3a3")):
            ax.plot(x, [classical[t].get(m, math.nan) for t in combined], "o-", color=color, lw=1.8, ms=4,
                    label=LABELS[m])
        for k, v in enumerate(variants):
            mean = [gnn[t].get(v, (math.nan, math.nan))[0] for t in combined]
            sd = [gnn[t].get(v, (math.nan, math.nan))[1] for t in combined]
            color = blues[k % len(blues)]
            ax.plot(x, mean, "s-", color=color, lw=1.8, ms=4, label=f"GNN {variant_label(v)}")
            ax.fill_between(x, [m - (s if math.isfinite(s) else 0) for m, s in zip(mean, sd)],
                            [m + (s if math.isfinite(s) else 0) for m, s in zip(mean, sd)], color=color, alpha=0.15, lw=0)
        ax.set_xlabel("viewpoints per scene (real scenes, combined rounds)")
        ax.set_ylabel(f"{a.metric} [%]")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "viewpoints.png"), dpi=150)
        plt.close(fig)
        written += ", viewpoints.png"
    print(f"\nwritten in {out}: {written}")


if __name__ == "__main__":
    main()
