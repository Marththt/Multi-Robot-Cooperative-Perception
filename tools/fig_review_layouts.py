"""Differences in correct scenes on the 444 real scenes with the LAYOUT as the unit (figure fig_layout_bootstrap.png).

Scenes of one layout share the boxes, so they are not independent. For each comparison (model A - MHT, model C - MHT,
model C - model A), the difference in the share of correct scenes is given with a 95% interval from a cluster
bootstrap that resamples whole layouts (10,000 resamples), and the layouts that favour each side are counted (sign
test on the layouts with a non-zero difference). Left: all scenes and each level, every seed of the graph network
on its own. Right: the difference of every layout (mean over the seeds), all scenes.

    python -m tools.fig_review_layouts --out results/figures/fig_layout_bootstrap.png
"""

import argparse
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["layout", "round"]


def sign_p(pos, neg):
    n = pos + neg
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(pos, neg) + 1)) / 2 ** n) if n else 1.0


def compare(x, y, rng, B=10000):
    j = x.join(y, lsuffix="_x", rsuffix="_y", how="inner")
    j["diff"] = j.correct_x - j.correct_y
    g = j.reset_index().groupby("layout")["diff"]
    s, n = g.sum().values, g.count().values
    L = len(s)
    boot = np.array([s[i].sum() / n[i].sum() for i in (rng.integers(0, L, L) for _ in range(B))])
    per = g.mean()
    pos, neg = int((per > 0).sum()), int((per < 0).sum())
    return dict(diff=100 * j["diff"].mean(), lo=100 * np.percentile(boot, 2.5), hi=100 * np.percentile(boot, 97.5),
                layouts=L, pos=pos, neg=neg, p=sign_p(pos, neg), per=100 * per)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/figures/fig_layout_bootstrap.png")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    m = pd.read_csv("results/evaluation_all/scenes.csv")
    m = m[m.method == "mht_sensor"]
    m["correct"] = ((m.dn == 0) & (m.n_matched == m.n_true)).astype(float)
    mht = m.groupby(KEY)[["correct"]].mean()
    A = pd.read_csv("results/yolo_gnn/test_scenes_marker.csv")
    C = pd.read_csv("results/yolo_gnn/test_scenes_full.csv")
    level = pd.read_csv("results/scene_levels.csv").set_index(KEY)["level"]

    def sel(df, lv):
        return df if lv == "all" else df[df.index.isin(level[level == lv].index)]

    rows = []
    for name, f in (("model A − MHT", lambda s: (A[A.seed == s].set_index(KEY)[["correct"]], mht)),
                    ("model C − MHT", lambda s: (C[C.seed == s].set_index(KEY)[["correct"]], mht)),
                    ("model C − model A", lambda s: (C[C.seed == s].set_index(KEY)[["correct"]],
                                                     A[A.seed == s].set_index(KEY)[["correct"]]))):
        for lv in ("all", "easy", "medium", "hard"):
            for seed in range(3):
                x, y = f(seed)
                r = compare(sel(x, lv), sel(y, lv), rng)
                rows.append(dict(comparison=name, level=lv, seed=seed, **{k: v for k, v in r.items() if k != "per"}))
                print(f"{name:18s} {lv:6s} seed {seed}: {r['diff']:5.1f} [{r['lo']:5.1f}, {r['hi']:5.1f}] "
                      f"{r['layouts']} layouts, +{r['pos']} -{r['neg']}, sign p={r['p']:.2g}")
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res.to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    colors = {"model A − MHT": "#C06A1E", "model C − MHT": "#1E2D4F", "model C − model A": "#6B7683"}
    y, ticks, labels = 0, [], []
    for lv in ("all", "easy", "medium", "hard"):
        for name in colors:
            for seed in range(3):
                r = res[(res.comparison == name) & (res.level == lv) & (res.seed == seed)].iloc[0]
                yy = y + (seed - 1) * 0.22
                ax.plot([r.lo, r.hi], [yy, yy], color=colors[name], lw=2)
                ax.plot(r["diff"], yy, "o", color=colors[name], ms=5)
            ticks.append(y)
            labels.append(f"{lv}: {name}")
            y -= 1
        y -= 0.5
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("difference in correct scenes [points], 95% layout bootstrap (one line per seed)")

    for k, name in enumerate(colors):
        per = []
        for seed in range(3):
            x, yv = {"model A − MHT": (A[A.seed == seed].set_index(KEY)[["correct"]], mht),
                     "model C − MHT": (C[C.seed == seed].set_index(KEY)[["correct"]], mht),
                     "model C − model A": (C[C.seed == seed].set_index(KEY)[["correct"]],
                                           A[A.seed == seed].set_index(KEY)[["correct"]])}[name]
            per.append(compare(x, yv, rng, B=1)["per"])
        per = pd.concat(per, axis=1).mean(axis=1).sort_values()
        jitter = (np.random.default_rng(k).random(len(per)) - 0.5) * 0.35
        bx.scatter(per.values, np.full(len(per), -k) + jitter, s=14, color=colors[name], alpha=0.7)
        pos, neg = int((per > 0).sum()), int((per < 0).sum())
        bx.text(102, -k, f"{pos} better, {neg} worse,\n{len(per) - pos - neg} equal", va="center", fontsize=8)
    bx.axvline(0, color="#999", lw=1)
    bx.set_yticks([0, -1, -2])
    bx.set_yticklabels(list(colors), fontsize=9)
    bx.set_xlim(-105, 160)
    bx.set_xticks([-100, -50, 0, 50, 100])
    bx.set_xlabel("difference of each layout [points], mean over seeds")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("figure:", a.out)


if __name__ == "__main__":
    main()
