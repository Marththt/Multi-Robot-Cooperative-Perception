"""Tables of the thesis on the real scenes: every method on the same 444 scenes, split the same way.

Each method is given as NAME=CSV[,CSV...]: per-scene results with either `layout` and `round` columns
(pipeline/evaluate_runs, gnn/train_sim, yolo_gnn/evaluate) or a `run` column like `R01_r0` (bevnet/train). Rows of the same scene
(several seeds, frames or methods filtered out) are averaged, so a scene counts once with the mean of its runs.
For pipeline/evaluate_runs results, pick the method with NAME=CSV@method (e.g. mht=scenes.csv@mht_sensor).

The scenes are split by group of layouts, number of boxes, distance between the two closest boxes and level
(tools/scene_levels.py), and every table is printed as LaTeX rows: correct scenes [%], and too few / too many.

    python -m tools.real_tables --levels results/scene_levels.csv \\
        --method "MHT=results/evaluation_all/scenes.csv@mht_sensor" \\
        --method "marker network=results/plan_v2/M4/gnn/detection_full_full_h2/real_all_scenes.csv" ...
"""

import argparse
import itertools
import math

import numpy as np
import pandas as pd

GROUP = {"L": "base", "A": "base", "B": "distance", "C": "obstacles", "D": "white objects", "E": "light",
         "M": "crowded"}
GROUPS = ["base", "distance", "obstacles", "white objects", "light", "crowded"]


def load(spec):
    name, _, paths = spec.partition("=")
    method = None
    if "@" in paths:
        paths, method = paths.rsplit("@", 1)
    frames = []
    for p in paths.split(","):
        d = pd.read_csv(p)
        if method is not None:
            d = d[d.method == method]
        if "k" in d.columns and "n_robots" in d.columns:
            d = d[d.k == d.n_robots]
        if "layout" not in d.columns or "round" not in d.columns:
            m = d.run.astype(str).str.extract(r"^(?P<layout>[A-Z]+\d+)\w*?_r(?P<round>\d+)$")
            d = d.assign(layout=m.layout, round=m["round"].astype(int))
        if "correct" not in d.columns:
            d = d.assign(correct=((d.dn == 0) & (d.n_matched == d.n_true)).astype(float))
        rep = d["seed"].astype(str) if "seed" in d.columns else (d["set"].astype(str) if "set" in d.columns else "0")
        frames.append(d.assign(rep=rep)[["layout", "round", "rep", "correct", "dn"]])
    d = pd.concat(frames)
    d = d.assign(few=(d.dn < 0).astype(float), many=(d.dn > 0).astype(float))
    REPS[name] = d.pivot_table(index=["layout", "round"], columns="rep", values="correct")
    return name, d.groupby(["layout", "round"])[["correct", "few", "many"]].mean()


REPS = {}      # per method: scenes x replicates (seeds, or frames) of `correct`


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired 0/1 outcomes: (only a right, only b right, p)."""
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p


def pairs_report(a, b, index):
    """McNemar of method a against b, replicate i against replicate i (a single replicate is reused)."""
    A, B = REPS[a].reindex(index), REPS[b].reindex(index)
    out = []
    for i in range(max(A.shape[1], B.shape[1])):
        x, y = A.iloc[:, min(i, A.shape[1] - 1)], B.iloc[:, min(i, B.shape[1] - 1)]
        out.append(mcnemar(x.round(), y.round()))
    return out


def oracle(names, index):
    """Correct scenes [%] of an oracle that takes, scene by scene, a method that is right; every combination of
    replicates is averaged."""
    mats = [REPS[n].reindex(index).to_numpy() for n in names]
    best = None
    for combo in itertools.product(*[range(m.shape[1]) for m in mats]):
        v = np.max(np.stack([m[:, j] for m, j in zip(mats, combo)]), axis=0)
        best = v if best is None else np.vstack([best, v])
    return float(np.atleast_2d(best).mean() * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="results/scene_levels.csv")
    ap.add_argument("--method", action="append", required=True, help="NAME=CSV[,CSV...][@method]")
    ap.add_argument("--errors", action="store_true", help="also print too few / too many per method")
    ap.add_argument("--pair", action="append", default=[],
                    help="A:B, McNemar of method A against B (all, and by level)")
    ap.add_argument("--oracle", action="append", default=[], help="A:B[:C], oracle over these methods by level")
    a = ap.parse_args()
    lv = pd.read_csv(a.levels)
    lv["group"] = lv.block.fillna("L").astype(str).str[0].map(GROUP)
    lv["boxes"] = lv.n_boxes.map(lambda v: "1" if v < 3 else ("3--4" if v <= 4 else ("5--6" if v <= 6 else "7")))
    lv["gap"] = np.where(lv.gap_m < 0.25, "closest centres $<25$~cm", "closest centres $\\ge25$~cm")
    lv["cause"] = np.where(lv.min_clear_views == 0, "a box in clear view of no robot", "")
    lv = lv.set_index(["layout", "round"])
    methods = [load(m) for m in a.method]
    for name, d in methods:
        missing = len(lv) - len(d.index.intersection(lv.index))
        print(f"% {name}: {len(d)} scenes, {missing} of the {len(lv)} real scenes missing")
    splits = [("group", GROUPS), ("boxes", ["3--4", "5--6", "7"]), ("gap", None), ("level", ["easy", "medium", "hard"]),
              ("cause", ["a box in clear view of no robot"])]
    for key, order in splits:
        order = order or sorted(lv[key].unique(), key=lambda v: "<" not in v)      # gap: < first
        for o in order:
            sel = lv[lv[key] == o]
            label = f"{o} boxes" if key == "boxes" else o
            vals = []
            for _, d in methods:
                x = d.reindex(sel.index)
                v = f"{x.correct.mean() * 100:.1f}"
                if a.errors:
                    v += f" & {x.few.mean() * 100:.1f} & {x.many.mean() * 100:.1f}"
                vals.append(v)
            print(f"{label} & {len(sel)} & {sel.reset_index().layout.nunique()} & " + " & ".join(vals) + "\\\\")
        print("\\midrule")
    vals = []
    for _, d in methods:
        x = d.reindex(lv.index)
        v = f"{x.correct.mean() * 100:.1f}"
        if a.errors:
            v += f" & {x.few.mean() * 100:.1f} & {x.many.mean() * 100:.1f}"
        vals.append(v)
    print(f"all & {len(lv)} & {lv.reset_index().layout.nunique()} & " + " & ".join(vals) + "\\\\")
    for pr in a.pair:
        x, y = pr.split(":")
        for lab, idx in [("all", lv.index)] + [(o, lv[lv.level == o].index) for o in ("easy", "medium", "hard")]:
            r = pairs_report(x, y, idx)
            print(f"% {x} vs {y}, {lab}: " + "; ".join(f"only {x} {u}, only {y} {v}, p={p:.3g}" for u, v, p in r))
    for orc in a.oracle:
        names = orc.split(":")
        rows = [("all", lv.index)] + [(o, lv[lv.level == o].index) for o in ("easy", "medium", "hard")] + \
               [(o, lv[lv.gap == o].index) for o in sorted(lv.gap.unique())] + \
               [("no clear view", lv[lv.min_clear_views == 0].index)]
        print(f"% oracle {' + '.join(names)}: " + "; ".join(f"{lab} {oracle(names, idx):.1f}" for lab, idx in rows))


if __name__ == "__main__":
    main()
