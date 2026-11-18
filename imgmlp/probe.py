"""Linear probe: what the k image features encode.

For every frame of a cache, the features z written by imgmlp/train.py (features_<test>_seed<s>.npy) are regressed
linearly (ridge, standardized z, 5-fold cross-validation over scenes) on known quantities of the frame
(frames.csv): boxes in the field of view, hidden boxes, mean distance of the boxes in view, mean grey level, and in
simulation ambient light, floor and wall grey, camera pitch. The R^2 of each quantity says how much of it the
features carry; the weights of each feature (fitted on all frames) say which feature carries it.

    python -m imgmlp.probe --variant results/imgmlp/noisy_fixed_k8_64x36 --cache data/imgmlp/plan_v2_val --test val
    python -m imgmlp.probe --variant results/imgmlp/noisy_fixed_k8_64x36 --cache data/imgmlp/real --test real

Writes probe_<test>.csv in the variant folder (one row per seed and quantity: R^2, then the standardized weight of
every feature).
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

TARGETS = ["in_view", "hidden", "mean_dist_m", "grey", "ambient", "floor", "wall", "pitch_deg"]


def ridge(X, y, lam=1.0):
    Xb = np.column_stack([X, np.ones(len(X))])
    reg = lam * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def cv_r2(X, y, groups, folds=5, lam=1.0):
    uniq = np.unique(groups)
    fold_of = {g: i % folds for i, g in enumerate(np.random.default_rng(0).permutation(uniq))}
    f = np.array([fold_of[g] for g in groups])
    pred = np.zeros_like(y)
    for k in range(folds):
        tr, te = f != k, f == k
        if te.any() and tr.sum() > X.shape[1] + 1:
            w = ridge(X[tr], y[tr], lam)
            pred[te] = np.column_stack([X[te], np.ones(te.sum())]) @ w
    return 1 - float(np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    with open(os.path.join(a.cache, "frames.csv")) as f:
        rows = list(csv.DictReader(f))
    groups = np.array([f"{r['run']}/{r['round']}" for r in rows])
    out = []
    for path in sorted(glob.glob(os.path.join(a.variant, f"features_{a.test}_seed*.npy"))):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        z = np.load(path)
        Z = (z - z.mean(0)) / (z.std(0) + 1e-9)
        for t in TARGETS:
            vals = np.array([float(r[t]) if r.get(t) not in (None, "") else np.nan for r in rows])
            ok = np.isfinite(vals)
            if ok.sum() < 50 or np.std(vals[ok]) == 0:
                continue
            y = (vals[ok] - vals[ok].mean()) / vals[ok].std()
            r2 = cv_r2(Z[ok], y, groups[ok], lam=a.lam)
            w = ridge(Z[ok], y, a.lam)[:-1]
            out.append(dict(test=a.test, seed=seed, target=t, frames=int(ok.sum()), r2=round(r2, 3),
                            **{f"w_z{i}": round(float(v), 3) for i, v in enumerate(w)}))
            if seed == 0:
                top = np.argsort(-np.abs(w))[:3]
                print(f"  seed {seed} {t:<12} R^2 {r2:5.2f}   strongest features: "
                      + ", ".join(f"z{i} ({w[i]:+.2f})" for i in top))
    if not out:
        raise SystemExit(f"no features_{a.test}_seed*.npy in {a.variant}")
    path = os.path.join(a.variant, f"probe_{a.test}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"written {path}")


if __name__ == "__main__":
    main()
