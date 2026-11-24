"""yolo_gnn/evaluate.py on the real scenes with the decoding thresholds chosen in simulation.

tools/grid_tune_thresholds.py chose the two thresholds of the marker network's decoding on the SIMULATED tune
set (chosen.json): t_false (a detection is dropped when p_false >= t_false) and t_link (clusters merge while their
average p_same > t_link). Here the real evaluation of yolo_gnn/evaluate.py is run unchanged except for those two
thresholds: gnn.model.decode is replaced, in the evaluate module only, by the same clustering with the two
thresholds. Nothing is chosen on the real data (the ray thresholds keep their two-fold cross-validation).

    python -m tools.real_tuned_thresholds --chosen results/scale_grid_tuned/chosen.json --net plan \\
        --gnn-dir results/plan_v2/M4/gnn/detection_full_full_h2 --mht results/evaluation_all/scenes.csv \\
        --out results/yolo_gnn_tuned/plan/

Every other option is passed to yolo_gnn/evaluate.py. --t-false / --t-link override chosen.json (0.5 0.5 gives
back the original evaluation).
"""

import argparse
import json
import sys

import numpy as np

from yolo_gnn import evaluate


def decode2(t_false, t_link):
    def decode(pos, pairs, p_same, p_false, threshold=None):
        pos = np.asarray(pos, float).reshape(-1, 2)
        n = len(pos)
        P = np.eye(n)
        for (i, j), p in zip(pairs, p_same):
            P[i, j] = P[j, i] = p
        keep = [i for i in range(n) if p_false[i] < t_false]
        clusters = []
        if keep:
            S = P[np.ix_(keep, keep)].copy()
            sizes = np.ones(len(keep))
            alive = np.ones(len(keep), bool)
            members = [[i] for i in keep]
            np.fill_diagonal(S, -np.inf)
            while alive.sum() > 1:
                link = S / np.outer(sizes, sizes)
                link[~alive, :] = -np.inf
                link[:, ~alive] = -np.inf
                a, c = np.unravel_index(np.argmax(link), link.shape)
                if link[a, c] <= t_link:
                    break
                a, c = min(a, c), max(a, c)
                S[a, :] += S[c, :]
                S[:, a] += S[:, c]
                S[a, a] = -np.inf
                sizes[a] += sizes[c]
                alive[c] = False
                members[a] += members[c]
            clusters = [members[k] for k in range(len(keep)) if alive[k]]
        assign = np.full(n, -1)
        for k, cl in enumerate(clusters):
            assign[cl] = k
        return [pos[cl].mean(axis=0) for cl in clusters], assign
    return decode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chosen", default="results/scale_grid_tuned/chosen.json")
    ap.add_argument("--net", default="plan", help="key of chosen.json")
    ap.add_argument("--t-false", type=float, default=None)
    ap.add_argument("--t-link", type=float, default=None)
    a, rest = ap.parse_known_args()
    with open(a.chosen) as f:
        ch = json.load(f)[a.net]
    tf = ch["t_false"] if a.t_false is None else a.t_false
    tl = ch["t_link"] if a.t_link is None else a.t_link
    print(f"decoding thresholds: t_false {tf}, t_link {tl}", flush=True)
    evaluate.decode = decode2(tf, tl)
    sys.argv = [sys.argv[0]] + rest
    evaluate.main()


if __name__ == "__main__":
    main()
