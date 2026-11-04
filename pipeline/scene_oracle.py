"""Scene-level oracle between the MHT with sensor model and the detection-node GNN, on saved results.

For every test scene: correct for MHT, correct for the GNN (per seed), and the oracle that takes whichever
method is right. Prints the correct-scene rates and the 2 x 2 table (MHT right/wrong x GNN right/wrong),
averaged over the GNN seeds.

    python pipeline/scene_oracle.py results                       # T0-T7 and the real scenes
    python pipeline/scene_oracle.py results \
        results/gnn_circle_full/detection_full_range0.5to5_h2_local/test_scenes_comm3.2.csv C0,Cn3,Cn4,Cn8,Cn12,Cn16
"""
import csv, sys, statistics as st
from collections import defaultdict

R = sys.argv[1]

def correct(r):
    return int(r["dn"]) == 0 and int(r["n_matched"]) == int(r["n_true"])

def mht_rows(path):
    """Correctness of the MHT with sensor model with the whole team, by (run, round)."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["method"] == "mht_sensor" and r["k"] == r["n_robots"]:
            out[(r["run"], r["round"])] = correct(r)
    return out

gnn = defaultdict(dict)          # (test, seed) -> run -> correct
check = 0
for r in csv.DictReader(open(sys.argv[2] if len(sys.argv) > 2 else f"{R}/gnn_sim/detection_full_ring_h2/test_scenes.csv")):
    c = correct(r)
    check += c != bool(int(r["correct"]))
    gnn[(r["test"], r["seed"])][(r["run"], r["round"])] = c
print("definition mismatches:", check)

tests = sys.argv[3].split(",") if len(sys.argv) > 3 else [f"T{i}" for i in range(8)] + ["REAL"]
print("test  n   MHT   GNN   oracle  gain | MHT-ok/GNN-ok  MHT-ok/GNN-no  MHT-no/GNN-ok  MHT-no/GNN-no")
for t in tests:
    m = mht_rows(f"{R}/evaluation_calib/scenes.csv" if t == "REAL" else f"{R}/baselines_sim/{t}/scenes.csv")
    seeds = sorted(s for (tt, s) in gnn if tt == t)
    acc = defaultdict(list)
    for s in seeds:
        g = gnn[(t, s)]
        runs = [x for x in g if x in m]
        n = len(runs)
        cells = [sum(m[x] and g[x] for x in runs), sum(m[x] and not g[x] for x in runs),
                 sum(not m[x] and g[x] for x in runs), sum(not m[x] and not g[x] for x in runs)]
        acc["n"].append(n)
        acc["mht"].append(100 * sum(m[x] for x in runs) / n)
        acc["gnn"].append(100 * sum(g[x] for x in runs) / n)
        acc["orc"].append(100 * sum(m[x] or g[x] for x in runs) / n)
        for k, c in enumerate(cells):
            acc[f"c{k}"].append(100 * c / n)
    f = lambda k: st.mean(acc[k])
    print(f"{t:5s} {int(f('n')):4d} {f('mht'):5.1f} {f('gnn'):5.1f} {f('orc'):6.1f} {f('orc')-f('mht'):+5.1f} |"
          f" {f('c0'):12.1f} {f('c1'):14.1f} {f('c2'):14.1f} {f('c3'):14.1f}")
