"""One table of every imgmlp variant: per test set the correct scenes (mean and sd over the seeds), too few / too
many boxes, and on the real tests the correct scenes per difficulty level (results/scene_levels.csv).

    python -m imgmlp.report --root results/imgmlp --out results/imgmlp/report.csv
"""

import argparse
import csv
import glob
import os

import numpy as np


def levels(path):
    with open(path) as f:
        return {(r["layout"], int(r["round"])): r["level"] for r in csv.DictReader(f)}


def by_level(scenes_csv, lv):
    """{level: correct %} over all seeds."""
    acc = {}
    with open(scenes_csv) as f:
        for r in csv.DictReader(f):
            key = lv.get((r["layout"], int(r["round"])))
            if key is not None:
                acc.setdefault(key, []).append(int(r["correct"]))
    return {k: round(100 * float(np.mean(v)), 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/imgmlp")
    ap.add_argument("--levels", default="results/scene_levels.csv")
    ap.add_argument("--out", default="results/imgmlp/report.csv")
    a = ap.parse_args()
    lv = levels(a.levels)
    rows = []
    for summary in sorted(glob.glob(os.path.join(a.root, "**", "summary.csv"), recursive=True)):
        folder = os.path.dirname(summary)
        with open(summary) as f:
            for r in csv.DictReader(f):
                row = dict(group=os.path.basename(os.path.dirname(folder)), variant=os.path.basename(folder), **r)
                scenes = os.path.join(folder, f"{r['test']}_scenes.csv")
                if r["test"].startswith("real") and os.path.exists(scenes):
                    row.update({f"{k}_pct": v for k, v in by_level(scenes, lv).items()})
                rows.append(row)
    keys = ["group", "variant", "test", "scenes", "correct_pct", "correct_sd", "too_few_pct", "too_many_pct",
            "easy_pct", "medium_pct", "hard_pct"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['group']:<12} {r['variant']:<34} {r['test']:<22} {float(r['correct_pct']):5.1f} "
              f"± {float(r['correct_sd']):3.1f}  few {float(r['too_few_pct']):5.1f}  many {float(r['too_many_pct']):5.1f}"
              + (f"  e/m/h {r.get('easy_pct', '-')}/{r.get('medium_pct', '-')}/{r.get('hard_pct', '-')}"
                 if r["test"].startswith("real") else ""))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
