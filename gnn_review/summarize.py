"""One table with every variant of the review: results/review/<variant>/summary.csv -> results/review/comparison.csv
(and a Markdown copy, comparison.md). The adopted model is the 'base' row.

    python -m gnn_review.summarize
"""

import argparse
import csv
import glob
import os

ORDER = ["base", "k1", "k3", "nofalse", "logdeg", "world"]
COLS = ["correct_pct", "exact_pct", "pd_pct", "false_per_scene", "rmse_all_cm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/review/")
    a = ap.parse_args()
    rows = []
    for path in glob.glob(os.path.join(a.out, "*", "summary.csv")):
        with open(path) as f:
            rows += list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no summary.csv under {a.out}")
    rank = {v: i for i, v in enumerate(ORDER)}
    rows.sort(key=lambda r: (r["test"] != "VAL", r["test"], rank.get(r["variant"], 99)))
    keys = ["test", "variant", "seeds"] + [k for c in COLS for k in (c, c + "_sd")]
    with open(os.path.join(a.out, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| test | variant | correct % | exact % | P_D % | false / scene | RMSE* cm |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{float(r[c]):.2f} ± {float(r[c + '_sd']):.2f}" for c in COLS]
        lines.append(f"| {r['test']} | {r['variant']} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten {os.path.join(a.out, 'comparison.csv')} and comparison.md")


if __name__ == "__main__":
    main()
