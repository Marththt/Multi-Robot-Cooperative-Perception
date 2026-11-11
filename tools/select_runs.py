"""Copies one run per layout of a capture campaign into a clean folder, so that every pipeline (classical
methods, GNN, image network) reads the same scenes and none of the aborted or repeated attempts.

The run kept for a layout is the one of pipeline.build_scans.best_runs: among the runs with the most
stored frames, the one whose folder name ends in F (marked as the good capture), otherwise the latest.

    python -m tools.select_runs --src data/survey_v2/ --dst data/survey_v2_best/
"""

import argparse
import glob
import os
import shutil

from pipeline.build_scans import best_runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/survey_v2/")
    ap.add_argument("--dst", default="data/survey_v2_best/")
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(a.src, "*")) if os.path.isdir(os.path.join(d, "raw")))
    keep = best_runs(runs)
    os.makedirs(a.dst, exist_ok=True)
    for d in keep:
        name = os.path.basename(os.path.normpath(d))
        target = os.path.join(a.dst, name)
        if not os.path.isdir(target):
            shutil.copytree(d, target)
    print(f"{len(keep)} layouts from {len(runs)} runs -> {a.dst}")
    print(" ".join(os.path.basename(os.path.normpath(d)) for d in keep))


if __name__ == "__main__":
    main()
