#!/usr/bin/env bash
# Every variant of the review, 3 seeds each, then the comparison table. Run from the repository root:
#     bash gnn_review/run_all.sh            (log in results/review/run_all.log)
# A stopped run can be restarted: seeds already saved are not trained again.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/review
for v in base k1 k3 nofalse logdeg world; do
    python -m gnn_review.train --variant "$v" --seeds 3 2>&1 | tee -a results/review/run_all.log
done
python -m gnn_review.summarize | tee -a results/review/run_all.log
