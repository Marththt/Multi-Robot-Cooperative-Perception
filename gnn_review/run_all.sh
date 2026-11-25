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
# send the results back: commit results/review (models, logs, tables) and push to the current branch
git add results/review
git commit -m "results/review: GNN review variants, 3 seeds each (gnn_review/run_all.sh)"
git pull --rebase && git push || echo "push failed: run 'git pull --rebase && git push' by hand"
