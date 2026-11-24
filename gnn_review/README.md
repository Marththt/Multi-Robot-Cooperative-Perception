# gnn_review: experiments asked for in the review of the GNN architecture slides

Advisor comments of November 2026 (`gnn_architecture_comments.pdf`), points that need a new training:

| variant  | change w.r.t. the adopted model                                         | comment |
|----------|-------------------------------------------------------------------------|---------|
| base     | none: K = 2, H = 64, two heads, robots' centroid frame (reference)      | -       |
| k1       | K = 1 message-passing layer                                             | 10      |
| k3       | K = 3 message-passing layers                                            | 10      |
| nofalse  | pair head only (no false head, no false term in the loss)               | 10      |
| logdeg   | node input + log(1 + \|N(d)\|), N(d) = detections within d_c = 1 m     | 6       |
| world    | positions in the world frame of the plan, not relative to the robots   | 3       |

The adopted model is `results/plan_v2/M4/gnn/detection_full_full_h2`; `base` is the same training
(`gnn/train_sim.py` settings: data/plan_v2 train 50,000 / val 2,000 scenes, nominal sensor, Adam lr 1e-3, weight
decay 1e-4, batch 32, at most 40 epochs, early stop after 8, best epoch on validation correct scenes; class weights
once over the training set; decoding thresholds 0.5 / 0.5). It is retrained so that every variant is compared on the
same machine and the same library versions. With the weights of the adopted model `base` gives exactly its outputs.

Each variant is evaluated on the validation set and on the plan test sets G0, E1, E3, S1 (`data/plan_v2/`).
Everything needed (code, `data/plan_v2/train`, `val` and the test sets, `config/sensor_calib.json`) is in the
repository.

## Commands (from the repository root)

```bash
pip install torch numpy                   # if not already there (same environment as gnn/train_sim.py)

# quick check that everything runs (a few minutes; output in results/review_check/, can be deleted)
python -m gnn_review.train --variant logdeg --max-train 200 --max-val 100 --epochs 1 --seeds 1 --tests G0_nominal_circle --out results/review_check/

# everything: 6 variants x 3 seeds, then the table
bash gnn_review/run_all.sh                                     # Linux / macOS
powershell -ExecutionPolicy Bypass -File gnn_review\run_all.ps1   # Windows

# or one variant at a time (e.g. on several machines; copy the results/review/<variant>/ folders together)
python -m gnn_review.train --variant base --seeds 3
python -m gnn_review.train --variant k1 --seeds 3
python -m gnn_review.train --variant k3 --seeds 3
python -m gnn_review.train --variant nofalse --seeds 3
python -m gnn_review.train --variant logdeg --seeds 3
python -m gnn_review.train --variant world --seeds 3
python -m gnn_review.summarize                                  # results/review/comparison.csv and .md
```

Time: on a 10-core laptop CPU one epoch takes 1.5-3 min and a seed 20-40 epochs, so about 1 h per seed,
3 h per variant, 15-20 h for all six. `--threads N` limits the CPU threads; a run that is stopped restarts from the
seeds already saved (`model_seed<k>.pt`). `--eval-only` re-evaluates saved models.

## Outputs (results/review/)

- `<variant>/model_seed<k>.pt`, `train_log.csv` (loss and validation correct % per epoch), `variant.json`
- `<variant>/summary.csv`: per test set, mean and sd over the seeds of correct %, exact count %, P_D, false
  estimates per scene, RMSE*; `test_scenes.csv`: one row per scene and seed
- `comparison.csv`, `comparison.md`: all variants side by side (from `python -m gnn_review.summarize`)

To bring the results back, commit `results/review/*/summary.csv`, `train_log.csv`, `variant.json` and
`comparison.*` (the models are small, about 250 kB each, and can be committed too).
