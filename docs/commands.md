# Commands

Run every command from the repository root. Scripts are run as modules (`python -m <package>.<script>`).

## Live trial (one terminal per robot) and live map

    python -m acquisition.static_trial --robot-id P0 --live --state-out data/live/P0.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 8 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P0.json
    python -m acquisition.static_trial --robot-id P1 --live --state-out data/live/P1.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 7 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json
    python -m acquisition.static_trial --robot-id P2 --live --state-out data/live/P2.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 9 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json
    python -m acquisition.static_trial --robot-id P3 --live --state-out data/live/P3.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 10 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json

    python -m acquisition.map_server --state-dir data/live/ --regions config/optitrack_regions.json --localize

## Record the box positions (OptiTrack)

    python -m tools.record_ground_truth --layout-id L4 --object 11:box --object 12:box --object 13:box --object 14:box --overwrite

## Move and capture (all robots, round mode)

    python -m acquisition.launch_all --robot P0:8 --robot P1:7 --robot P4:6:config/checkerboard_calib_P1.json --robot P3:10:config/checkerboard_calib_P1.json --layout-id L4 --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out data/survey_rounds/ --aruco-marker-size 0.077 --radius 1.3 --step-deg 15 --sweep-deg 90 --range-fit 1.064 -0.080 --camera-offset-forward-m 0.12 --instance-threshold-m 0.2

## Sensor calibration and evaluation of the classical methods

    python -m pipeline.calibrate data/survey_rounds/ --camera-offset 0 --out config/sensor_calib.json
    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --out results/evaluation_calib/

Older session (L1-L3, runs saved with the body pose: camera offset 0.12 m):

    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds_t02/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --camera-offset 0.13 --range-fit 1.070 -0.073 --out results/evaluation/ --names L1=Line L2=Square L3=Pairs

Scans and per-round analysis:

    python -m pipeline.build_scans --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out results/analysis/ --plots results/analysis/

## GNN

    python -m gnn.train --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --baseline results/evaluation_calib/scenes.csv --out results/evaluation_gnn/

Whole arrangement types left out of training:

    python -m gnn.train ... --folds "L4,L5,L6;L7,L8;L9,L10,L11" --out results/evaluation_gnn_groups/

## Simulator

Fidelity check: sensor model fitted on L5, L7, L10 only (rare events from all layouts), checked on L4, L6, L8, L9, L11:

    python -m sim.calibrate --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts L5,L7,L10 --rare-events all --out config/sensor_model_fidelity.json
    python -m sim.generate replay --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --model config/sensor_model_fidelity.json --layouts L4,L6,L8,L9,L11 --reps 200 --seed 1 --out data/sim_replay/
    python -m pipeline.evaluate_runs --survey-dir data/sim_replay/ --ground-truth data/sim_replay/ground_truth.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --full-only --quiet --out results/evaluation_sim_replay/
    python -m sim.fidelity --real-survey-dir data/survey_rounds/ --real-ground-truth config/ground_truth_positions.json --sim-dir data/sim_replay/ --sensor-calib config/sensor_calib.json --layouts L4,L6,L8,L9,L11 --real-scenes results/evaluation_calib/scenes.csv --sim-scenes results/evaluation_sim_replay/scenes.csv --out results/fidelity/

Sensor model for everything else (failure map, training, test conditions), fitted on all layouts:

    python -m sim.calibrate --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts L4,L5,L6,L7,L8,L9,L10,L11 --rare-events all --out config/sensor_model.json

Failure map of the classical methods (one factor changed at a time around the real setup; 500 scenes per value):

    python -m sim.sweep --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --runs 500 --seed 1 --out results/failure_map/

Only some factors: `--factors spacing,occlusion` (spacing, boxes, robots, distance, occlusion, false_rate, two_face).

New scenes from a scenario (training set, or a test condition T0-T7 in `sim/specs/`):

    python -m sim.generate random --model config/sensor_model.json --spec sim/specs/train_general.json --runs 1000 --name TRAIN --seed 1 --out data/sim_train/
    python -m sim.generate random --model config/sensor_model.json --spec sim/specs/T1_close.json --runs 1000 --name T1 --seed 11 --out data/sim_T1/

## GNN trained on the simulator

Needs PyTorch (`pip install torch`). Build every set (training 50,000 scenes, validation 2,000, T0-T7 1,000 each) and run the classical methods on T0-T7:

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json

Main model (robot nodes, all inputs, ring neighbours, 2 hops, 5 seeds), tested on T0-T7 and on the 56 real scenes:

    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ --sensor-calib config/sensor_calib.json --standard-tests --graph robot --features full --edges ring --hops 2 --out results/gnn_sim/

The other variants change only the last flags:

    --graph robot --features positions --edges ring --hops 2         (positions only)
    --graph detection --features full --edges ring --hops 2          (detection nodes)
    --graph detection --features positions --edges ring --hops 2
    --graph robot --features full --edges range --comm-range 1.5 --hops 1   (also --hops 2, --hops 3)
    --graph robot --features full --edges full --hops 1               (upper bound: every robot hears every robot)
    --graph robot --features full --edges opposite --hops 2           (each robot linked to the robot in front and the two diagonals)
    --graph robot --features full --edges full --hops 2               (every robot linked to every robot, 2 rounds)

Each variant writes `results/gnn_sim/<variant>/summary.csv` (GNN mean and sd over the seeds, classical methods on the same scenes) and `mcnemar.csv`. `--eval-only` re-evaluates saved models; `--max-train 2000 --epochs 3 --seeds 1` is a quick check that everything runs.

Real scenes with more viewpoints: the rounds of each real run are combined (4, 8, ... 28 viewpoints) and the classical methods are run on them; then every trained variant is evaluated on them (no training, writes `summary_combined.csv` next to `summary.csv`):

    python -m pipeline.combine_rounds --regions config/optitrack_regions.json
    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ --sensor-calib config/sensor_calib.json --combined-tests --eval-only --graph robot --features full --edges ring --hops 2 --out results/gnn_sim/

(same `--eval-only` command with the flags of every other variant)

All the variants in one table and one figure per metric (detection probability, RMSE over all true boxes with a missed box = 0.15 m, false estimates per scene): `results/gnn_sim/comparison_<metric>.csv/.png`, and `viewpoints_<metric>.png` for the combined real scenes:

    python -m gnn.collect --results-dir results/gnn_sim/

## Two-circle scenario (targets in an inner circle, robots on an outer circle, communication range)

Sets: training (3-10 robots), validation, the nominal test C0 (6 robots at 1.5 m, inner circle 0.7 m) and its variants with 3-16 robots (Cn*), with the classical methods:

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --sets circle

GNN whose robots exchange messages only within a communication range, drawn between 0.5 and 5 m for every training scene, with the three skip-connection variants:

    python -m gnn.train_sim --train-dir data/sim_Ctrain/ --val-dir data/sim_Cval/ --sensor-calib config/sensor_calib.json --circle-tests --edges range --comm-range-train 0.5 5 --comm-range 2.0 --skip residual --out results/gnn_circle/
    (same with --skip none and --skip jk)

The same models at other communication ranges (no training; 1 m: no neighbour, 2 m: the nearest ones, 3.2 m: everyone with 6 robots at 1.5 m):

    python -m gnn.train_sim --train-dir data/sim_Ctrain/ --val-dir data/sim_Cval/ --sensor-calib config/sensor_calib.json --circle-tests --edges range --comm-range-train 0.5 5 --comm-range 1.0 --skip residual --eval-only --results-name comm1.0 --out results/gnn_circle/

Distributed GNN (--association local): two detections are associated only if their robots are at most --hops links apart in the communication graph, so the communication range and the number of hops limit what each robot can merge (folder suffix _local). Same commands with --association local, e.g. hops 1-3 and the three skip variants:

    python -m gnn.train_sim --train-dir data/sim_Ctrain/ --val-dir data/sim_Cval/ --sensor-calib config/sensor_calib.json --circle-tests --edges range --comm-range-train 0.5 5 --comm-range 2.0 --association local --hops 2 --skip residual --out results/gnn_circle/
    python -m gnn.train_sim ... --association local --hops 2 --comm-range 1.0 --eval-only --results-name comm1.0 --out results/gnn_circle/


## Re-evaluating everything after a change of the metrics (no retraining)

A box is found only by an estimate within `MATCH_CUTOFF_M` (pipeline/evaluate_runs.py, 0.15 m: half the side of a box) of its centre; P_D, RMSE* (a missed box counts as that distance), false estimates and correct scenes all follow from it. After changing it, every stored result is recomputed from the sets and the trained models:

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --baselines-only
    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --out results/evaluation_calib/
    python -m pipeline.combine_rounds --regions config/optitrack_regions.json
    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ --sensor-calib config/sensor_calib.json --standard-tests --eval-only --out results/gnn_sim/
    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ --sensor-calib config/sensor_calib.json --combined-tests --eval-only --out results/gnn_sim/
    python -m sim.sweep --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --runs 500 --seed 1 --out results/failure_map/
    python -m gnn.collect --results-dir results/gnn_sim/

Two-circle sets: the same with `--sets circle --baselines-only`, then `--eval-only` for each trained variant (same flags as its training).
