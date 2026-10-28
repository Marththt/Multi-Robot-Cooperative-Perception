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

Fit the sensor model (L5, L7, L10; rare events from all layouts):

    python -m sim.calibrate --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --rare-events all --out config/sensor_model.json

Fidelity check on L4, L6, L8, L9, L11:

    python -m sim.generate replay --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --model config/sensor_model.json --layouts L4,L6,L8,L9,L11 --reps 200 --seed 1 --out data/sim_replay/
    python -m pipeline.evaluate_runs --survey-dir data/sim_replay/ --ground-truth data/sim_replay/ground_truth.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --full-only --quiet --out results/evaluation_sim_replay/
    python -m sim.fidelity --real-survey-dir data/survey_rounds/ --real-ground-truth config/ground_truth_positions.json --sim-dir data/sim_replay/ --sensor-calib config/sensor_calib.json --layouts L4,L6,L8,L9,L11 --real-scenes results/evaluation_calib/scenes.csv --sim-scenes results/evaluation_sim_replay/scenes.csv --out results/fidelity/

New scenes from a scenario (training set, or a test condition T0-T7 in `sim/specs/`):

    python -m sim.generate random --model config/sensor_model.json --spec sim/specs/train_general.json --runs 1000 --name TRAIN --seed 1 --out data/sim_train/
    python -m sim.generate random --model config/sensor_model.json --spec sim/specs/T1_close.json --runs 1000 --name T1 --seed 11 --out data/sim_T1/
