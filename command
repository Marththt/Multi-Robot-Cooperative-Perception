trial live 

    python static_trial.py --robot-id P0 --live --state-out live/P0.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 8 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P0.json

    python static_trial.py --robot-id P1 --live --state-out live/P1.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 7 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json

    python static_trial.py --robot-id P2 --live --state-out live/P2.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 9 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json

    python static_trial.py --robot-id P3 --live --state-out live/P3.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 10 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json

    python map_server.py --state-dir live/ --regions config/optitrack_regions.json --bayesian --localize


capture box 

    python -m tools.record_ground_truth --layout-id L4 --object 11:box --object 12:box --object 13:box --object 14:box --overwrite

move, capture and analyze

move and capture:

python launch_all.py --robot P0:8 --robot P1:7 --robot P4:6:config/checkerboard_calib_P1.json --robot P3:10:config/checkerboard_calib_P1.json --layout-id L4 --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out survey_rounds/ --aruco-marker-size 0.077 --radius 1.3 --step-deg 15 --sweep-deg 90 --range-fit 1.064 -0.080 --camera-offset-forward-m 0.12 --instance-threshold-m 0.2

analyze:

    python build_scans.py --survey-dir survey/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out analysis/ --plots analysis/



python evaluate_runs.py --survey-dir survey_rounds_t02/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --camera-offset 0.13 --range-fit 1.070 -0.073 --out evaluation/ --names L1=Line L2=Square L3=Pairs