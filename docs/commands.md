# Commands

Run every command from the repository root. Scripts are run as modules (`python -m <package>.<script>`).

## Live demo: every method continuously, in the browser

The robots stand still and stream what they see. Place them, move them by hand, move, add or take away boxes, put
other objects in the way, at any time: the page follows within a second for MHT, MHT with sensor model and the
graph network, within a few seconds for the two image networks.

`demo/live.py` only reads the files the robots write, so it runs in its own terminal and its own Python environment
(the robot SDK needs an old Python; the networks need a recent one with `torch`). One-time setup on the capture PC,
from the repository root:

    py -3.11 -m venv .venv_demo
    .venv_demo\Scripts\activate.bat
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install transformers pillow opencv-python numpy scipy matplotlib

Terminal 1 (the usual capture environment): the robots and OptiTrack. It starts one camera process per robot, one
after the other, and writes the boxes OptiTrack tracks inside the arena (every box must be a rigid body in Motive,
11-17). Ctrl+C here stops every robot; `--windows` also shows the camera window of each robot.

    python -m demo.live_robots --robot P0:8 --robot P1:7 --robot P3:10:config/checkerboard_calib_P1.json --robot P5:6:config/checkerboard_calib_P1.json

Terminal 2 (demo environment): the methods and the page, http://localhost:8000. The first start downloads the
segmentation models (Grounding DINO, SAM) once.

    .venv_demo\Scripts\activate.bat
    python -m demo.live

Moving the robots from the page: click a robot (or its button in the "Move" bar), then a point of any map: the
robot drives there on a path clear of the boxes and of the other robots and arrives facing the centre of the
boxes (acquisition/live_drive.py, the controller of the captures). "turn left/right" turns it by 22.5 degrees,
"face the boxes" turns it towards them, "stop" halts it, "stop all robots" halts every robot. Goals too close to a
box or to the arena border are refused, and the reason shows next to the buttons. `--no-drive` on
`demo.live_robots` turns this off (robots moved only by hand).

Each panel shows the boxes its method finds now against the true boxes, the verdict (correct, too few, too many,
misplaced), how old its input is and how often it was correct in the last 30 seconds; above the panels, the
difficulty level of the scene as it is now; below, the frames of the robots with the box masks and the markers.

Camera angles for the image networks: by default the pitch and yaw offset of each camera are fitted continuously on
the markers the robots see. To fix them before the demo, leave the robots looking at some boxes for a minute while
saving them, then start the demo with the file:

    python -m demo.live --write-camera-calib config/demo_cameras.json
    python -m demo.live --camera-calib config/demo_cameras.json

Rehearsal without robots: a recorded run played as if it were live, one round every 4 seconds:

    python -m demo.live --replay data/survey_v2/R01_20261109-153345F --replay-every 4

Other options: `--window-s` (seconds of detections pooled per object, default 1), `--no-images` (only the methods
on the detections, for a PC without GPU), `--gnn`, `--image`, `--image-markers` (variant folders in `results/`),
`--port`. A method whose packages or model files are missing is shown as "not available" and the others keep
working.

## Record the box positions (OptiTrack)

    python -m tools.record_ground_truth --layout-id L4 --object 11:box --object 12:box --object 13:box --object 14:box --overwrite

## Move and capture (all robots, round mode)

    python -m acquisition.launch_all --robot P0:8 --robot P1:7 --robot P4:6:config/checkerboard_calib_P1.json --robot P3:10:config/checkerboard_calib_P1.json --layout-id L4 --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out data/survey_rounds/ --aruco-marker-size 0.077 --radius 1.3 --step-deg 15 --sweep-deg 90 --range-fit 1.064 -0.080 --camera-offset-forward-m 0.12 --instance-threshold-m 0.2

Second campaign (R01-R104): the layouts are planned first (`docs/capture_plan_v2.csv/.json`,
`docs/capture_plan_v3.json`, one page per layout in `docs/capture_guide_v*.pdf`), then every layout is placed with
the live OptiTrack check, which starts `launch_all` when the boxes are in place:

    python -m tools.capture_plan                 # plan v2 (R01-R80) from docs/capture_plan_v2.csv
    python -m tools.capture_plan_v3              # plan v3 (R81-R104: medium scenes, white objects, light)
    python -m tools.place_check --layout R81 --tolerance 0.03

One run per layout (the one with the most frames), for every evaluation of the second campaign:

    python -m tools.select_runs --src data/survey_v2/ --dst data/survey_v2_best/

## Sensor calibration and evaluation of the classical methods

    python -m pipeline.calibrate data/survey_rounds/ --camera-offset 0 --out config/sensor_calib.json
    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --out results/evaluation_calib/

Older session (L1-L3, runs saved with the body pose: camera offset 0.12 m):

    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds_t02/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --camera-offset 0.13 --range-fit 1.070 -0.073 --out results/evaluation/ --names L1=Line L2=Square L3=Pairs

Scans and per-round analysis:

    python -m pipeline.build_scans --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out results/analysis/ --plots results/analysis/

Second campaign (same calibration and sensor model as the first; `--full-only`: all the robots of each scene):

    python -m pipeline.evaluate_runs --survey-dir data/survey_v2_best/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --sensor-model config/sensor_model.json --full-only --quiet --out results/evaluation_v3/

Difficulty level of every real scene, from its geometry only (results/scene_levels.csv):

    python -m tools.scene_levels

## Simulator

Fidelity check: sensor model fitted on L5, L7, L10 only (rare events from all layouts), checked on L4, L6, L8, L9, L11:

    python -m sim.calibrate --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts L5,L7,L10 --rare-events all --out config/sensor_model_fidelity.json
    python -m sim.generate replay --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --model config/sensor_model_fidelity.json --layouts L4,L6,L8,L9,L11 --reps 200 --seed 1 --out data/sim_replay/
    python -m pipeline.evaluate_runs --survey-dir data/sim_replay/ --ground-truth data/sim_replay/ground_truth.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --full-only --quiet --out results/evaluation_sim_replay/
    python -m sim.fidelity --real-survey-dir data/survey_rounds/ --real-ground-truth config/ground_truth_positions.json --sim-dir data/sim_replay/ --sensor-calib config/sensor_calib.json --layouts L4,L6,L8,L9,L11 --real-scenes results/evaluation_calib/scenes.csv --sim-scenes results/evaluation_sim_replay/scenes.csv --out results/fidelity/

Sensor model for everything else (failure map, training, test conditions), fitted on all layouts:

    python -m sim.calibrate --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts L4,L5,L6,L7,L8,L9,L10,L11 --rare-events all --out config/sensor_model.json

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

Two-circle sets: the same with `--sets circle --baselines-only`, then `--eval-only` for each trained variant (same flags as its training).


## MHT with sensor model (the same per-detection inputs as the GNN)

`mht_sensor` (vf/fusion/sensor_aware_mht.py) is the MHT that also reads distance, line of sight and viewing angle of every report and the fitted sensor model (config/sensor_model.json). It is one of the classical methods of pipeline/evaluate_runs.py, so new evaluations include it. To add it to results already computed without running the other methods again (their rows in scenes.csv are kept):

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --baselines-only --methods mht_sensor
    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --sets circle --baselines-only --methods mht_sensor
    python -m pipeline.evaluate_runs --survey-dir data/survey_rounds/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --methods mht_sensor --keep-other-methods --out results/evaluation_calib/
    python -m pipeline.combine_rounds --regions config/optitrack_regions.json --methods mht_sensor

The GNN results then compare with it after `--eval-only` (same commands as above).


## Random layouts (R0)

The nominal sensor of T0, but 3-8 boxes and 4-8 robots anywhere in a 3 m x 3 m area, robots looking in a random direction; a layout is drawn again until every box is in range and in the field of view of at least two robots (sim/specs/R0_random.json). No training set has this placement. Build the set, run the MHT with sensor model, then test the saved detection-node GNN on it:

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --sets R0 --methods mht_sensor
    python -m gnn.train_sim --train-dir data/sim_train/ --val-dir data/sim_val/ --sensor-calib config/sensor_calib.json --test R0 data/sim_R0/ data/sim_R0/ground_truth.json results/baselines_sim/R0/scenes.csv --graph detection --features full --edges ring --hops 2 --seeds 5 --eval-only --results-name random --out results/gnn_sim/

GNN trained on random layouts (sim/specs/R_train.json: the R0 placement with the randomised sensor of the general training set), tested on T0-T7, the real scenes and R0:

    python -m sim.make_sets --model config/sensor_model.json --sensor-calib config/sensor_calib.json --regions config/optitrack_regions.json --sets Rtrain,Rval --no-baselines
    python -m gnn.train_sim --train-dir data/sim_Rtrain/ --val-dir data/sim_Rval/ --sensor-calib config/sensor_calib.json --standard-tests --test R0 data/sim_R0/ data/sim_R0/ground_truth.json results/baselines_sim/R0/scenes.csv --graph detection --features full --edges ring --hops 2 --seeds 3 --out results/gnn_random/

GNN trained on both placements (ring layouts of the general training set and random layouts, 50,000 scenes each, pooled; validation on both), tested on the same sets:

    python -m gnn.train_sim --train-dir data/sim_train/ data/sim_Rtrain/ --val-dir data/sim_val/ data/sim_Rval/ --sensor-calib config/sensor_calib.json --standard-tests --test R0 data/sim_R0/ data/sim_R0/ground_truth.json results/baselines_sim/R0/scenes.csv --graph detection --features full --edges ring --hops 2 --seeds 3 --out results/gnn_mixed/


## Research plan: MHT with a learned normality gate (modules M0-M1)

M0, simulator and datasets (sim/plan.py): two scene families (circle, random) in the world frame, the nominal sensor file, the error knobs k_f, k_2, h, and every set of the plan with fixed seeds (train 50,000, gate 10,000, val 2,000, 52 test sets of 1,000 scenes: G0-G2, E1-E3, S1, V1-V2 in both families). Seeds in results/plan/seeds.csv. About 3 GB in data/plan/.

    python -m sim.plan --list
    python -m sim.plan --model config/sensor_model.json
    python -m sim.plan_check --examples --out results/plan/M0/

M1, generator and gate-input check (sim/plan_check.py) on the gate-training set: position error and detection probability against the sensor file, two-face and false-detection rates, histograms of the 8 gate-input features by family, N and B, region sizes and purity (Fig. M1-1 to M1-4, table_M1.csv):

    python -m sim.plan_check --set-dir data/plan/gate/ --out results/plan/M1/

Region link distance: 0.20 m instead of the plan's 0.35 m (pipeline/plan_regions.py LINK_M). With 0.35 m the largest region has 101 reports (M_max 64) and 27% of the regions hold two or more boxes, also in G0; with 0.20 m 54 reports, 7% (3% in G0), 0.4% of the boxes split over two regions.

M2, MHT setup (pipeline/plan_mht.py): MHT with sensor model in simulation without the two real-data terms, H in {100, 300, 1000}, three random robot orders, on the validation set; writes table_M2.csv, fig_M2-1 and prints the chosen H (smallest within 0.5 points of H = 1000). Then MHT with the chosen H on the gate-training set, and M1-3 again on the regions MHT gets right only:

    python -m pipeline.plan_mht --sets data/plan/val/ --hypotheses 100 300 1000 --orders 3 --choose-h --out results/plan/M2/
    python -m pipeline.plan_mht --sets data/plan/gate/ --hypotheses H --orders 1 --out results/plan/M2/
    python -m sim.plan_check --set-dir data/plan/gate/ --out results/plan/M1/ --mht-estimates results/plan/M2/gate/estimates_HH_o0.jsonl
(replace H with the chosen value)

M2 result: H = 100 (70.47% correct on validation; 300 and 1000 give 70.52%), the three orders within 1 point, so one order from here on.

M3, MHT failure map (pipeline/plan_failure_map.py): MHT with H = 100 on every test set, S1 also with H = 300 and 1000; table_M3.csv, one figure per scenario, kept_cases.csv (special cases where MHT loses at least 5 points from G0):

    python -m pipeline.plan_mht --sets "data/plan/G*" "data/plan/E*" "data/plan/V*" --hypotheses 100 --out results/plan/M3/
    python -m pipeline.plan_mht --sets "data/plan/S1_*" --hypotheses 100 300 1000 --out results/plan/M3/
    python -m pipeline.plan_failure_map --results results/plan/M3/ --out results/plan/M3/ --data data/plan/

(--data adds the share of scenes in which every box was reported by some robot, and MHT's correct share on those.)

Evaluation scenarios kept for the thesis (advisor: at most 3-4): the nominal scene G0 as reference, E1 (false detections x1-8), E3 (robots hidden by obstacles, h 0-2), S1 (dense boxes 20 cm apart, B 4-16); both families. Dropped: E2 (MHT loses at most 6 points), G1, G2, V2 (MHT flat), V1 (nearly no scene has every box reported).

M4, GNN and oracle selector (pipeline/plan_oracle.py): detection-node GNN trained on the plan's training set (nominal sensor, both families, every pair of robots linked, edges within 1 m), 3 seeds, its boxes saved on the kept test sets; then the region-level oracle against MHT (H = 100):

    python -m gnn.train_sim --train-dir data/plan/train/ --val-dir data/plan/val/ --sensor-calib config/sensor_calib.json --graph detection --features full --edges full --hops 2 --seeds 3 --plan-tests "G0_*" "E1_*" "E3_*" "S1_*" --save-estimates --out results/plan/M4/gnn/
    python -m pipeline.plan_oracle --sets "G0_*" "E1_*" "E3_*" "S1_*" --mht results/plan/M3/ --gnn results/plan/M4/gnn/detection_full_full_h2/ --out results/plan/M4/


## Image-level simulator (imgsim/)

imgsim/render.py renders the camera image of every robot in a scene of sim/plan.py (same seeds: scene i of data/imgsim/<set>/ is scene i of data/plan/<set>/), with the real camera calibration, the camera height and pitch measured on the real frames, the real boxes (white, blue tape, ArUco 4X4_50 id 0, 77 mm), obstacles, the other robots, the room and the robot's own gripper (imgsim/assets, from the real frames of P5). imgsim/generate.py runs the real detector (vf/perception/aruco_detector.py) on 5 frames per stop and keeps one detection per group closer than 0.2 m, as acquisition/static_trial.capture_trial does on the robots, and writes the reports in the format of the real runs. imgsim/compare.py fits the range / bearing calibration of an image set (as pipeline/calibrate.py) and compares the image reports with the sensor file fitted on the real runs (Fig. IMG-1 to IMG-3, table_IMG.csv). Needs OpenCV with cv2.aruco (opencv-python >= 4.7 or opencv-contrib-python).

E1 and E2 multiply rates of the sensor file, which the renderer does not use: in the images false detections and second faces come from the detector itself, so only G0, E3 and S1 are rendered.

    python -m imgsim.generate --sets G0,E3,S1 --runs 1000 --save-images 20
    python -m imgsim.compare --set-dir data/imgsim/G0_nominal_circle/ --save-calib config/sensor_calib_imgsim.json
    python -m imgsim.compare --set-dir data/imgsim/G0_nominal_random/
    python -m pipeline.plan_mht --sets "data/imgsim/G0_*" "data/imgsim/E3_*" "data/imgsim/S1_*" --sensor-calib config/sensor_calib_imgsim.json --hypotheses 100 --out results/imgsim/M3/

GNN trained on the measurement-level sets (M4), tested on the image sets, then the oracle on the image sets:

    python -m gnn.train_sim --train-dir data/plan/train/ --val-dir data/plan/val/ --sensor-calib config/sensor_calib_imgsim.json --graph detection --features full --edges full --hops 2 --seeds 3 --eval-only --plan-dir data/imgsim/ --plan-tests "G0_*" "E3_*" "S1_*" --save-estimates --results-name imgsim --out results/plan/M4/gnn/
    python -m pipeline.plan_oracle --sets "G0_*" "E3_*" "S1_*" --data data/imgsim/ --mht results/imgsim/M3/ --gnn results/plan/M4/gnn/detection_full_full_h2/ --estimates-dir estimates_imgsim --sensor-calib config/sensor_calib_imgsim.json --out results/imgsim/M4/

Appearance settings (imgsim/render.py Appearance) can be changed with --appearance <json> on imgsim.generate; --save-images -1 saves every image (for an image-based model).

## Image-based network (bevnet/)

bevnet/ learns the boxes from the camera images and the robot poses only (no ArUco, no solvePnP, no sensor file). Each robot encodes its image with a small CNN; the features are read where every cell of a 4 x 4 m floor grid (5 cm cells), lifted to 0.05, 0.20 and 0.35 m, falls in the image (known pose, pitch and lens) and turned into a message of --message channels per cell it sees; the team fuses the messages (mean, maximum, number of robots per cell) and a U-Net on the floor grid gives the probability of a box centre per cell and its offset. Boxes = local maxima above a threshold chosen on the validation set; same "correct scene" metric as MHT and the GNN (count right, every box within 15 cm). The scenes are those of sim/plan.py (same seeds), rendered by imgsim/render.py at 256 x 144 (--scale 0.2). Needs PyTorch (pip install torch; a CUDA GPU makes training minutes instead of hours).

Render the sets (about 80 kB per scene: 4 GB for the 50,000 training scenes):

    python -m bevnet.data --sets train,val,G0,E3,S1
    python -m bevnet.data --sets train,val,G0,E3,S1 --no-markers --out data/bev_nomarkers/   # same scenes, boxes without paper and marker

Train, choose epoch and threshold on val, test (results/bev/bev_c64_m16/: model.pt, train_log.csv, test_scenes.csv, summary.csv); on a CPU only, start with --max-train 10000 --workers 4:

    python -m bevnet.train --train data/bev/train --val data/bev/val --tests "data/bev/G0_*" "data/bev/E3_*" "data/bev/S1_*"
    python -m bevnet.train --message 4 ...           # smaller message per cell (bandwidth study)
    python -m bevnet.train ... --eval-only           # test the saved model again
    python -m bevnet.train --train data/bev_nomarkers/train --val data/bev_nomarkers/val --tests "data/bev_nomarkers/G0_*" "data/bev_nomarkers/E3_*" "data/bev_nomarkers/S1_*" --out results/bev_nomarkers/

Occluders and other objects in the training scenes. --obstacles 2 puts the E3 cylinders (0, 1 or 2 per box, drawn per scene) in the training and validation scenes; --distractors N adds 0..N free-standing objects to any set; --obstacle-style random gives every obstacle its own shape (--obstacle-shapes from cylinder, box, prism, hexagon, blob), height (15-100 cm), colour (a quarter white or light grey) and pattern (plain, noise, stripes, bands, checker). Training on three shapes and testing on the two others measures how the network copes with objects it has never seen:

    python -m bevnet.data --sets train --runs 20000 --obstacles 2 --distractors 4 --obstacle-style random --obstacle-shapes cylinder,box,prism --out data/bev_rand/
    python -m bevnet.data --sets val --obstacles 2 --distractors 4 --obstacle-style random --obstacle-shapes cylinder,box,prism --out data/bev_rand/
    python -m bevnet.data --sets G0,E3 --distractors 4 --obstacle-style random --obstacle-shapes hexagon,blob --out data/bev_unseen/
    python -m bevnet.train --train data/bev_rand/train --val data/bev_rand/val --tests "data/bev/G0_*" "data/bev/E3_*" "data/bev/S1_*" --out results/bev_rand/
    python -m bevnet.train --eval-only --out results/bev_rand/ --tests "data/bev_unseen/*" --results-name unseen

Real frames (bevnet/real.py): the 56 real scenes (data/survey_rounds, 8 layouts x 7 rounds, 4 robots) in the same format, one set per reading k = 0..4 of each stop: each robot's frame resized to 256 x 144, its OptiTrack camera pose, its own camera matrix and distortion, pitch 11.5 deg (P5 22 deg), true boxes from ground_truth.json, all shifted to the centre of the robots. --check writes fig_REAL-0.png with the true box centres projected into the frames. Then any model trained in simulation is tested on them without retraining (results in summary_real.csv):

    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real/ --readings 0,1,2,3,4 --check
    python -m bevnet.train --eval-only --out results/bev/ --tests "data/bev_real/*" --results-name real
    python -m bevnet.train --eval-only --out results/bev_obst/ --tests "data/bev_real/*" --results-name real

Twins of the real scenes (bevnet/twin.py: same poses, pitch and boxes, rendered) and fine-tuning on the real frames, leave-one-layout-out (bevnet/finetune.py):

    python -m bevnet.twin --real data/bev_real/real_k0 --out data/bev_real_twin/ --figure
    python -m bevnet.train --eval-only --out results/bev_obst/ --tests data/bev_real_twin/twin_k0 --results-name twin
    python -m bevnet.finetune --init results/bev_obst/bev_c64_m16 --real "data/bev_real/real_k*" --sim data/bev_obst/train --sim-scenes 500 --out results/bev_real_ft/

Towards real frames: --visual random renders robots with wheels and arm, windows, light spots on the floor, desks with screens and wider greys; --encoder resnet18 uses an ImageNet-pretrained ResNet-18 (pip install torchvision); --jitter changes colour and brightness of the training images:

    python -m bevnet.data --sets train --runs 20000 --obstacles 2 --distractors 4 --obstacle-style random --obstacle-shapes cylinder,box,prism --visual random --out data/bev_vis/
    python -m bevnet.data --sets val --obstacles 2 --distractors 4 --obstacle-style random --obstacle-shapes cylinder,box,prism --visual random --out data/bev_vis/
    python -m bevnet.train --encoder resnet18 --jitter 0.3 --name resnet_vis --train data/bev_vis/train --val data/bev_vis/val --tests "data/bev/G0_*" "data/bev/E3_*" "data/bev/S1_*" --out results/bev_vis/
    python -m bevnet.train --eval-only --name resnet_vis --out results/bev_vis/ --tests "data/bev_real/*" --results-name real
    python -m bevnet.train --jitter 0.3 --name small_vis --train data/bev_vis/train --val data/bev_vis/val --tests "data/bev/G0_*" "data/bev/E3_*" "data/bev/S1_*" --out results/bev_vis/

Camera pitch of the real robots: the camera is 29 cm above the floor, so a pitch error of 1 deg moves the middle of a box at 1.5 m by about 60 cm along the ray. --calibrate fits the pitch and a yaw offset per robot from the ArUco detections stored with the frames (camera calibration only), leave one layout out. --pitch-noise / --yaw-noise train the network with errors on the given pitch and yaw:

    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real_cal/ --readings 0,1,2,3,4 --calibrate --check
    python -m bevnet.train --eval-only --out results/bev_obst/ --tests "data/bev_real_cal/*" --results-name real_cal
    python -m bevnet.train --pitch-noise 1.5 --yaw-noise 1.0 --name pitchnoise --train data/bev_obst/train --val data/bev_obst/val --tests "data/bev/G0_*" "data/bev/E3_*" "data/bev/S1_*" --out results/bev_pn/

Side by side with MHT and the GNN on the same scenes (table_BEV.csv, fig_BEV-1):

    python -m bevnet.compare --method "MHT=results/plan/M3/summary.csv" "GNN=results/plan/M4/gnn/detection_full_full_h2/summary.csv" "MHT (images, ArUco)=results/imgsim/M3/summary.csv" "Image network=results/bev/bev_c64_m16/summary.csv" --out results/bev/

Boxes only. --masks renders only the box silhouettes (white on black; obstacles and robots hide boxes but are not drawn, no room, no gripper, no camera effects). The real frames get the same treatment from bevnet/segment.py (Grounding DINO "a white box." + SAM, pip install transformers; full-size masks cached in <out>/cache/, overlay check in <out>/fig_REAL-masks.png). Training relaxes the conditions: --mask-noise 1 makes the training masks imperfect (edges, holes, false blobs, missed boxes), --pitch-noise / --yaw-noise / --height-noise put errors on the given camera geometry:

    python -m bevnet.real --runs "data/survey_rounds/*" --out data/bev_real_masks/ --readings 0,1,2,3,4 --calibrate --masks
    python -m bevnet.data --sets train --runs 20000 --obstacles 2 --masks --out data/bev_mask/
    python -m bevnet.data --sets val --obstacles 2 --masks --out data/bev_mask/
    python -m bevnet.data --sets G0,E3,S1 --masks --out data/bev_mask/
    python -m bevnet.train --pitch-noise 1.5 --yaw-noise 1.0 --height-noise 0.025 --mask-noise 1.0 --name masks --train data/bev_mask/train --val data/bev_mask/val --tests "data/bev_mask/G0_*" "data/bev_mask/E3_*" "data/bev_mask/S1_*" --out results/bev_mask/
    python -m bevnet.train --eval-only --name masks --out results/bev_mask/ --tests "data/bev_real_masks/*" --results-name real

Final image networks of the thesis (measured geometry, wide camera pitch, imperfect masks). Masks only (`masks_wp`):

    python -m bevnet.data --sets train --runs 20000 --obstacles 2 --masks --appearance config/appearance_wide_pitch.json --out data/bev_mask_wp/
    python -m bevnet.data --sets val --obstacles 2 --masks --appearance config/appearance_wide_pitch.json --out data/bev_mask_wp/
    python -m bevnet.data --sets G0,E3,S1 --masks --out data/bev_mask_wp/
    python -m bevnet.train --train data/bev_mask_wp/train --val data/bev_mask_wp/val --tests "data/bev_mask_wp/G0_*" "data/bev_mask_wp/E3_*" "data/bev_mask_wp/S1_*" --out results/bev_mask_wp/ --name masks_wp --pitch-noise 1.5 --yaw-noise 1.0 --height-noise 0.025 --mask-noise 1.0

Masks + detected markers (`masks_mk`): second image channel with the marker detections, white objects without
markers among the obstacles, markers dropped in training:

    python -m bevnet.data --sets train --runs 20000 --obstacles 2 --distractors 3 --obstacle-style random --masks --appearance config/appearance_markers.json --out data/bev_mask_mk/
    python -m bevnet.data --sets val --runs 2000 --obstacles 2 --distractors 3 --obstacle-style random --masks --appearance config/appearance_markers.json --out data/bev_mask_mk/
    python -m bevnet.data --sets G0,E3,S1 --masks --appearance config/appearance_markers_test.json --out data/bev_mask_mk/
    python -m bevnet.train --train data/bev_mask_mk/train --val data/bev_mask_mk/val --tests "data/bev_mask_mk/G0_*" "data/bev_mask_mk/E3_*" "data/bev_mask_mk/S1_*" --out results/bev_mask_mk/ --name masks_mk --pitch-noise 1.5 --yaw-noise 1.0 --height-noise 0.025 --mask-noise 1.0 --marker-dropout 1.0

Real campaigns in the network's format (masks cached; first campaign calibrated on the other layouts, second on
the run itself because the cameras moved between layouts), then the tests:

    python -m bevnet.real --runs "data/survey_rounds/*" --masks --calibrate --readings 0,1,2,3,4 --out data/bev_real_masks/
    python -m bevnet.real --runs "data/survey_rounds/*" --masks --calibrate --mask-cache data/bev_real_masks/cache --marker-channel --readings 0,1,2,3,4 --out data/bev_real_mk5/
    python -m bevnet.real --runs "data/survey_v2_best/*" --masks --calibrate --calibrate-scope same-run --mask-cache data/bev_real_v2_masks/cache --readings 0,1,2,3,4 --out data/bev_real_v3_masks/
    python -m bevnet.real --runs "data/survey_v2_best/*" --masks --calibrate --calibrate-scope same-run --mask-cache data/bev_real_v2_masks/cache --marker-channel --readings 0,1,2,3,4 --out data/bev_real_v3_mk5/
    python -m bevnet.train --eval-only --out results/bev_mask_wp/ --name masks_wp --val data/bev_mask_wp/val --tests "data/bev_real_v3_masks/real_k*" --results-name real_v3
    python -m bevnet.train --eval-only --out results/bev_mask_mk/ --name masks_mk --val data/bev_mask_mk/val --tests "data/bev_real_v3_mk5/real_k*" --results-name real_v3
    python -m bevnet.train --eval-only --out results/bev_mask_mk/ --name masks_mk --val data/bev_mask_mk/val --tests "data/bev_real_v3_mk5/real_k*" --results-name real_v3_nomk --drop-markers

`--oracle-markers` (bevnet.real) adds the marker of every visible box the detector missed: what the network does
when no visible marker is lost. Variants that did not help on the real scenes, kept for the record: `--cutout`
(the frame inside the mask) and `--instances` / `config/appearance_*borders.json` (a line between touching boxes).

