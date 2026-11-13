# Multi-Robot Cooperative Perception

Master's thesis code. A team of RoboMaster EP robots, localised by OptiTrack, looks at the same arena
through its cameras and has to estimate **how many boxes there are and where they are**.

Every box carries the same ArUco marker on purpose: a robot knows *what* it sees but not *which* box it is,
so deciding which observations of different robots are the same box is the core problem. Three families of
methods are implemented and compared on simulated scenes and on two real campaigns:

| Family | Input | Code |
|---|---|---|
| Classical fusion: zone and class, union, MHT, PHD, CPHD, MHT with sensor model | feature vectors of the ArUco detections | `vf/fusion/`, `pipeline/` |
| Graph neural network (one node per detection) | the same feature vectors | `gnn/` |
| Image network (features lifted to a floor grid, fused cell by cell) | camera frames as box masks, optionally with the detected markers | `bevnet/`, `imgsim/` |

Run every command from the repository root, with the scripts as modules: `python -m <package>.<script> ...`.
All the commands, in the order of the pipeline, are in `docs/commands.md`.

## Folder structure

| Path | Content |
|---|---|
| `robomaster_sdk/` | Robot SDK used for the data collection: connection (`robot`), camera (`camera`), robot registry (`config`) |
| `vf/` | Shared library: motion (`control/`), ArUco detection and feature vectors (`perception/`), robot pose from OptiTrack (`localization/`), fusion methods (`fusion/`: sensor model, zone and class, union, MHT, PHD/CPHD, MHT with sensor model) |
| `acquisition/` | Everything that runs with the robots: `launch_all` (all robots, round by round), `survey` (one robot driving to its stops and capturing), `static_trial` (one capture of one robot), `record_obstacle`, `regroup_captures`, `pack_runs` |
| `tools/` | Lab tools: camera and OptiTrack calibration, motion check, ground truth recorder; capture plans of the real campaigns (`capture_plan`, `capture_plan_v3`), live placement check (`place_check`), run selection (`select_runs`), difficulty levels of the scenes (`scene_levels`) |
| `pipeline/` | Processing of the recorded runs: `build_scans` (loading, sanity checks), `analyze_rounds` (fusion of one round), `calibrate` (sensor calibration), `evaluate_runs` (classical methods, metrics, figures), `combine_rounds` (real scenes with more viewpoints); on the simulated sets: `plan_mht`, `plan_failure_map`, `plan_regions`, `plan_oracle` |
| `sim/` | Measurement-level simulator: library (`geometry`, `model`, `scene`, `sensor`, `writer`), fit on the real runs (`calibrate`), check against them (`generate` replay, `fidelity`), the sets of the thesis (`plan`, `plan_check`) and the earlier training sets (`make_sets`, `specs/`) |
| `gnn/` | Graph network for the association: `dataset` (scenes from runs), `model`, `train_sim` (training on simulated scenes, evaluation on simulated and real ones) |
| `imgsim/` | Image simulator: `render` (camera images, box masks, marker channel), `generate` and `compare` (validation with the real ArUco detector) |
| `bevnet/` | Image network: `data` (rendered sets), `model`, `geometry` (floor grid and projection), `train`, `segment` (Grounding DINO + SAM masks of real frames), `real` (real scenes in the network's format), `twin`, `finetune`, `compare` |
| `demo/` | Live comparison of the methods in the browser: `live_robots` (every robot streams what it sees, OptiTrack the boxes), `live` (runs every method continuously) and its page `map.html` |
| `config/` | Arena regions, camera calibrations, ArUco ID to class map, ground truth of the layouts, sensor calibration, sensor model, appearance files of the image simulator |
| `data/` | Real runs: `survey_rounds/` (first campaign, L4-L11), `survey_v2/` (second campaign, R01-R104). Simulated and image sets are regenerated (`data/sim_*`, `data/plan`, `data/bev*`, not versioned) |
| `results/` | Outputs of every step; `results/scene_levels.csv` and `results/sim_levels.csv` hold the difficulty level of every real and simulated scene |
| `docs/` | `commands.md` (all the commands), capture plans and guides of the second campaign |

## Requirements

- Robots: the `robomaster` SDK, OpenCV with the ArUco module (`opencv-contrib-python`), numpy, matplotlib; the
  `natnet` package and Motive running, with every robot and box defined as a rigid body.
- Analysis of the classical methods: numpy, scipy, matplotlib.
- Networks: PyTorch (`torch`); for the segmentation of real frames also `transformers` and a GPU.

## Pipeline in short

1. **Calibrate once**: cameras (`tools.calibrate_camera`), OptiTrack frame if needed (`tools.calibrate_optitrack`).
2. **Capture**: place the boxes of a layout (`tools.place_check`), then `acquisition.launch_all` moves all the
   robots round by round and stores frames, detections, poses and the true boxes of the run.
3. **Classical methods on real runs**: `pipeline.calibrate` (sensor calibration), `pipeline.evaluate_runs`.
4. **Simulator**: `sim.calibrate` fits the sensor model on the real runs, `sim.fidelity` checks it,
   `sim.plan` builds the training, validation and test sets with fixed seeds.
5. **Graph network**: `gnn.train_sim` trains on the simulated sets and evaluates on simulated and real scenes;
   `pipeline.plan_mht`, `pipeline.plan_failure_map` and `pipeline.plan_oracle` give MHT and the oracle on the same sets.
6. **Image network**: `bevnet.data` renders the sets (box masks, optional marker channel), `bevnet.train`
   trains and tests, `bevnet.real` converts the real runs (segmentation, camera calibration).
7. **Difficulty levels**: `tools.scene_levels` assigns every real scene a level from its geometry.
8. **Live demo**: `demo.live_robots` and `demo.live` show every method continuously while boxes and robots are
   moved by hand.

Files set aside during clean-ups are in `_to_delete/` (not versioned).
