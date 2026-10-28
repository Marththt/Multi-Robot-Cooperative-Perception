# Vision-Fault-Robust Multi-Robot Perception

Master's thesis code. Several RoboMaster EP robots look at the
same arena through their cameras, detect ArUco-marked objects, place each
detection on a shared world map using OptiTrack, and fuse the reports of
all robots to estimate **how many objects there are and where they are**.

All objects of the same class carry the same ArUco ID on purpose, so
robots know the class of an object but not its identity: deciding which
reports from different robots are the same object is the core problem.

Python 3.7 (conda environment used in the lab). Run every command from
this folder (the project root).

## Folder structure

| Path | Content |
|---|---|
| `static_trial.py` | One process per robot: live camera preview + live state for the dashboards (`--live`), or capture of one static trial |
| `dashboard.py` | Terminal dashboard: reads the live state of all robots and shows the per-instance fusion and the multi-object localization |
| `map_server.py` + `map.html` | Same information as a live top-down map in the browser |
| `merge_static_trials.py` | Fuses saved static trials (class-vote baselines per instance) |
| `localize_trials.py` | Runs every localization method on saved static trials, optional ground truth (OSPA) and plots |
| `simulate_localization.py` | Compares all localization methods in simulation |
| `survey.py` | Autonomous survey: one process per robot, the robot drives by itself to stops around a fixed box layout and captures what it sees |
| `build_scans.py` | Offline analysis of the survey: sanity checks against the ground truth, scans composed from the observation points, the five methods, breakdown of the failures |
| `vf/control/` | Closed-loop motion to a pose with OptiTrack (`goto.py`, from `goto_point.py`) |
| `vf/perception/` | ArUco detector, detection data, object position from robot pose, YOLO client |
| `vf/localization/` | Robot pose providers: OptiTrack, odometry, camera offset, frame correction |
| `vf/fusion/` | Class-vote baselines, sensor model, MHT, PHD/CPHD, common interface of the localization methods |
| `tools/` | Camera calibration, OptiTrack frame calibration, ground truth recorder, yaw reader, data converter |
| `robomaster_sdk/` | Robot SDK (connection, camera, arm). `arm.py`, `arm_safety.py`, `teleop.py`, `recorder.py` belong to the older pick & throw project and are unchanged |
| `config/` | Arena regions, camera calibrations, ArUco ID -> class map |
| `trials/`, `live/` | Saved trials (one `.jsonl` per robot) and live state files (one `.json` per robot) |

## Requirements

- the `robomaster` SDK and OpenCV with the ArUco module (`opencv-contrib-python`), numpy, matplotlib
- the `natnet` package for OptiTrack (Motive streaming)
- Motive running, with every robot defined as a rigid body (note its ID)

## Coming from the Italian version

The code, the file names, the command-line options and the keys of the
data files are now in English. Convert your existing data once
(the old folder is only read, never modified):

    python -m tools.convert_data_to_english --old ../statico --new .

It converts `trials/`, `live/` and `config/` (for example
`config/regioni_optitrack.json` -> `config/optitrack_regions.json`,
`calib_scacchiera_P0.json` -> `checkerboard_calib_P0.json`, the key
`"classe"` -> `"class_name"`, the class `"scatola"` -> `"box"`). Files that
already exist are not overwritten (the English config files shipped here
are kept).

## Calibration (once)

Camera intrinsics, per robot (checkerboard):

    python -m tools.calibrate_camera --source robot --robot-id P0 --out config/checkerboard_calib_P0.json

OptiTrack frame correction (only if a robot's OptiTrack frame is rotated/shifted
with respect to the shared frame): see `python -m tools.calibrate_optitrack --help`.
Check the yaw sign of a rigid body:

    python -m tools.read_yaw --rb-id 8

## Live: N robots + dashboards

The system works with any number of robots N: every robot runs its own
process (one terminal each) and writes its own state file in `live/`; the
dashboards and the fusion take all the robots they find there. The lab
setup uses three robots (P0, P1, P2); to add one, start one more process
with its own `--robot-id`, `--state-out`, OptiTrack rigid body and camera
calibration.

One terminal per robot (here the three robots of the lab):

    python static_trial.py --robot-id P0 --live --state-out live/P0.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 8 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P0.json

    python static_trial.py --robot-id P1 --live --state-out live/P1.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 7 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json

    python static_trial.py --robot-id P2 --live --state-out live/P2.json --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 9 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P1.json

(P2 has no calibration of its own yet: it uses P1's, same camera model.)

Then either the terminal dashboard:

    python dashboard.py --state-dir live/ --regions config/optitrack_regions.json --localize

or the map in the browser:

    python map_server.py --state-dir live/ --regions config/optitrack_regions.json --localize

and open http://localhost:8765/map.html. In the "Localization" panel click a
method to draw its estimates on the map.

Useful options:

- `--localize mht,cphd`: only some methods (all five: `zone_class, union, mht, phd, cphd`)
- `--zone-side 0.5`: the zones used by `zone_class` are a regular grid over the arena of `--regions`, with cells of at least this side in metres (default 0.5 m: a cell has at least 4 times the area of a 0.20 m box; the lab arena gives 9 x 8 = 72 cells). The named regions of the file are still used by the robots for their own zone label.
- `--p-d`, `--sigma-pos`, `--clutter`, `--n-prior`, `--hfov-deg`, `--r-max`: sensor model shared by all methods. The defaults are ideal conditions: p_D=1 inside the field of view, no false alarms, 10 cm position error, 88 degree field of view, 3 m range. If an object disappears when one robot does not see it (for example hidden behind another box), try `--p-d 0.95`.
- `--p-d 0.8` on a robot process (`static_trial.py --live`): simulated interference, each detection is kept with probability p_D.

## Static trials

Capture one trial (static scene) per robot, same `--trial-id` for all robots:

    python static_trial.py --robot-id P0 --trial-id trial001 --out trials/ --regions config/optitrack_regions.json --localization optitrack --optitrack-rb-id 8 --detector aruco --aruco-marker-size 0.08 --camera-calib config/checkerboard_calib_P0.json

Each row stores the robot's observation of one marker: class, distance,
bearing (`bearing_rad`) and the camera pose of the frame (`robot_pose`).
These are the inputs of the localization: its first step turns them into
world positions (`measurements_by_class` in
`vf/fusion/localization_methods.py`), and the pose also gives each robot's
field of view. Rows saved before the bearing was recorded use their stored
`position` instead.

Fuse the trials (class-vote baselines per instance):

    python merge_static_trials.py --trial-dir trials/

Localize with every method, optionally against ground truth and with plots:

    python localize_trials.py --trial-dir trials/ --regions config/optitrack_regions.json --ground-truth config/ground_truth_positions.json --plots trial_plots/

Ground truth format (written by `tools/record_ground_truth.py`, see below):

    {"trial001": [{"class_name": "box", "x": 0.52, "y": -1.10},
                  {"class_name": "box", "x": -0.30, "y": 0.05}],
     "trial002": []}

## Autonomous survey (offline evaluation of the methods)

The boxes stay fixed; the robots drive by themselves around them and
capture what they see; everything is analysed offline at the end.

The scene is static, so the robots need no synchronisation: every capture
is an independent observation point (camera pose + observations). The
scans the methods are tested on are composed afterwards from these points,
with any number K of "virtual robots".

1. Place the boxes (each with its own OptiTrack rigid body) and record
   their true positions under a layout id:

        python -m tools.record_ground_truth --layout-id L1 --object 20:box --object 21:box --object 22:box

2. Check the plan (no robot moves; the PNG is a top view with boxes,
   stops and paths, to compare with Motive):

        python survey.py --robot-id P0 --layout-id L1 --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --plan-only --plan-png plan_L1.png

   Once per robot, check that OptiTrack rigid body, yaw and command signs
   agree (six short moves, about 0.2 m; it says what is wrong and the
   `--yaw-offset-deg` to give to survey.py):

        python -m tools.check_motion --robot-id P0 --optitrack-rb-id 8

3. Run all the robots ROUND BY ROUND with ONE command (`launch_all.py`).

   FIXED POSITIONS (`--radius R`, use it for comparable layouts): the
   robots stand on a circle of radius R around `--circle-center X Y`
   (default: arena centre), evenly spaced, the first at
   `--first-angle-deg` (0): the SAME positions for every layout, only the
   boxes change. Place the robots roughly near the circle: each one drives
   to the nearest free position (round 0); then, round after round, ALL
   the robots rotate together on the circle by `--step-deg`, always facing
   its centre, for `--sweep-deg` in total (default 360 / number of robots:
   together they cover the whole circle once). The path preview prints the
   largest radius that fits in the arena; a position too close to a box or
   to the border is not used (the robot "stays" in that round; at round 0
   the launch is refused).

   Without `--radius`: round 0 where the robots were placed, the formation
   rotates around the boxes at that distance (default `--sweep-deg` 180).

   `--robot ID:RIGID_BODY[:CAMERA_CALIB]`, one per robot; every other
   option goes to `survey.py`. Optional check before starting (reads where
   the robots are now, nothing moves):

        python launch_all.py --robot P0:8 --robot P1:7 --robot P2:9:config/checkerboard_calib_P1.json --robot P3:10:config/checkerboard_calib_P1.json --layout-id L2 --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --radius 1.3 --step-deg 15 --plan-only --plan-png plan_L2.png

   then the survey:

        python launch_all.py --robot P0:8 --robot P1:7 --robot P2:9:config/checkerboard_calib_P1.json --robot P3:10:config/checkerboard_calib_P1.json --layout-id L2 --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out survey_rounds/ --aruco-marker-size 0.077 --range-fit 1.102 -0.208 --radius 1.3 --step-deg 15

   Every launch is a NEW RUN in its own folder
   `survey_rounds/<layout>_<date-time>/` (captures, raw readings, frames,
   path preview, live figures): every round is driven and captured again,
   nothing is skipped because an earlier run was there. `analyze_rounds.py`
   and `build_scans.py` take one run folder or the whole `survey_rounds/`
   (every run analysed separately, `run` column in the CSV).

   Ctrl+C stops all the robots cleanly (each one stops, closes camera and
   connection, and exits).

   LIVE: as soon as all the robots have finished a round, the launcher
   fuses what they saw with the five methods and shows it in a window
   (robots, fields of view, measurements, estimates with count and OSPA
   of every method), prints one line per method and saves the figure in
   `<run folder>/rounds_live/`; at the end it shows the overall scene too.
   `--range-fit A B` corrects the distances in the live analysis;
   `--no-live` keeps only the saved figures.

   More rounds: a smaller `--step-deg` (e.g. 15).

   Stops: rings around the layout centre (`--radii`, default 1.0 1.4 1.8 m),
   one every `--step-deg` (30), the robot facing the centre, plus captures
   turned by `--yaw-offsets-deg` (0 -30 30). The robot never passes closer
   than `--keep-out-m` (0.45 m) to a box and never plans a stop or a path
   closer than `--margin` (0.5 m) to the arena border, where the OptiTrack
   tracking gets unstable; if it still gets closer than half of that, it is
   stopped and the survey ends. Check stops and paths with `--plan-only`
   first. If a survey is interrupted, the same command resumes it. Move the boxes, record the
   next layout (L2, ...), repeat.

   The motion controller (`vf/control/goto.py`) is `goto_point.py`: same
   gains and limits; command signs checked with `tools/check_motion.py`
   (SIGN_Y = -1).
   Without hardware: `--sim` (fake robot, synthetic observations from the
   ground truth with field of view, occlusions and noise).

4. Round by round: every round (and yaw offset) is one multi-robot scene;
   the five methods fuse the observations of all the robots of that round.
   One figure per scene (robots, fields of view, measurements, estimates
   of every method with count and OSPA), `rounds.csv`, a summary, and one
   overall scene with all the rounds together:

        python analyze_rounds.py --survey-dir survey/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out rounds/

5. Random combinations of observation points (any K), sanity checks of
   the single measurements:

        python build_scans.py --survey-dir survey/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --out analysis/ --plots analysis/

   Step 1 checks the single measurements against the ground truth (bias,
   RMS, range and bearing error, missed detections, false measurements,
   per distance band and per robot, with a flag on the robot that looks
   wrong). Step 2 composes `--scans-per-k` random scans for every K in
   `--k` (default 1 2 3 4 6), runs the five methods and breaks exact
   count and OSPA down by K, layout, coverage of the boxes, missed
   detections and distance. Results in `analysis/measurements.csv` and
   `analysis/scans.csv`.

## Simulation

    python simulate_localization.py
    python simulate_localization.py --trials 200 --max-objects 8
    python simulate_localization.py --robots 7

Compares all the localization methods in ideal conditions: every robot
reports every object in its field of view, at its exact position, with
no false alarms. The only difficulty left is deciding which reports are
the same object (several objects of the same class, overlapping fields
of view). `--robots N` sets the number of robots (default 3, the lab poses).