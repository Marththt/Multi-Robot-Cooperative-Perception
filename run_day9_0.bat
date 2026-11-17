@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_day9_0.log 2>&1
set RESULT=%errorlevel%
git add -f results/sensor_model_dump config/sensor_calib.json config/sensor_model.json config/sensor_model_fidelity.json results/evaluation_all results/evaluation_sim_replay_all results/fidelity_all run_day9_0.log
git commit -m "day 9, block 0: calibration, sensor model and classical methods on all the real scenes, real box size"
call :push > run_day9_0_push.log 2>&1
exit /b %RESULT%

:run
python -m pipeline.calibrate data/survey_all/ --camera-offset 0 --out config/sensor_calib.json || exit /b 1
python -m sim.calibrate --survey-dir data/survey_all/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts all --rare-events all --out config/sensor_model.json || exit /b 1
python -m sim.calibrate --survey-dir data/survey_all/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts all --rare-events all --dump results/sensor_model_dump --out results/sensor_model_dump/sensor_model.json
python -m tools.fig_measurements --dump results/sensor_model_dump --out results/sensor_model_dump/measurements.png
python -m tools.fig_layouts --out results/sensor_model_dump/fig_layouts.png
python -m pipeline.evaluate_runs --survey-dir data/survey_all/ --ground-truth config/ground_truth_positions.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --sensor-model config/sensor_model.json --full-only --quiet --out results/evaluation_all/ || exit /b 1
python -m sim.calibrate --survey-dir data/survey_all/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --layouts L4,L6,L8,L10,R01,R03,R05,R07,R09,R11,R13,R15,R17,R19,R21,R23,R25,R27,R29,R31,R33,R35,R38,R40,R42,R44,R46,R48,R50,R52,R54,R56,R58,R60,R70,R73,R75,R82,R84,R86,R88,R90,R92,R94,R96,R98,R100,R102,R104 --rare-events calibration --out config/sensor_model_fidelity.json || exit /b 1
python -m sim.generate replay --survey-dir data/survey_all/ --ground-truth config/ground_truth_positions.json --sensor-calib config/sensor_calib.json --model config/sensor_model_fidelity.json --layouts L5,L7,L9,L11,R02,R04,R06,R08,R10,R12,R14,R16,R18,R20,R22,R24,R26,R28,R30,R32,R34,R36,R39,R41,R43,R45,R47,R49,R51,R53,R55,R57,R59,R69,R71,R74,R81,R83,R85,R87,R89,R91,R93,R95,R97,R99,R101,R103 --reps 50 --seed 1 --out data/sim_replay_all/ || exit /b 1
python -m pipeline.evaluate_runs --survey-dir data/sim_replay_all/ --ground-truth data/sim_replay_all/ground_truth.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --full-only --quiet --out results/evaluation_sim_replay_all/ || exit /b 1
python -m sim.fidelity --real-survey-dir data/survey_all/ --real-ground-truth config/ground_truth_positions.json --sim-dir data/sim_replay_all/ --sensor-calib config/sensor_calib.json --layouts L5,L7,L9,L11,R02,R04,R06,R08,R10,R12,R14,R16,R18,R20,R22,R24,R26,R28,R30,R32,R34,R36,R39,R41,R43,R45,R47,R49,R51,R53,R55,R57,R59,R69,R71,R74,R81,R83,R85,R87,R89,R91,R93,R95,R97,R99,R101,R103 --real-scenes results/evaluation_all/scenes.csv --sim-scenes results/evaluation_sim_replay_all/scenes.csv --out results/fidelity_all/ || exit /b 1
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
