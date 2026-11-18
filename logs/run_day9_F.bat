@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_day9_F.log 2>&1
exit /b 0

:run
echo BLOCK F started %date% %time%: classical methods on the simulated replays, simulator fidelity
python -m pipeline.evaluate_runs --survey-dir data/sim_replay_all/ --ground-truth data/sim_replay_all/ground_truth.json --regions config/optitrack_regions.json --sensor-calib config/sensor_calib.json --full-only --out results/evaluation_sim_replay_all/ || exit /b 1
python -m sim.fidelity --real-survey-dir data/survey_all/ --real-ground-truth config/ground_truth_positions.json --sim-dir data/sim_replay_all/ --sensor-calib config/sensor_calib.json --layouts L5,L7,L9,L11,R02,R04,R06,R08,R10,R12,R14,R16,R18,R20,R22,R24,R26,R28,R30,R32,R34,R36,R39,R41,R43,R45,R47,R49,R51,R53,R55,R57,R59,R69,R71,R74,R81,R83,R85,R87,R89,R91,R93,R95,R97,R99,R101,R103 --real-scenes results/evaluation_all/scenes.csv --sim-scenes results/evaluation_sim_replay_all/scenes.csv --out results/fidelity_all/ || exit /b 1
echo BLOCK F DONE %date% %time%
exit /b 0
