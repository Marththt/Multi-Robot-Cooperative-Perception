cd "C:\GNN for vision fault\Multi-Robot-Cooperative-Perception"
git stash
git pull --no-rebase --no-edit
.venv\Scripts\python -c "import torch; print('torch==' + torch.__version__)" | Out-File -Encoding ascii torch_pin.txt
.venv\Scripts\python -c "import ultralytics" 2>$null; if ($LASTEXITCODE -ne 0) { .venv\Scripts\python -m pip install "ultralytics==8.4.213" -c torch_pin.txt }
Set-Content -Path run_day9_all.bat -Encoding Ascii -Value @'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -c "import sim.geometry as g; assert g.APOTHEM_M < 0.08" || goto :oldcode
python -c "import yolo_gnn.evaluate, yolo_gnn.train_seg, yolo_gnn.evaluate_sim, yolo_gnn.segment_sim, sim.scaling" || goto :oldcode
python -c "import inspect, sim.calibrate as c; assert 'nargs=\"+\"' in inspect.getsource(c)" || goto :oldcode
python -c "import ultralytics, torch; assert torch.cuda.is_available()" || goto :nodeps
if not exist data\survey_all mkdir data\survey_all
for /d %%d in (data\survey_rounds\* data\survey_v2_best\*) do if not exist "data\survey_all\%%~nxd" mklink /J "data\survey_all\%%~nxd" "%%~fd" > nul
if exist run_day9_A.done del run_day9_A.done
if exist run_day9_Y.done del run_day9_Y.done
echo BLOCK 0 started %date% %time%: calibration and sensor model on all the real scenes, classical methods, simulator check
call run_day9_0.bat
if errorlevel 1 goto :fail0
for /f %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do set NGPU=%%i
set GPU_A=0
set GPU_Y=0
if %NGPU% GEQ 2 set GPU_A=1
start "day9 A (marker network)" cmd /c run_day9_A.bat
start "day9 Y (YOLO)" cmd /c run_day9_Y.bat
echo BLOCK 0 done. Blocks A and Y started on %NGPU% GPU(s) %date% %time%. Logs: run_day9_A.log and run_day9_Y.log
:wait
if not exist run_day9_A.done (timeout /t 120 /nobreak > nul & goto :wait)
if not exist run_day9_Y.done (timeout /t 120 /nobreak > nul & goto :wait)
echo Blocks A and Y done %date% %time%, now block S (validation in simulation, up to 50 robots)
call run_day9_S.bat
echo Block S done %date% %time%, now block H (YOLO + marker network on the real scenes)
call run_day9_H.bat
echo ALL DONE %date% %time%: see run_day9_H.log and results\yolo_gnn\real_tables.txt
goto :eof
:oldcode
echo The code on this PC is old: push from the Mac, then run run_day9_all.bat again.
goto :eof
:nodeps
echo ultralytics is missing or CUDA is not available in .venv: install ultralytics==8.4.213 without changing torch.
goto :eof
:fail0
echo BLOCK 0 failed: see run_day9_0.log
'@
Set-Content -Path run_day9_0.bat -Encoding Ascii -Value @'
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
'@
Set-Content -Path run_day9_A.bat -Encoding Ascii -Value @'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set CUDA_VISIBLE_DEVICES=%GPU_A%
call :run > run_day9_A.log 2>&1
git add -f results/plan_v2 results/scale/seeds.csv results/scale/gnn run_day9_A.log
git commit -m "day 9, block A: simulator, MHT and marker networks with the real box size, scalability sets"
call :push > run_day9_A_push.log 2>&1
echo done > run_day9_A.done
exit /b

:run
echo BLOCK A started %date% %time%
python -m sim.plan --model config/sensor_model.json --out data/plan_v2/ --seed-table results/plan_v2/seeds.csv || exit /b 1
python -m pipeline.plan_mht --sets data/plan_v2/val/ --hypotheses 100 300 1000 --orders 3 --choose-h --out results/plan_v2/M2/ || exit /b 1
python -m pipeline.plan_mht --sets "data/plan_v2/G*" "data/plan_v2/E*" "data/plan_v2/V*" --hypotheses 100 --out results/plan_v2/M3/ || exit /b 1
python -m pipeline.plan_mht --sets "data/plan_v2/S1_*" --hypotheses 100 300 1000 --out results/plan_v2/M3/ || exit /b 1
python -m pipeline.plan_failure_map --results results/plan_v2/M3/ --out results/plan_v2/M3/ --data data/plan_v2/ || exit /b 1
python -m gnn.train_sim --train-dir data/plan_v2/train/ --val-dir data/plan_v2/val/ --sensor-calib config/sensor_calib.json --graph detection --features full --edges full --hops 2 --seeds 3 --plan-dir data/plan_v2/ --plan-tests "G0_*" "E1_*" "E3_*" "S1_*" --save-estimates --out results/plan_v2/M4/gnn/ || exit /b 1
python -m pipeline.plan_oracle --sets "G0_*" "E1_*" "E3_*" "S1_*" --data data/plan_v2/ --mht results/plan_v2/M3/ --gnn results/plan_v2/M4/gnn/detection_full_full_h2/ --out results/plan_v2/M4/ || exit /b 1
python -m gnn.train_sim --train-dir data/plan_v2/train/ --val-dir data/plan_v2/val/ --sensor-calib config/sensor_calib.json --graph detection --features full --edges full --hops 2 --seeds 3 --out results/plan_v2/M4/gnn/ --eval-only --test REAL data/survey_all/ config/ground_truth_positions.json results/evaluation_all/scenes.csv --results-name real_all || exit /b 1
python -m sim.scaling --model config/sensor_model.json --out data/scale/ --seed-table results/scale/seeds.csv || exit /b 1
python -m gnn.train_sim --train-dir data/plan_v2/train/ data/scale/train/ --val-dir data/plan_v2/val/ data/scale/val/ --sensor-calib config/sensor_calib.json --graph detection --features full --edges full --hops 2 --seeds 3 --out results/scale/gnn/ || exit /b 1
python -m gnn.train_sim --train-dir data/plan_v2/train/ data/scale/train/ --val-dir data/plan_v2/val/ data/scale/val/ --sensor-calib config/sensor_calib.json --graph detection --features full --edges full --hops 2 --seeds 3 --out results/scale/gnn/ --eval-only --test REAL data/survey_all/ config/ground_truth_positions.json results/evaluation_all/scenes.csv --results-name real_all || exit /b 1
echo BLOCK A DONE %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
'@
Set-Content -Path run_day9_Y.bat -Encoding Ascii -Value @'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set CUDA_VISIBLE_DEVICES=%GPU_Y%
call :run > run_day9_Y.log 2>&1
if not exist results\yolo_gnn mkdir results\yolo_gnn
if exist data\yolo_real\fig_yolo_real.jpg copy /Y data\yolo_real\fig_yolo_real.jpg results\yolo_gnn\fig_yolo_real.jpg > nul
git add -f results/yolo_seg/box_seg/weights/best.pt results/yolo_seg/box_seg/args.yaml results/yolo_seg/box_seg/results.csv results/yolo_seg/box_seg/*.png results/yolo_gnn/fig_yolo_real.jpg run_day9_Y.log
git commit -m "day 9, block Y: YOLO-seg trained on simulated images only, all the real frames segmented"
call :push > run_day9_Y_push.log 2>&1
echo done > run_day9_Y.done
exit /b

:run
echo BLOCK Y started %date% %time%
python -m yolo_gnn.train_seg --data-dir data/yolo_sim --sensor-model config/sensor_model.json --out results/yolo_seg --device 0 || exit /b 1
python -m yolo_gnn.segment_real --runs "data/survey_all/*" --weights results/yolo_seg/box_seg/weights/best.pt --cache data/yolo_real/cache --device 0 || exit /b 1
echo BLOCK Y DONE %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
'@
Set-Content -Path run_day9_S.bat -Encoding Ascii -Value @'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_day9_S.log 2>&1
if not exist results\scale mkdir results\scale
if exist data\yolo_sim_eval\fig_yolo_sim.jpg copy /Y data\yolo_sim_eval\fig_yolo_sim.jpg results\scale\fig_yolo_sim.jpg > nul
git add -f results/scale run_day9_S.log
git commit -m "day 9, block S: MHT, marker network and YOLO + marker network on simulated scenes up to 50 robots and 50 boxes"
call :push > run_day9_S_push.log 2>&1
exit /b

:run
echo BLOCK S started %date% %time%
python -m yolo_gnn.segment_sim --sets data/scale/tune "data/scale/S*" --weights results/yolo_seg/box_seg/weights/best.pt --cache data/yolo_sim_eval/cache --readings 3 --device 0 || exit /b 1
python -m yolo_gnn.evaluate_sim --tune-set data/scale/tune --sets "data/scale/S*" --gnn plan=results/plan_v2/M4/gnn/detection_full_full_h2 --gnn scale=results/scale/gnn/detection_full_full_h2 --cache data/yolo_sim_eval/cache --readings 3 --out results/scale/ || exit /b 1
echo BLOCK S DONE %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
'@
Set-Content -Path run_day9_H.bat -Encoding Ascii -Value @'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_day9_H.log 2>&1
git add -f results/yolo_gnn results/yolo_gnn_same_run run_day9_H.log
git commit -m "day 9, block H: YOLO + marker network on all the real scenes, thresholds by two-fold cross-validation over the layouts"
call :push > run_day9_H_push.log 2>&1
exit /b

:run
echo BLOCK H started %date% %time%
python -m yolo_gnn.evaluate --survey-dir data/survey_all/ --runs "data/survey_all/*" --gnn-dir results/plan_v2/M4/gnn/detection_full_full_h2 --cache data/yolo_real/cache --calibrate-scope other-layouts --mht results/evaluation_all/scenes.csv --out results/yolo_gnn/ || exit /b 1
python -m yolo_gnn.evaluate --survey-dir data/survey_all/ --runs "data/survey_all/*" --gnn-dir results/plan_v2/M4/gnn/detection_full_full_h2 --cache data/yolo_real/cache --calibrate-scope same-run --mht results/evaluation_all/scenes.csv --out results/yolo_gnn_same_run/
python -m tools.real_tables --levels results/scene_levels.csv --method "MHT=results/evaluation_all/scenes.csv@mht_sensor" --method "marker=results/yolo_gnn/test_scenes_marker.csv" --method "yolo_gnn=results/yolo_gnn/test_scenes_full.csv" --errors --pair yolo_gnn:MHT --pair yolo_gnn:marker --pair marker:MHT --oracle MHT:yolo_gnn > results/yolo_gnn/real_tables.txt
echo BLOCK H DONE %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
'@
Start-Process cmd.exe -ArgumentList '/k','run_day9_all.bat'
