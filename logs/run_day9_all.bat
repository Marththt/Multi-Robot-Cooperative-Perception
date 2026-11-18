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
