@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not exist results\evaluation_all\scenes.csv goto :missing
if not exist config\sensor_model.json goto :missing
if not exist config\sensor_model_fidelity.json goto :missing
if not exist data\sim_replay_all\ground_truth.json goto :missing
if exist run_day9_A.done del run_day9_A.done
if exist run_day9_Y.done del run_day9_Y.done
for /f %%i in ('python -c "import torch; print(torch.cuda.device_count())"') do set NGPU=%%i
set GPU_A=0
set GPU_Y=0
if %NGPU% GEQ 2 set GPU_A=1
start "day9 A (marker network)" cmd /c run_day9_A.bat
start "day9 Y (YOLO)" cmd /c run_day9_Y.bat
start "day9 F (simulator fidelity)" cmd /c run_day9_F.bat
echo Blocks A, Y and F started on %NGPU% GPU(s) %date% %time%. Logs: run_day9_A.log, run_day9_Y.log, run_day9_F.log
:wait
if not exist run_day9_A.done (timeout /t 120 /nobreak > nul & goto :wait)
if not exist run_day9_Y.done (timeout /t 120 /nobreak > nul & goto :wait)
echo Blocks A and Y done %date% %time%, now block S (validation in simulation, up to 50 robots)
call run_day9_S.bat
echo Block S done %date% %time%, now block H (YOLO + marker network on the real scenes)
call run_day9_H.bat
echo ALL DONE %date% %time%: see run_day9_H.log and results\yolo_gnn\real_tables.txt
goto :eof
:missing
echo Block 0 did not reach the simulated replays: send run_day9_0.log
