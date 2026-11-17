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
