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
