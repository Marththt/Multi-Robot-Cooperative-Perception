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
