@echo off
rem Robots x boxes grid (2, 4, 8, 16, 32, 50 each, 36 cells, 100 scenes per cell): scenes (sim/scaling.py),
rem frames rendered and segmented (yolo_gnn/segment_sim.py), MHT + marker network + YOLO (yolo_gnn/evaluate_sim.py),
rem F1 heat maps (tools/fig_grid_f1.py). Start it after run_imgmlp.bat has finished (same GPU and CPUs).
rem Rerunnable: the frames already in the cache and the MHT rows already in results/scale_grid/mht are reused.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_grid.log 2>&1
git add -f results/scale_grid results/scale/seeds.csv run_grid.log
git commit -m "robots x boxes grid (2-50): F1 of MHT, marker network and YOLO + marker network, heat maps"
call :push > run_grid_push.log 2>&1
exit /b

:run
echo GRID started %date% %time%
if not exist data\scale\G_n50_b50\geometry.jsonl python -m sim.scaling --sets G --test-runs 100 --out data/scale/ || exit /b 1
echo scenes done %date% %time%
python -m yolo_gnn.segment_sim --sets "data/scale/G_*" --weights results/yolo_seg/box_seg/weights/best.pt --cache data/yolo_sim_eval/cache --device 0 || exit /b 1
echo frames done %date% %time%
python -m yolo_gnn.evaluate_sim --tune-set data/scale/tune --sets "data/scale/G_*" --gnn plan=results/plan_v2/M4/gnn/detection_full_full_h2 --gnn scale=results/scale/gnn/detection_full_full_h2 --cache data/yolo_sim_eval/cache --mht-density 0.2 --reuse-mht --out results/scale_grid/ || exit /b 1
echo evaluation done %date% %time%
python -m tools.fig_grid_f1 --summary results/scale_grid/summary.csv --out results/scale_grid/
python -m tools.fig_grid_f1 --summary results/scale_grid/summary.csv --metric correct_pct --out results/scale_grid/
echo GRID finished %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
