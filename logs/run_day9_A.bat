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
