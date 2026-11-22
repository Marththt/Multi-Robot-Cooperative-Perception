@echo off
rem imgmlp variants on the robots x boxes grid (2, 4, 8, 16, 32, 50 each, 36 cells, 100 scenes per cell): the G sets
rem of sim/scaling.py (the same as run_grid.bat), their frames (imgmlp/frames.py, for k > 0), evaluation of the trained
rem variants (imgmlp/grid.py, nothing trained), F1 and correct-scene heat maps. Start it after run_imgmlp2.bat.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_imgmlp_grid.log 2>&1
git add -f results/imgmlp/grid run_imgmlp_grid.log
git commit -m "imgmlp on the robots x boxes grid (2-50): correct scenes and F1 of the trained variants, heat maps"
call :push > run_imgmlp_grid_push.log 2>&1
exit /b

:run
echo IMGMLP GRID started %date% %time%
set R=results/imgmlp
if not exist data\scale\G_n50_b50\geometry.jsonl python -m sim.scaling --sets G --test-runs 100 --out data/scale/ || exit /b 1
echo scenes done %date% %time%
for /d %%S in (data\scale\G_*) do (
  if not exist data\imgmlp\grid\%%~nxS\frames.npy python -m imgmlp.frames sim --set data/scale/%%~nxS --out data/imgmlp/grid/%%~nxS || exit /b 1
)
echo frames done %date% %time%
set V=%R%/sweep_k/noisy_fixed_k0_64x36 %R%/sweep_k/noisy_fixed_k8_64x36
set V=%V% %R%/measure/sensor_fixed_k0_64x36 %R%/measure/sensor_fixed_k8_64x36 %R%/measure/sensor_fixed_B_k0_64x36 %R%/measure/sensor_fixed_B_k8_64x36
set V=%V% %R%/rand/noisyrand_fixed_k0_64x36 %R%/rand/noisyrand_fixed_k0_64x36_unc %R%/rand/noisyrand_fixed_k8_64x36
rem declared uncertainty for the --unc variant: the values of the calibration (as "calib" in run_imgmlp2.bat)
python -m imgmlp.grid --variants %V% --sets "data/scale/G_*" --cache-root data/imgmlp/grid --declare range_sd_m=0.037 range_sd_rel=0 bearing_sd_deg=2.0 scale_bias_sd=0.02 bearing_bias_sd_deg=1.0 miss_prob=0.36 false_rate=0.15 --out %R%/grid/ || exit /b 1
echo evaluation done %date% %time%
set M=sensor_fixed_k0_64x36 sensor_fixed_k8_64x36 noisyrand_fixed_k0_64x36 noisyrand_fixed_k0_64x36_unc noisyrand_fixed_k8_64x36
for %%T in (f1_mean correct_pct) do (
  python -m tools.fig_grid_f1 --summary %R%/grid/summary.csv --metric %%T --methods %M% --reference sensor_fixed_k0_64x36 --tag imgmlp --out %R%/grid/
  python -m tools.fig_grid_f1 --summary %R%/grid/summary.csv --metric %%T --methods noisy_fixed_k0_64x36 noisy_fixed_k8_64x36 sensor_fixed_B_k0_64x36 sensor_fixed_B_k8_64x36 --reference noisy_fixed_k0_64x36 --tag imgmlp_ablation --out %R%/grid/
)
echo IMGMLP GRID finished %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
