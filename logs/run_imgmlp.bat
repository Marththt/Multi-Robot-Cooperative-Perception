@echo off
rem Whole image + measurements -> MLP -> GNN (imgmlp/): frame caches, sweep of k, geometry and measurement ablations,
rem graph A and B, look-randomized images (sim -> real gap), linear probes. Everything trained in simulation; the real scenes are test only.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_imgmlp.log 2>&1
git add -f results/imgmlp run_imgmlp.log
git commit -m "imgmlp: whole image + measurements -> MLP -> GNN, sweep of k, geometry and measurement ablations, probes"
call :push > run_imgmlp_push.log 2>&1
exit /b

:run
echo IMGMLP started %date% %time%
set C=data/imgmlp
set R=results/imgmlp
set COMMON=--val data/plan_v2/val %C%/plan_v2_val --real data/survey_all/ %C%/real --sensor-calib config/sensor_calib.json --seeds 3

rem ---- frame caches (128 x 72 grey; the model reduces them to --img-size)
if not exist %C%\plan_v2_train\frames.npy python -m imgmlp.frames sim --set data/plan_v2/train --max-scenes 20000 --out %C%/plan_v2_train || exit /b 1
if not exist %C%\plan_v2_val\frames.npy python -m imgmlp.frames sim --set data/plan_v2/val --out %C%/plan_v2_val || exit /b 1
if not exist %C%\plan_v2_train_random\frames.npy python -m imgmlp.frames sim --set data/plan_v2/train --max-scenes 20000 --look random --out %C%/plan_v2_train_random || exit /b 1
if not exist %C%\real\frames.npy python -m imgmlp.frames real --runs "data/survey_all/*" --out %C%/real || exit /b 1

rem ---- how many features: k = 0 (no image), 2, 4, 8, 16, 32; fixed geometry, noisy measurements; graph A and B
for %%G in (A B) do (
  for %%K in (0 2 4 8 16 32) do (
    python -m imgmlp.train --train data/plan_v2/train %C%/plan_v2_train %COMMON% --measure noisy --geometry fixed --graph %%G --k %%K --out %R%/sweep_k || exit /b 1
  )
)

rem ---- geometry: strict (no geometric layer), with and without image
for %%G in (A B) do (
  for %%K in (0 8) do (
    python -m imgmlp.train --train data/plan_v2/train %C%/plan_v2_train %COMMON% --measure noisy --geometry strict --graph %%G --k %%K --epochs 60 --patience 12 --out %R%/strict || exit /b 1
  )
)

rem ---- measurements: ideal, and the fitted ArUco sensor model of the set
for %%M in (ideal sensor) do (
  python -m imgmlp.train --train data/plan_v2/train %C%/plan_v2_train %COMMON% --measure %%M --geometry fixed --k 8 --out %R%/measure || exit /b 1
)

rem ---- image resolution: 128 x 72
python -m imgmlp.train --train data/plan_v2/train %C%/plan_v2_train %COMMON% --measure noisy --geometry fixed --k 8 --img-size 128 72 --out %R%/resolution || exit /b 1

rem ---- sim -> real gap: randomized look of the training images
python -m imgmlp.train --train data/plan_v2/train %C%/plan_v2_train_random %COMMON% --measure noisy --geometry fixed --k 8 --out %R%/look_random || exit /b 1

rem ---- linear probes of the image features
for %%V in (fixed fixed_B) do (
  for %%K in (2 4 8 16 32) do (
    python -m imgmlp.probe --variant %R%/sweep_k/noisy_%%V_k%%K_64x36 --cache %C%/plan_v2_val --test val
    python -m imgmlp.probe --variant %R%/sweep_k/noisy_%%V_k%%K_64x36 --cache %C%/real --test real
  )
)
echo IMGMLP DONE %date% %time%
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
