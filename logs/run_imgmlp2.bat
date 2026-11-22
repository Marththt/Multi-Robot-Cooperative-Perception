@echo off
rem imgmlp, second round (paper): can the GNN work from rho and theta alone, without a fitted sensor model?
rem   1. measure   sensor model, with and without image (the clean ablation of the image), graph A and B
rem   2. rand      domain randomization of the measurement error (no sensor model), with and without the declared
rem                uncertainty as input, with and without image; real scenes also without calibration (real_raw)
rem   3. pd_curve  fixed noise, miss probability 0.20 / 0.35 / 0.50 (0.10 is sweep_k/noisy_fixed_k0)
rem Results are committed and pushed after every block. Everything trained in simulation; real scenes are test only.
rem Frame caches of run_imgmlp.bat are reused. Expected time on PC1: about 1 day.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
call :run > run_imgmlp2.log 2>&1
git add -f results/imgmlp run_imgmlp2.log
git commit -m "imgmlp round 2: sensor-model ablation, noise randomization, declared uncertainty, miss-probability curve"
call :push > run_imgmlp2_push.log 2>&1
exit /b

:run
echo IMGMLP2 started %date% %time%
set C=data/imgmlp
set R=results/imgmlp
set TRAIN=--train data/plan_v2/train %C%/plan_v2_train
set COMMON=--val data/plan_v2/val %C%/plan_v2_val --real data/survey_all/ %C%/real --sensor-calib config/sensor_calib.json --seeds 3 --real-raw
rem declared uncertainty on the real scenes: generic (NoiseModel defaults, not fitted), calib (values of the
rem calibration, config/sensor_calib.json and results/sensor_model_dump), and calib halved / doubled
set D_CALIB=range_sd_m=0.037 range_sd_rel=0 bearing_sd_deg=2.0 scale_bias_sd=0.02 bearing_bias_sd_deg=1.0 miss_prob=0.36 false_rate=0.15
set D_HALF=range_sd_m=0.0185 range_sd_rel=0 bearing_sd_deg=1.0 scale_bias_sd=0.01 bearing_bias_sd_deg=0.5 miss_prob=0.18 false_rate=0.075
set D_DOUBLE=range_sd_m=0.074 range_sd_rel=0 bearing_sd_deg=4.0 scale_bias_sd=0.04 bearing_bias_sd_deg=2.0 miss_prob=0.70 false_rate=0.30
set DECL=--declare generic --declare calib %D_CALIB% --declare half %D_HALF% --declare double %D_DOUBLE%

for %%F in (plan_v2_train plan_v2_val real) do if not exist %C%\%%F\frames.npy (echo missing frame cache %C%\%%F, run run_imgmlp.bat first & exit /b 1)

rem ---- 1. sensor model: k = 0 (no image) against k = 8, graph A and B
python -m imgmlp.train %TRAIN% %COMMON% --measure sensor --geometry fixed --k 0 --out %R%/measure || exit /b 1
python -m imgmlp.train %COMMON% --measure sensor --geometry fixed --k 8 --eval-only --out %R%/measure || exit /b 1
python -m imgmlp.train %TRAIN% %COMMON% --measure sensor --geometry fixed --graph B --k 0 --out %R%/measure || exit /b 1
python -m imgmlp.train %TRAIN% %COMMON% --measure sensor --geometry fixed --graph B --k 8 --out %R%/measure || exit /b 1
call :save "imgmlp round 2: sensor model with and without image"

rem ---- 2. no sensor model: randomized measurement error
python -m imgmlp.train %TRAIN% %COMMON% --measure noisyrand --geometry fixed --k 0 --out %R%/rand || exit /b 1
python -m imgmlp.train %TRAIN% %COMMON% --measure noisyrand --geometry fixed --k 0 --unc %DECL% --out %R%/rand || exit /b 1
call :save "imgmlp round 2: noise randomization, declared uncertainty"
python -m imgmlp.train %TRAIN% %COMMON% --measure noisyrand --geometry fixed --k 8 --out %R%/rand || exit /b 1
call :save "imgmlp round 2: noise randomization with image"

rem ---- 3. what the sensor model brings: miss probability of the training noise
for %%P in (020 035 050) do (
  python -m imgmlp.train %TRAIN% %COMMON% --measure noisy --geometry fixed --k 0 --noise-miss-prob 0.%%P --tag miss%%P --out %R%/pd_curve || exit /b 1
)
python -m imgmlp.report --root %R% --out %R%/report.csv
echo IMGMLP2 DONE %date% %time%
exit /b 0

:save
python -m imgmlp.report --root %R% --out %R%/report.csv
git add -f results/imgmlp
git commit -m %1
git pull --no-rebase --no-edit
git push
exit /b 0

:push
git pull --no-rebase --no-edit && git push && exit /b 0
timeout /t 120 /nobreak
git pull --no-rebase --no-edit && git push
exit /b
