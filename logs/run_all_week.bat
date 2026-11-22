@echo off
rem Everything of this week, one after the other (same GPU and CPUs):
rem   1. run_imgmlp2.bat      imgmlp round 2 (sensor-model ablation, noise randomization, miss-probability curve)  ~22 h
rem   2. run_imgmlp_grid.bat  imgmlp variants on the robots x boxes grid 2-50 (nothing trained)                    ~2-3 h
rem   3. run_grid.bat         MHT, marker network, YOLO + marker network on the same grid                          MHT is the slow part
rem Every script commits and pushes its own results; one failing does not stop the next.
cd /d "%~dp0"
call run_imgmlp2.bat
call run_imgmlp_grid.bat
call run_grid.bat
