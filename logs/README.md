# Logs of the PC1 runs

Logs of the runs that produced the results in `results/` (day 6, day 9, day 10 blocks, M4 on the real scenes), kept
as a record of where the numbers come from.

The only script still in use is `run_imgmlp.bat` (whole image + measurements -> MLP -> GNN, `imgmlp/`). It `cd`s to
its own folder, so it is copied to the repository root and started by `docs/paste_in_powershell_PC1_imgmlp.txt`.
The second round, `run_imgmlp2.bat` (sensor-model ablation, noise randomization, declared uncertainty, miss-probability
curve; it reuses the frame caches of the first), is started, together with `run_imgmlp_grid.bat` (imgmlp variants on the robots x boxes grid 2-50) and
`run_grid.bat` (MHT, marker network, YOLO on the same grid), by `run_all_week.bat` from
`docs/paste_in_powershell_PC1_imgmlp2.txt`.
