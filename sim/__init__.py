"""Measurement-level simulator of the multi-robot box-localization setup.

Library modules:
  geometry   triangular-prism boxes, obstacles, ray casting for occlusion
  model      the sensor model (fitted on real runs) and its JSON file
  scene      scenes: boxes, obstacles and camera poses (recreated from real runs or generated at random)
  sensor     one capture of one robot: visibility, detection, measurement error, false detections
  writer     writes simulated runs in the same JSONL format as the real runs

Scripts (run from the repository root):
  python -m sim.calibrate   fit the sensor model on real layouts
  python -m sim.generate    simulated runs: replays of real layouts or random scenes from sim/specs/
  python -m sim.fidelity    compare real layouts with their simulated replays

The simulated runs go through the same code as the real ones (pipeline.evaluate_runs, gnn.train).
"""
