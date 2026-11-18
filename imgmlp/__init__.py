"""Whole image + measurements -> MLP -> detection-based GNN.

Every robot gives three things and nothing else: its whole camera image (only resized, grey), the distance and
bearing (rho, theta) of every object it measured (whatever the sensor), and its pose (x, y, psi). An MLP turns the
image into k features z; a GNN with one node per measured object fuses [rho, theta, x, y, psi, z] of all robots
into the number of boxes and their positions.

  frames.py   image caches: simulated frames rendered from the scene geometry, real frames of the 444 scenes
  measure.py  scenes: measurements (ideal, noisy, or from a sensor) + robot poses + image of every robot
  model.py    ImageMLP + ImageDetectionGNN, two graphs (A: z on every report node; B: robot nodes carrying z)
              and two geometry modes (fixed layer / strict)
  train.py    training in simulation, evaluation in simulation and on the real scenes
  probe.py    linear probe: what the k image features encode
"""
