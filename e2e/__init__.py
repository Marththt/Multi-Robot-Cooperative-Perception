"""End-to-end network (model B): the camera images and poses of the robots -> the boxes on the floor, trained in
simulation only. No marker detection and no segmentation: the network learns everything from the images.

  geometry.py   floor grid, projection of the cells into the images, targets and decoding
  data.py       simulated sets (the scenes of sim/plan.py rendered by imgsim/render.py) and the data loader
  real.py       the real scenes in the same format
  model.py      image encoder, lift to the floor, graph neural network over the robots per cell, floor decoder
  train.py      training, threshold on the validation set, tests on simulated and real scenes
"""
