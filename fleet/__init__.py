"""A fleet of robots that follows a captain, scans the whole arena and reports how many targets there are and where.

    fleet/world.py     arena, obstacles, targets (random or from a file), grids
    fleet/sensing.py   simulated camera (field of view, range, occlusion, noise, misses, false detections) and
                       the ToF on the gimbal (one beam, 20 Hz)
    fleet/brain.py     the shared map and the goals of every robot: captain on lanes, followers in formation or
                       spread by Voronoi coverage (Lloyd) around the captain; A* paths on the map
    fleet/fusion.py    detections of every robot -> objects (count, positions): online Tracker (tentative /
                       confirmed / deleted) and batch objects(); the score against the truth
    fleet/live.py      the live window (python -m fleet.sim --live)
    fleet/sim.py       the simulation (python -m fleet.sim)
"""
