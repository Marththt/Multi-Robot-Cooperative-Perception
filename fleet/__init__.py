"""A fleet of robots that follows a captain, scans the whole arena and reports how many targets there are and where.

    fleet/world.py     arena, obstacles, targets (random or from a file), grids
    fleet/sensing.py   simulated camera (field of view, range, occlusion, noise, misses, false detections)
    fleet/brain.py     the shared map and the goals of every robot: captain on lanes, followers in formation or
                       spread by Voronoi coverage (Lloyd) around the captain; A* paths on the map
    fleet/fusion.py    detections of every robot -> objects (count, positions), and the score against the truth
    fleet/sim.py       the simulation (python -m fleet.sim)
"""
