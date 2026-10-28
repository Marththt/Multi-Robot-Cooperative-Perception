"""Synthetic multi-robot simulation for testing the aggregation baselines
WITHOUT hardware (Topic 4, high-level part).

Why start here and not from the robot
----------------------------------------
The aggregation baselines (see aggregation.py) and a possible graph/GNN
extension can be designed and validated on synthetic data: there is no
need to wait for the real robot to be ready before starting the research
part (the part closest to the professor's work on Byzantine-robust
aggregation in GNNs). The real perception code (perception.py, patrol.py,
yolo_server.py/yolo_client.py) stays ready and parked for the hardware
validation phase, which the professor's document places later in the
10-week plan, not at the beginning.

What this is NOT: it is not a physical simulation (no MuJoCo/Gazebo, no
robot kinematics). It simulates only the signal the baselines care about:
N robots observing the same zone at the same instant, each with its own
baseline sensor error, optionally with a systematic fault on top (the same
FaultInjector as perception.py, to stay consistent when we move to real
data).

Model
-----
At each instant t the zone has a "ground truth": one object present (a
class) or none. Each robot produces a local detection: first an "honest"
sensor error (TRUE_POSITIVE_PROB / FALSE_POSITIVE_PROB, independent of
faults -- even a healthy robot is wrong now and then), then, only for the
robots designated as faulty, the systematic FaultInjector corruption
(dropout/mislabel) on top of the honest detection.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vf.perception.perception import CLASSES, FaultInjector

TRUE_POSITIVE_PROB = 0.85   # if the object is there, prob. of detecting it (honest)
FALSE_POSITIVE_PROB = 0.05  # if the object is NOT there, prob. of "seeing" one


@dataclass
class Observation:
    robot_id: str
    class_name: Optional[str]     # None = "no object detected"
    confidence: float
    fault_injected: bool
    dropped: bool


def _honest_detection(rng, truth: Optional[str]) -> Tuple[Optional[str], float]:
    """Baseline sensor error, BEFORE any systematic fault.
    No robot, not even a healthy one, is an oracle."""
    if truth is not None:
        if rng.random() < TRUE_POSITIVE_PROB:
            return truth, rng.uniform(0.5, 0.95)
        return None, 0.0
    else:
        if rng.random() < FALSE_POSITIVE_PROB:
            return rng.choice(CLASSES), rng.uniform(0.25, 0.6)
        return None, 0.0


def simulate_tick(t, truth: Optional[str], robot_ids: List[str],
                  injectors: Dict[str, FaultInjector],
                  seed=0) -> Dict[str, Observation]:
    """One instant: fixed ground truth, one observation per robot.

    `injectors` is ONE FaultInjector per robot, created once for the whole
    experiment by simulate_experiment (not here) and reused at every tick:
    it is the object itself, with the internal state of its random
    generator advancing call after call, that makes the fault probabilistic
    over time. A new injector at every tick (fixed bug: that is exactly
    what used to happen) instead ALWAYS gives the same first random number
    -- the fault becomes a fixed fate for that robot ("always dropped" or
    "never", for the whole experiment) instead of happening with the
    configured probability.
    """
    observations = {}
    for robot_id in robot_ids:
        # A string, not hash((...)): hash() of a tuple containing strings
        # depends on PYTHONHASHSEED (randomized per process) -- with a
        # string random.Random() uses a stable hash, so the seed is truly
        # reproducible across different runs. The seed strings ("onesto",
        # "verita") are kept from the original code on purpose: changing
        # them would change the random sequences and break reproducibility
        # of earlier results.
        rng = random.Random(f"{seed}-onesto-{t}-{robot_id}")
        class_name, confidence = _honest_detection(rng, truth)

        injector = injectors[robot_id]
        if class_name is not None:
            class_name, confidence, fault, dropped = injector.apply(
                class_name, confidence)
        else:
            # The current model does not mislabel "no object" into an
            # injected false positive: that is a future extension, not
            # needed for the first accuracy curves.
            fault, dropped = False, False

        observations[robot_id] = Observation(
            robot_id=robot_id, class_name=class_name, confidence=confidence,
            fault_injected=fault, dropped=dropped)
    return observations


def simulate_experiment(n_tick, robot_ids, faulty_robot_ids,
                        object_present_prob=0.5, dropout_prob=None,
                        mislabel_prob=None, seed=0):
    """Generate n_tick independent instants with random ground truth.
    Returns list[(truth, observations_dict)]."""
    # One injector per robot, created HERE only once for the whole
    # experiment (see the comment in simulate_tick for why).
    injectors = {
        robot_id: FaultInjector(
            robot_id=robot_id, faulty_robot_ids=faulty_robot_ids,
            dropout_prob=dropout_prob, mislabel_prob=mislabel_prob,
            seed=seed)
        for robot_id in robot_ids
    }

    truth_rng = random.Random(f"{seed}-verita")
    experiment = []
    for t in range(n_tick):
        if truth_rng.random() < object_present_prob:
            truth = truth_rng.choice(CLASSES)
        else:
            truth = None
        observations = simulate_tick(t, truth, robot_ids, injectors, seed)
        experiment.append((truth, observations))
    return experiment
