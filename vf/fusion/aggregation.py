"""Class-vote fusion baselines: from the observations of N robots of the same zone (or object
instance) at the same instant to ONE decision, the class believed to be present, or None.

    baseline_single_robot          one robot only, no fusion (worst-case reference)
    baseline_0_union               trust every detection: the most frequent class wins
                                   (ties by the highest confidence)
    baseline_1_majority            majority vote, "no object" included
    baseline_chair_varshney        Chair-Varshney fusion, optimal under conditional independence
                                   of the robots; needs per-robot P_d / P_fa; extended to multi-class
    baseline_2_robust_statistics   geometric median (Weiszfeld) of per-robot evidence vectors over
                                   [classes..., none]: robust to a bounded number of arbitrary reports
    baseline_3a_trimmed_mean       coordinate-wise trimmed mean of the same vectors
    ReputationAggregator           robots weighted by how often they agreed with past decisions

Without faults (p_D = 1, no misclassification) union, majority and Chair-Varshney agree, which is a
useful correctness test. The multi-object localization methods (MHT, PHD/CPHD) are in
localization_methods.py.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vf.perception.perception import CLASSES


@dataclass
class Observation:
    """What one robot reports about one zone (or object instance) at one instant."""
    robot_id: str
    class_name: Optional[str]     # None = "no object detected"
    confidence: float
    fault_injected: bool
    dropped: bool


def baseline_single_robot(observations: Dict[str, Observation],
                          robot_id: Optional[str] = None) -> Optional[str]:
    """Spec Baseline 0: a single robot, no fusion. If robot_id is not
    given, the first one in iteration order is used -- deterministic, not
    chosen post-hoc as "the best" or "the worst"."""
    if robot_id is None:
        robot_id = next(iter(observations))
    return observations[robot_id].class_name


def baseline_0_union(observations: Dict[str, Observation]) -> Optional[str]:
    classes = [o.class_name for o in observations.values() if o.class_name is not None]
    if not classes:
        return None
    count = Counter(classes)
    maximum = count.most_common(1)[0][1]
    candidates = [c for c, n in count.items() if n == maximum]
    if len(candidates) == 1:
        return candidates[0]
    best, best_conf = None, -1.0
    for o in observations.values():
        if o.class_name in candidates and o.confidence > best_conf:
            best, best_conf = o.class_name, o.confidence
    return best


def baseline_1_majority(observations: Dict[str, Observation]) -> Optional[str]:
    votes = Counter(o.class_name for o in observations.values())  # None is a valid vote
    return votes.most_common(1)[0][0]


def _evidence_vector(o: Observation) -> List[float]:
    """Represent an observation as an evidence vector over
    [classes..., none] (last coordinate = "no object"): all the mass
    (=confidence) on the reported class, the rest implicitly towards
    "none". It is NOT a complete probability distribution (it may sum to
    less than 1): it is just enough structure for the geometric
    median/trimmed mean (designed for continuous vectors) to make sense on
    a categorical observation."""
    v = [0.0] * (len(CLASSES) + 1)
    if o.class_name is None or o.class_name not in CLASSES:
        v[-1] = 1.0
    else:
        v[CLASSES.index(o.class_name)] = o.confidence
        v[-1] = 1.0 - o.confidence
    return v


def _class_from_vector(v: List[float]) -> Optional[str]:
    best_idx = max(range(len(v)), key=lambda i: v[i])
    return None if best_idx == len(v) - 1 else CLASSES[best_idx]


def _geometric_median(vectors: List[List[float]], iterations: int = 50,
                      eps: float = 1e-6) -> List[float]:
    """Weiszfeld algorithm: the point minimizing the sum of Euclidean
    distances from the given vectors. Unlike the mean (sensitive to a
    single arbitrarily distant outlier -- exactly the "Byzantine robot"
    case of Section 7.4), the geometric median moves much less: that is
    the sense of "robust" here. Starting point: the mean; then a few
    reweighting iterations by the inverse distance from the current
    point."""
    n = len(vectors)
    dim = len(vectors[0])
    point = [sum(v[i] for v in vectors) / n for i in range(dim)]
    for _ in range(iterations):
        weights = []
        for v in vectors:
            d = math.sqrt(sum((v[i] - point[i]) ** 2 for i in range(dim)))
            weights.append(1.0 / d if d > eps else 1.0 / eps)
        weight_sum = sum(weights)
        new = [sum(w * v[i] for w, v in zip(weights, vectors)) / weight_sum
               for i in range(dim)]
        shift = math.sqrt(sum((new[i] - point[i]) ** 2 for i in range(dim)))
        point = new
        if shift < eps:
            break
    return point


def baseline_2_robust_statistics(observations: Dict[str, Observation]) -> Optional[str]:
    """Geometric median of the per-robot evidence vectors (see
    _geometric_median and _evidence_vector above): a robot reporting a vector
    far from the others moves the median much less than it would move a mean."""
    vectors = [_evidence_vector(o) for o in observations.values()]
    if not vectors:
        return None
    return _class_from_vector(_geometric_median(vectors))


def baseline_3a_trimmed_mean(observations: Dict[str, Observation],
                             trim_fraction: float = 0.2) -> Optional[str]:
    """Simpler alternative to the geometric median, equally valid
    according to the document (Section 7.4: "geometric median OR trimmed
    mean"): for each coordinate of the evidence vector, the most extreme
    values are discarded (a trim_fraction fraction from each side) and the
    remaining ones are averaged. Simpler to implement and to explain (no
    iterations), slightly less robust than the geometric median in theory
    but sufficient for a fair comparison in the thesis."""
    vectors = [_evidence_vector(o) for o in observations.values()]
    if not vectors:
        return None
    n = len(vectors)
    dim = len(vectors[0])
    k = int(trim_fraction * n)
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    mean = []
    for i in range(dim):
        values = sorted(v[i] for v in vectors)
        central = values[k: n - k] if k > 0 else values
        mean.append(sum(central) / len(central))
    return _class_from_vector(mean)


def baseline_chair_varshney(observations: Dict[str, Observation],
                            robot_params: Optional[Dict[str, Tuple[float, float]]] = None,
                            p_d_default: Tuple[float, float] = (0.85, 0.05)
                            ) -> Optional[str]:
    """Level 2.5 of the document (Section 7.3, Chair & Varshney 1986):
    PROVABLY OPTIMAL fusion under conditional independence of the sensors
    -- the theoretical ceiling against which both the heuristics
    (union/majority) and the GNN are compared.

        w_i = log(P_d,i / P_fa,i)          if robot i votes "detected"
        w_i = log((1-P_d,i) / (1-P_fa,i))  if robot i votes "not detected"
        decision = sign(sum_i w_i)

    The original paper is binary (presence/absence of ONE target). Here it
    is extended to multiple classes: for EACH candidate class, "the robot
    reported exactly this class" is treated as the binary "detected" vote
    for that hypothesis, a separate score is computed per class, and the
    class with the highest score is chosen -- only if positive (otherwise
    no hypothesis is "more likely than not", decision = no object).

    robot_params: robot_id -> (P_d, P_fa) of THAT robot, from real
    calibration (Week 3-4 of the plan) -- an EXPLICIT parameter, not
    guessed in here, because that is how the document treats them ("known
    or estimated... per robot"). p_d_default is used ONLY for a robot
    missing from robot_params (useful for tests or when not all robots
    have their own calibration yet -- do NOT use it as a substitute for a
    real calibration in the thesis)."""
    if robot_params is None:
        robot_params = {}
    scores = {}
    for class_name in CLASSES:
        total = 0.0
        for robot_id, o in observations.items():
            p_d, p_fa = robot_params.get(robot_id, p_d_default)
            p_d = min(max(p_d, 1e-6), 1 - 1e-6)
            p_fa = min(max(p_fa, 1e-6), 1 - 1e-6)
            if o.class_name == class_name:
                total += math.log(p_d / p_fa)
            else:
                total += math.log((1 - p_d) / (1 - p_fa))
        scores[class_name] = total
    if not scores:
        return None
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


class ReputationAggregator:
    """Baseline 3: keeps a per-robot weight across successive calls (it
    needs state, unlike baselines 0-2 which are pure functions) -- one
    instance per experiment, not per tick."""

    def __init__(self, robot_ids, learning_rate=0.1):
        self.weights = {r: 1.0 for r in robot_ids}
        self.learning_rate = learning_rate

    def decide(self, observations: Dict[str, Observation]) -> Optional[str]:
        scores = defaultdict(float)
        for robot_id, o in observations.items():
            scores[o.class_name] += self.weights.get(robot_id, 1.0)
        decision = max(scores.items(), key=lambda kv: kv[1])[0]

        for robot_id, o in observations.items():
            agrees = 1.0 if o.class_name == decision else 0.0
            current_weight = self.weights.get(robot_id, 1.0)
            new_weight = ((1 - self.learning_rate) * current_weight
                          + self.learning_rate * agrees * 2.0)
            self.weights[robot_id] = max(0.05, new_weight)  # never zero
        return decision