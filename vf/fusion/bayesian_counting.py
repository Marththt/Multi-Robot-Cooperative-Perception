"""Discrete Bayesian counting (level 3b of the professor's document,
Section 3.2/3.3) -- "the best classical method for the class-only counting
problem... primary Part 1 deliverable". It is NOT a baseline like those in
aggregation.py: those produce ONE vote per robot (a class, or none) and
fuse votes; this one estimates a number -- how many objects of a class are
REALLY in a zone -- even when there is MORE than one.

Why a separate module
------------------------------------
zone+class merge (level 1: union/majority/Chair-Varshney/median above)
implicitly assumes at most one object per (zone, class): one vote per
robot becomes a single decision. The document is explicit that this
breaks down when a zone really contains TWO objects of the same class --
exactly the case Bayesian counting is designed to handle, and on which it
must be tested on purpose (Section 3.5, point 2: "tested on layouts with
more than one same-class object per zone -- specifically to demonstrate
the case where zone+class merge breaks down and this method does not").

Single snapshot, not continuous patrol
-----------------------------------------
The document is explicit (Section 3.2, note on Discrete Bayesian
counting): "exactly one observation is counted per robot per static trial
(a single snapshot, not one per video frame) -- since the objects are
static, treating repeated frames of the same unchanged scene as
independent new evidence would inflate confidence without adding real
information." So the input here is ONE count per robot per trial (how
many objects of this class that robot detected in THAT snapshot), not a
stream of accumulated events as in patrol.py/fondi_log_reali.py --
consistent with the document's "trial" experimental setup (place, snap,
record), which differs from the continuous patrolling already built (that
remains useful for the "in the field" comparison mentioned separately,
but it is a different experiment from the one Bayesian counting needs).

Likelihood model -- declared, not hidden
------------------------------------------------------------
The document says TO DO a Bayesian update with the fault-model likelihood
(Section 7), but does not give the exact likelihood formula for a count
(it gives p_D(x) for a single binary detection instead). The choice here,
explicitly:

    P(n_i | k) = Binomial(n_i; k, p_d_i)

i.e. each of the k TRUE objects present is detected independently by
robot i with probability p_d_i (its p_D(x) for the geometry of this
snapshot -- see modello_guasto.py, to be built once real calibration data
exist, Week 3-4). n_i > k has likelihood 0 under this model -- a declared
SIMPLIFICATION: it does not yet model false positives (a robot wrongly
counting MORE objects than there are). Reason for doing it this way for
now: under p_D=1 (the document's fault-free stage) this collapses EXACTLY
to "simple counting" -- P(n_i=k|k)=1, every other k has likelihood 0 for
at least one robot that counted correctly, so the posterior is a point
mass on k -- the property the document explicitly asks for the fault-free
stage. The extension with a false-alarm term (e.g. convolution with a
Poisson) is natural once Week 3-4 provides a real rate to calibrate; not
done here to stay within the scope of this session.
"""

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

K_MAX_DEFAULT = 5  # maximum number of objects per (zone, class) considered
                   # plausible -- tune it to your arena; None/too high slows
                   # things down needlessly (the sum over k costs O(k_max)).


def _comb(n: int, k: int) -> int:
    """Binomial coefficient n choose k, in pure Python -- NOT math.comb().
    math.comb() only exists from Python 3.8 on; some conda environments
    (typically kept on an older Python version for compatibility with
    other already installed packages, as here) do not have it, and
    "pip install comb" solves nothing: that name on PyPI is a completely
    unrelated CLI package, in no way connected to the standard math
    module. This reimplementation avoids the dependency entirely, without
    having to upgrade the whole conda environment just for this
    function."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _binom_pmf(n: int, k: int, p: float) -> float:
    if n > k or n < 0:
        return 0.0
    return _comb(k, n) * (p ** n) * ((1 - p) ** (k - n))


def bayesian_count(observed_counts: Dict[str, int],
                   p_d_per_robot: Dict[str, float],
                   k_max: int = K_MAX_DEFAULT,
                   prior: Optional[List[float]] = None,
                   p_d_default: float = 0.85) -> Dict[int, float]:
    """Posterior distribution P(k | observations) for k=0..k_max.

    observed_counts: robot_id -> how many objects of THIS class that robot
    detected in THIS snapshot (see the module docstring: one static
    snapshot, not a stream of events).

    p_d_per_robot: robot_id -> probability of detecting EACH object
    present, for this robot's geometry in this snapshot (Section 7.1:
    p_D(x) = p_D(occlusion, angle, distance) -- here already evaluated
    upstream, see modello_guasto.py). p_d_default is used ONLY for a robot
    missing from p_d_per_robot (useful in tests or until there is a
    calibration for that robot -- not a substitute for real data in the
    thesis).

    prior: P(k) for k=0..k_max, uniform by default (no prior knowledge
    about how many objects to expect in a zone)."""
    if prior is None:
        prior = [1.0 / (k_max + 1)] * (k_max + 1)
    if len(prior) != k_max + 1:
        raise ValueError("prior must have k_max+1 values")

    unnormalized_posterior = []
    for k in range(k_max + 1):
        likelihood = 1.0
        for robot_id, n_i in observed_counts.items():
            p_d = p_d_per_robot.get(robot_id, p_d_default)
            likelihood *= _binom_pmf(n_i, k, p_d)
            if likelihood == 0.0:
                break
        unnormalized_posterior.append(prior[k] * likelihood)

    total = sum(unnormalized_posterior)
    if total <= 0.0:
        # No k between 0 and k_max explains the observations with non-zero
        # probability (e.g. an n_i > k_max reported by mistake) -- fall
        # back to the prior instead of dividing by zero, and report it in
        # the log: it is a symptom that k_max is too low for the real
        # arena, not a normal case to ignore silently.
        log.warning("bayesian_count: no k in [0,%d] explains the "
                    "observations %s (p_d=%s) -- falling back to the prior. "
                    "If a p_d is 1.0 and the robots report different "
                    "counts, that is the cause: with p_d=1 a robot cannot "
                    "miss an object. Otherwise: k_max too "
                    "low.",
                    k_max, observed_counts, p_d_per_robot)
        return {k: prior[k] for k in range(k_max + 1)}
    return {k: v / total for k, v in enumerate(unnormalized_posterior)}


def map_count_estimate(observed_counts: Dict[str, int],
                       p_d_per_robot: Dict[str, float],
                       k_max: int = K_MAX_DEFAULT,
                       prior: Optional[List[float]] = None,
                       p_d_default: float = 0.85) -> int:
    """Point estimate: the most probable k a posteriori (MAP) -- handy
    when a single number is needed, not the whole distribution (e.g. to
    compare against a ground truth with metriche.py-style metrics)."""
    posterior = bayesian_count(observed_counts, p_d_per_robot,
                               k_max, prior, p_d_default)
    return max(posterior, key=posterior.get)


if __name__ == "__main__":
    # Quick manual inspection, not a real CLI (no real data to read until
    # there is a multi-robot snapshot pipeline with per-class counts --
    # see the module docstring).
    p_d = {"r0": 0.85, "r1": 0.85, "r2": 0.85}
    for label, obs in [
        ("all agree on 2", {"r0": 2, "r1": 2, "r2": 2}),
        ("one robot misses one", {"r0": 2, "r1": 1, "r2": 2}),
        ("all see 1", {"r0": 1, "r1": 1, "r2": 1}),
    ]:
        post = bayesian_count(obs, p_d)
        map_k = map_count_estimate(obs, p_d)
        print(f"{label}: observations={obs} -> MAP k={map_k}, "
              f"posterior={ {k: round(v, 3) for k, v in post.items()} }")
