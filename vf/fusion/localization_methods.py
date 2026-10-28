"""All the fusion methods of Section 3.2 of the professor's document,
with the same interface, to localize SEVERAL objects of ONE class in
ONE static shot:

    method(order, reports, poses, sensor, zones=None) -> dict(
        objects=[(x, y), ...], n=int, extra={...})

    order     robot_id of ALL the robots of the shot (even those that saw
              nothing: "I looked and it was not there" is information)
    reports   {robot_id: [array([x, y]), ...]} positions in the world frame
    poses     {robot_id: (x, y, yaw) of the camera, or None}
    sensor    SensorModel (sensor_model.py), the same for all

The black box receives the robots' OBSERVATIONS (distance, bearing,
class of every marker + the robot pose); the first step of every method,
measurements_by_class() below, turns them into the measurements above.

Methods kept (the simplest and the most complex of the document's table):
    zone_class      1  zones + class: at most one object per zone (the
                       zones are a grid of cells, sensor_model.grid_zones)
    union           1  clustering by position: every cluster of
                       measurements within threshold_m is one object
    mht             3  multi-hypothesis tracking, robots as scans
                       (data_association.py)
    phd, cphd       4  Random Finite Set filters (phd.py)

Metric: OSPA (Schuhmacher, Vo, Vo 2008), the standard distance between
two SETS of positions. It combines position error and count error: every
extra or missing object costs c metres. 0 = perfect.
"""

import math
from itertools import permutations

import numpy as np

from vf.fusion.data_association import mht
from vf.fusion.phd import cphd, phd
from vf.perception.object_position import global_position


def _all_reports(reports):
    points, robots = [], []
    for rid, zs in reports.items():
        for z in zs:
            points.append(np.asarray(z, float))
            robots.append(rid)
    return points, robots


def _cluster(points, threshold_m):
    """Same greedy clustering as object_position.cluster_by_position."""
    centers, members = [], []
    for i, p in enumerate(points):
        best, dmin = None, threshold_m
        for k, c in enumerate(centers):
            d = float(np.hypot(*(p - c / len(members[k]))))
            if d <= dmin:
                best, dmin = k, d
        if best is None:
            centers.append(p.copy())
            members.append([i])
        else:
            centers[best] = centers[best] + p
            members[best].append(i)
    return [(c / len(m), m) for c, m in zip(centers, members)]


def zone_class(order, reports, poses, sensor, zones=None):
    if not zones:
        raise ValueError("zone_class requires the zones (--regions)")
    per_zone = {}
    points, _ = _all_reports(reports)
    for p in points:
        for z in zones:
            if z["x_min"] <= p[0] < z["x_max"] and z["y_min"] <= p[1] < z["y_max"]:
                per_zone.setdefault(z["name"], []).append(p)
                break
    objects = [tuple(float(v) for v in np.mean(ps, axis=0)) for _, ps in sorted(per_zone.items())]
    return dict(objects=objects, n=len(objects), extra=dict(zones=sorted(per_zone)))


def union(order, reports, poses, sensor, zones=None, threshold_m=0.5):
    """Every cluster of measurements (greedy clustering within threshold_m)
    is one object, placed at the mean of its measurements."""
    points, _ = _all_reports(reports)
    kept = [c for c, _members in _cluster(points, threshold_m)]
    return dict(objects=[tuple(float(v) for v in c) for c in kept], n=len(kept), extra={})


METHODS = dict(zone_class=zone_class, union=union, mht=mht, phd=phd, cphd=cphd)
_TAKE_ZONES = ("zone_class", "union")


def run_method(name, order, reports, poses, sensor, zones=None):
    """Set the surveilled region of THIS shot (see
    SensorModel.set_coverage) and run the method."""
    sensor.set_coverage([poses.get(r) for r in order])
    fn = METHODS[name]
    if name in _TAKE_ZONES:
        return fn(order, reports, poses, sensor, zones=zones)
    return fn(order, reports, poses, sensor)


# ====================================================================
# From the robots' observations to the measurements Z
# ====================================================================

def measurement_from_observation(pose, distance, bearing):
    """World position z = [x + d cos(psi - beta), y + d sin(psi - beta)]
    of a marker seen at distance d and bearing beta (positive to the
    right) by a camera with pose (x, y, psi). Same formula as
    object_position.global_position, which it calls."""
    return np.asarray(global_position(pose, bearing, distance), float)


def measurements_by_class(robots, require_pose=True):
    """First step of every classical method: the observations of all the
    robots -> the measurements Z, per class.

    robots: {robot_id: dict(pose=(x, y, psi) or None,
                            observations=[dict(class_name, distance, bearing,
                                               pose=None, position=None), ...])}
      distance [m] and bearing [rad] as measured by the robot; the
      optional per-observation `pose` is the camera pose of the frame the
      marker was seen in (default: the robot pose). `position` is only a
      fallback for data without a bearing (trials recorded before the
      bearing was saved).
    Robots without a pose are left out (require_pose=True, live use):
    their observations cannot be put in the world frame and their field of
    view is unknown. With require_pose=False (old saved trials) they are
    kept with pose None, using the stored positions.

    Returns (used, poses, {class_name: {robot_id: [z, ...]}}, n_fallback)."""
    used = sorted(rid for rid, r in robots.items()
                  if r.get("pose") is not None or not require_pose)
    poses = {rid: (tuple(robots[rid]["pose"]) if robots[rid].get("pose") is not None else None)
             for rid in used}
    per_class = {}
    n_fallback = 0
    for rid in used:
        for ob in robots[rid].get("observations", []):
            class_name = ob.get("class_name")
            if class_name is None:
                continue
            frame_pose = ob.get("pose") or poses[rid]
            if (frame_pose is not None and ob.get("distance") is not None
                    and ob.get("bearing") is not None):
                z = measurement_from_observation(frame_pose, ob["distance"], ob["bearing"])
            elif ob.get("position") is not None:
                z = np.asarray(ob["position"], float)
                n_fallback += 1
            else:
                continue
            per_class.setdefault(class_name, {}).setdefault(rid, []).append(z)
    return used, poses, per_class, n_fallback


def localize_live(robots, sensor, methods, zones=None):
    """For the dashboard and the live map.

    robots: the observations of every robot, format of
    measurements_by_class (distance, bearing, class + robot pose).
    Returns (used, {class_name: {method: dict(n, objects, extra)}})."""
    used, poses, per_class, _ = measurements_by_class(robots)
    results = {}
    for class_name, reports in sorted(per_class.items()):
        results[class_name] = {}
        for m in methods:
            if m == "zone_class" and not zones:
                continue
            out = run_method(m, used, reports, poses, sensor, zones=zones)
            results[class_name][m] = dict(n=out["n"], objects=out["objects"],
                                          extra=out.get("extra", {}))
    return used, results


def ospa(estimated, truth, c=0.5, p=1):
    X = [np.asarray(x, float) for x in estimated]
    Y = [np.asarray(y, float) for y in truth]
    if not X and not Y:
        return 0.0
    if len(X) > len(Y):
        X, Y = Y, X
    m, n = len(X), len(Y)
    if m == 0:
        return float(c)
    D = np.array([[min(c, float(np.hypot(*(x - y)))) ** p for y in Y] for x in X])
    if math.factorial(n) // math.factorial(n - m) <= 50000:
        best = min(sum(D[i, j] for i, j in enumerate(perm))
                   for perm in permutations(range(n), m))
    else:  # greedy, only for very crowded scenes
        best, free = 0.0, set(range(n))
        for i in range(m):
            j = min(free, key=lambda jj: D[i, jj])
            best += D[i, j]
            free.discard(j)
    return float(((best + c ** p * (n - m)) / n) ** (1.0 / p))