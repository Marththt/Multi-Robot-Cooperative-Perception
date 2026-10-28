"""Calibrates the frame correction (rotation + translation) between the raw
pose of an OptiTrack robot and the true GLOBAL reference frame.

Why it is needed
-------------
The raw pose of an OptiTrack rigid body (x, y, yaw) may not be registered
in the same "true" frame used to define where the objects really are in the
room -- this happens when the rigid body was created/zeroed in Motive
without an explicit alignment to the room axes. [NOTE September 2026: the
bearing math was NOT correct -- global_position used yaw + bearing instead
of yaw - bearing, now fixed; corrections fitted before that fix must be
redone.] Neither the camera calibration (checkerboard) nor the plausibility
of the single robot's pose (all already verified) touch this problem: a
rigid body can report a pose that is perfectly self-consistent and still be
rotated/translated with respect to the true frame. The effect: every object
detected by that robot (object_position.global_position(), which uses the
robot's x,y,yaw directly) inherits a CONSTANT rotation+translation error for
that robot, which shows up as disagreement between the positions computed
by different robots for the same physical object -- confirmed in
conversation by comparing the positions with an independent ground-truth
reference (see below): up to 1.8m of error for a single robot on a single
object.

How the ground-truth reference is measured
------------------------------------------
You need an OptiTrack rigid body ATTACHED to an object (e.g. the box),
whose TRUE position (x,y) you read directly from OptiTrack --
mocap_pose.MocapPose(object_rb_id, server, local).get_pose() -- WITHOUT
going through the camera/bearing/distance of any robot. For N different
positions of the object: note (a) its true position from this, (b) the
position that ONE robot computes for the same object (xy= shown by
static_trial.py --live). With AT LEAST 2 (true, computed) pairs the
rotation+translation is solved exactly; with 3+ a least-squares fit is done
(more points = more robust to the noise of single measurements, and it lets
you check the residual). The calibration is PER ROBOT: P0 and P1 must be
calibrated separately (different frames = different corrections), see
--points/--out to do one pass at a time.

The fit is a 2D rigid alignment (Procrustes/Kabsch without scale): it finds
the rotation theta and translation (dx,dy) that minimize
sum |R(theta)*computed_i + (dx,dy) - true_i|^2, with the closed form for
the 2D case:
    theta = atan2(sum(Sx'*Ty' - Sy'*Tx'), sum(Sx'*Tx' + Sy'*Ty'))
where S' = computed - centroid(computed), T' = true - centroid(true),
followed by the translation that aligns the two centroids after the
rotation.

Usage
---
    python -m tools.calibrate_optitrack --points points_P1.json --out correction_P1.json

points_P1.json:
    {"points": [{"true": [0.022, -0.066], "computed": [-0.33, -0.11]},
                {"true": [-0.409, 0.852], "computed": [-0.16, 1.00]},
                {"true": [0.752, -0.537], "computed": [-0.93, -0.82]}]}

correction_P1.json (the output) is then passed to static_trial.py with
--frame-correction-file correction_P1.json (see FrameCorrectedPose in
vf/localization/localization.py for how it is applied)."""

import argparse
import json
import math


def fit_rigid_transform(points):
    """points: list of dicts {"true": [x,y], "computed": [x,y]}.
    Returns (rotation_rad, dx, dy, residuals) where residuals is the list
    of point-by-point distances between corrected-computed and true, after
    applying the transformation found (lower = better fit; a high residual
    on all points means the error is not explained by a single constant
    rotation+translation)."""
    if len(points) < 2:
        raise ValueError(
            "at least 2 (true, computed) pairs are needed to solve "
            "rotation+translation; with 3+ the fit is least-squares "
            "and more robust to the noise of single measurements")

    true_pts = [tuple(p["true"]) for p in points]
    computed_pts = [tuple(p["computed"]) for p in points]
    n = len(points)

    cx_v = sum(v[0] for v in true_pts) / n
    cy_v = sum(v[1] for v in true_pts) / n
    cx_c = sum(c[0] for c in computed_pts) / n
    cy_c = sum(c[1] for c in computed_pts) / n

    sum_cross = 0.0
    sum_dot = 0.0
    for (vx, vy), (cx, cy) in zip(true_pts, computed_pts):
        sx, sy = cx - cx_c, cy - cy_c
        tx, ty = vx - cx_v, vy - cy_v
        sum_cross += sx * ty - sy * tx
        sum_dot += sx * tx + sy * ty

    theta = math.atan2(sum_cross, sum_dot)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # required: true_centroid = R(theta)*computed_centroid + (dx,dy)
    rcx = cos_t * cx_c - sin_t * cy_c
    rcy = sin_t * cx_c + cos_t * cy_c
    dx = cx_v - rcx
    dy = cy_v - rcy

    residuals = []
    for (vx, vy), (cx, cy) in zip(true_pts, computed_pts):
        px = cos_t * cx - sin_t * cy + dx
        py = sin_t * cx + cos_t * cy + dy
        residuals.append(math.hypot(px - vx, py - vy))

    return theta, dx, dy, residuals


def main():
    p = argparse.ArgumentParser(
        description="Calibrates rotation+translation between the pose of ONE "
                    "OptiTrack robot and the true frame, from pairs "
                    "(true position, position computed by the robot) "
                    "for the same independently tracked object.")
    p.add_argument("--points", required=True,
                   help="JSON file with {\"points\": [{\"true\": [x,y], "
                        "\"computed\": [x,y]}, ...]} -- see the "
                        "module docstring")
    p.add_argument("--out", default=None,
                   help="if given, writes the result here "
                        "(rotation_rad, dx_m, dy_m) as JSON, ready "
                        "for --frame-correction-file of static_trial.py")
    args = p.parse_args()

    with open(args.points) as f:
        data = json.load(f)
    points = data["points"]

    theta, dx, dy, residuals = fit_rigid_transform(points)

    print(f"rotation: {math.degrees(theta):+.2f} degrees "
          f"({theta:+.4f} rad)")
    print(f"translation: dx={dx:+.3f}m dy={dy:+.3f}m")
    print(f"\npoint-by-point residuals after the correction ({len(points)} "
          f"points):")
    for i, r in enumerate(residuals):
        print(f"  point {i + 1}: {r:.3f}m")
    print(f"mean residual: {sum(residuals) / len(residuals):.3f}m  "
          f"max: {max(residuals):.3f}m")
    if max(residuals) > 0.3:
        print("\nmax residual above 0.3m: rotation+translation alone "
              "does not explain all the disagreement -- a common error "
              "remains to be checked separately (e.g. configured vs real "
              "marker size, offset between the ground-truth tracker and "
              "the physical point measured on the object).")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rotation_rad": theta, "dx_m": dx, "dy_m": dy},
                      f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
