"""Checks, on ONE robot, everything the autonomous motion (survey.py,
vf/control/goto.py) relies on, before letting it drive by itself:

  1. the OptiTrack rigid body (--optitrack-rb-id) is really this robot;
  2. the yaw read from OptiTrack points where the robot's front points
     (otherwise every command goes in a rotated direction);
  3. the command signs of goto.py (SIGN_X, SIGN_Y, SIGN_Z).

The robot makes six short moves at low speed, each followed by the
opposite one, so it ends up roughly where it started: forward/back,
left/right, turn left/turn right (0.15 m/s or 30 deg/s for 1.5 s, about
0.2 m and 45 degrees). Leave at least 0.5 m of free space around it.

For each move the displacement measured by OptiTrack is compared with the
one expected from the yaw it reports. At the end it prints what is wrong
(if anything) and the value to pass as --yaw-offset-deg to survey.py.

    python -m tools.check_motion --robot-id P0 --optitrack-rb-id 8
"""

import argparse
import math
import time

from vf.control.goto import SIGN_X, SIGN_Y, SIGN_Z, stop, wrap_pi

SPEED_M_S = 0.15
TURN_DEG_S = 30.0
MOVE_S = 1.5


def mean_pose(pose_fn, seconds=0.5):
    """Average pose over a short time (yaw averaged on the circle)."""
    xs, ys, cs, ss = [], [], [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        p = pose_fn()
        if p is not None:
            xs.append(p[0])
            ys.append(p[1])
            cs.append(math.cos(p[2]))
            ss.append(math.sin(p[2]))
        time.sleep(0.02)
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys), math.atan2(sum(ss), sum(cs)))


def drive(chassis, vx=0.0, vy=0.0, wz_deg=0.0, seconds=MOVE_S):
    """Same command convention as goto.py: vx forward, vy left [m/s],
    wz counter-clockwise [deg/s], then the SIGN constants."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chassis.drive_speed(x=SIGN_X * vx, y=SIGN_Y * vy, z=SIGN_Z * wz_deg, timeout=0.3)
        time.sleep(0.05)
    stop(chassis)
    time.sleep(0.6)


def verdict(results, rb_id):
    """results: forward=(dist, err_deg), left=(dist, err_deg), turn=dyaw_deg
    (any may be missing) -> list of findings."""
    out = []
    fwd, left, turn = results.get("forward"), results.get("left"), results.get("turn")
    if fwd is not None and fwd[0] < 0.08:
        return [f"rigid body {rb_id} barely moved while the robot was driving: it is NOT this robot "
                "(wrong --optitrack-rb-id), or the robot did not move"]
    if turn is not None:
        if abs(turn) < 15:
            out.append("the yaw barely changes while the robot turns: wrong rigid body or wrong UP_AXIS "
                       "in vf/localization/mocap_pose.py")
        elif turn < 0:
            out.append("the yaw DECREASES when the robot turns left: flip SIGN_Z in vf/control/goto.py "
                       "(if the robot physically turned left, the planar frame is mirrored: check UP_AXIS)")
    if fwd is not None and left is not None:
        ef, el = fwd[1], left[1]
        rel = math.degrees(wrap_pi(math.radians(el - ef)))
        if abs(abs(rel) - 180) < 40:
            out.append("LEFT and FORWARD are not 90 degrees apart (left goes right): flip SIGN_Y in "
                       "vf/control/goto.py, or the planar frame is mirrored (UP_AXIS)")
        else:
            off = math.degrees(math.atan2(math.sin(math.radians(ef)) + math.sin(math.radians(el)),
                                          math.cos(math.radians(ef)) + math.cos(math.radians(el))))
            if abs(off) > 10:
                out.append(f"the robot's front is {off:+.0f} deg from the yaw OptiTrack reports: "
                           f"run survey.py with --yaw-offset-deg {off:.0f}")
            else:
                out.append(f"yaw and robot front agree (within {abs(off):.0f} deg)")
    if not out or (len(out) == 1 and out[0].startswith("yaw and robot front agree")):
        out.append("everything consistent: the robot can drive by itself")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-id", required=True)
    p.add_argument("--sn", default=None)
    p.add_argument("--ip", default=None)
    p.add_argument("--optitrack-rb-id", type=int, required=True)
    p.add_argument("--optitrack-server", default="127.0.0.1")
    p.add_argument("--optitrack-local", default="127.0.0.1")
    a = p.parse_args()

    from vf.localization.mocap_pose import MocapPose
    from robomaster_sdk import Robot

    mocap = MocapPose(a.optitrack_rb_id, a.optitrack_server, a.optitrack_local)
    mocap.wait_ready()
    pose_fn = mocap.get_pose

    print("1) pose at rest (2 s)")
    still = []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        q = pose_fn()
        if q is not None:
            still.append(q)
        time.sleep(0.02)
    if not still:
        raise SystemExit("no pose from OptiTrack for this rigid body")
    x0, y0, yaw0 = mean_pose(pose_fn)
    jitter = max(math.hypot(q[0] - x0, q[1] - y0) for q in still)
    print(f"   x={x0:+.3f} y={y0:+.3f} yaw={math.degrees(yaw0):+.1f} deg, "
          f"{len(still)} poses, jitter {1000 * jitter:.1f} mm")

    robot = Robot(a.robot_id, sn=a.sn, ip=a.ip)
    robot.connect()
    chassis = robot.chassis
    results = {}
    try:
        moves = [("forward", dict(vx=SPEED_M_S), 0.0),
                 ("left", dict(vy=SPEED_M_S), 90.0)]
        for name, cmd, expected_rel in moves:
            input(f"\n2) move {name.upper()} then back: Enter to start (Ctrl+C to abort) ")
            before = mean_pose(pose_fn)
            drive(chassis, **cmd)
            after = mean_pose(pose_fn)
            drive(chassis, **{k: -v for k, v in cmd.items()})
            if before is None or after is None:
                print("   pose lost during the move")
                continue
            dx, dy = after[0] - before[0], after[1] - before[1]
            dist = math.hypot(dx, dy)
            measured = math.atan2(dy, dx)
            expected = before[2] + math.radians(expected_rel)
            err = math.degrees(wrap_pi(measured - expected))
            results[name] = (dist, err)
            print(f"   moved {dist:.3f} m towards {math.degrees(measured):+.1f} deg; "
                  f"expected towards {math.degrees(expected):+.1f} deg (yaw {math.degrees(before[2]):+.1f}"
                  f"{' + 90' if expected_rel else ''}) -> difference {err:+.1f} deg")

        input("\n3) TURN LEFT (counter-clockwise seen from above) then back: Enter to start ")
        before = mean_pose(pose_fn)
        drive(chassis, wz_deg=TURN_DEG_S)
        after = mean_pose(pose_fn)
        drive(chassis, wz_deg=-TURN_DEG_S)
        if before is not None and after is not None:
            dyaw = math.degrees(wrap_pi(after[2] - before[2]))
            results["turn"] = dyaw
            print(f"   yaw changed by {dyaw:+.1f} deg (expected about +{TURN_DEG_S * MOVE_S:.0f})")
    except KeyboardInterrupt:
        print("aborted")
    finally:
        stop(chassis)
        robot.disconnect()
        mocap.close()

    print("\n=== result ===")
    for line in verdict(results, a.optitrack_rb_id):
        print("- " + line)


if __name__ == "__main__":
    main()