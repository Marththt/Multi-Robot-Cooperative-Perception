"""Which way the gimbal of one RoboMaster EP turns for a positive SDK yaw and pitch, and what it reads back.

DJI does not document the signs. This moves the gimbal (the chassis does not move: SDK "free" mode) to
SDK yaw +20 deg, back, then SDK pitch +10 deg, back, pausing on each so you can look; it prints the angles the
gimbal reports. Then set robomaster_sdk/gimbal.py YAW_SIGN / PITCH_SIGN (or pass --gimbal-yaw-sign /
--gimbal-pitch-sign to acquisition/ir_live.py):

  SDK yaw +20 turned the gimbal to the LEFT  -> YAW_SIGN = +1      to the RIGHT -> YAW_SIGN = -1
  SDK pitch +10 tilted it DOWN               -> PITCH_SIGN = +1    UP           -> PITCH_SIGN = -1

    python -m tools.gimbal_check --robot-id P0
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-id", required=True)
    ap.add_argument("--sn", default=None)
    ap.add_argument("--ip", default=None)
    ap.add_argument("--pause-s", type=float, default=3.0)
    a = ap.parse_args()
    from robomaster_sdk import Robot
    from robomaster_sdk.gimbal import Gimbal

    with Robot(a.robot_id, sn=a.sn, ip=a.ip) as robot:
        g = Gimbal.from_robot(robot).start()
        try:
            print(f"first message from the SDK, as it arrived (pitch, yaw, pitch_ground, yaw_ground): {g.first_raw}")
            g.recenter()
            time.sleep(1.0)

            def show(what):
                r = g.raw()
                print(f"{what:<28} reads pitch {r[1]:+6.1f}  yaw {r[2]:+6.1f}  (ground: pitch {r[3]:+6.1f}  "
                      f"yaw {r[4]:+6.1f}) [SDK deg]" if r else f"{what}: no angle reading", flush=True)

            show("centred")
            for label, pitch, yaw in [("SDK yaw +20: LEFT or RIGHT?", 0, 20), ("back to centre", 0, 0),
                                      ("SDK pitch +10: UP or DOWN?", 10, 0), ("back to centre", 0, 0)]:
                g.g.moveto(pitch=pitch, yaw=yaw, pitch_speed=60, yaw_speed=60).wait_for_completed(5)
                time.sleep(a.pause_s)
                show(label)
        finally:
            g.close()


if __name__ == "__main__":
    main()
