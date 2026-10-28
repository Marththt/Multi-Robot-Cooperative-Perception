"""Continuously prints x, y, yaw (degrees) of ONE OptiTrack rigid body --
used for the alignment check: place the two robots physically PARALLEL
(same heading, use a ruler or a wall as reference), run this on one robot
at a time with its --rb-id, and compare the printed yaw. If for an
identical physical alignment the two robots give a different yaw, that
difference is the rigid body calibration offset to correct before
trusting any distance/position computed by patrol.py/static_trial.py (see
object_position.global_position, which adds bearing + this yaw).

Usage
---
    python -m tools.read_yaw --rb-id 8
    python -m tools.read_yaw --rb-id 7 --server 127.0.0.1 --local 127.0.0.1

'q' + enter to quit, or Ctrl+C. Only one robot/rigid body at a time is
intentional: it is a check, there is no need to read them together.
"""

import argparse
import math
import sys
import threading
import time
from vf.localization.mocap_pose import MocapPose


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rb-id", type=int, required=True,
                   help="rigid body id in Motive for THIS robot")
    p.add_argument("--server", default="127.0.0.1")
    p.add_argument("--local", default="127.0.0.1")
    p.add_argument("--hz", type=float, default=2.0,
                   help="how many readings per second to print (default 2)")
    args = p.parse_args()
    source = MocapPose(args.rb_id, args.server, args.local)
    source.wait_ready()

    stop = threading.Event()

    def wait_for_exit():
        try:
            input()
        except EOFError:
            pass
        stop.set()

    print("'q' + enter (or Ctrl+C) to quit. Printing x, y, yaw until "
          "you stop.\n")
    threading.Thread(target=wait_for_exit, daemon=True).start()

    try:
        while not stop.is_set():
            pose = source.get_pose()
            if pose is None:
                print("\rno pose available (tracking lost?)"
                      "                    ", end="", flush=True)
            else:
                x, y, yaw_rad = pose
                yaw_deg = math.degrees(yaw_rad)
                print(f"\rrb_id={args.rb_id}  x={x:+.3f}  y={y:+.3f}  "
                      f"yaw={yaw_deg:+7.2f} degrees ({yaw_rad:+.4f} rad)"
                      f"          ", end="", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        source.close()


if __name__ == "__main__":
    main()
