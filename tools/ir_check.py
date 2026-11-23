"""Live readout of the infrared (ToF) sensors of one RoboMaster EP, to find which index is the front sensor and to
check its range by hand (a box or a hand in front of one sensor at a time).

    python -m tools.ir_check --robot-id P0                 # prints the 4 distances 5 times a second
    python -m tools.ir_check --robot-id P0 --stats 10      # then 10 s of statistics per sensor (median, spread)

Index i of the printout is --ir-index i of acquisition/ir_live.py.
"""

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-id", required=True)
    ap.add_argument("--sn", default=None)
    ap.add_argument("--ip", default=None)
    ap.add_argument("--freq", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=30.0, help="duration of the live printout")
    ap.add_argument("--stats", type=float, default=0.0, help="then this many seconds of statistics")
    a = ap.parse_args()
    from robomaster_sdk import Robot
    from robomaster_sdk.ir import IRSensors

    with Robot(a.robot_id, sn=a.sn, ip=a.ip) as robot, IRSensors.from_robot(robot, freq=a.freq, history=max(200, int(a.freq * (a.stats + 5)))) as ir:
        time.sleep(0.5)
        print(f"first message from the SDK, as it arrived: {ir.first_raw}")
        print("index:      0        1        2        3   [m]  (Ctrl+C to stop)")
        t0 = time.time()
        try:
            while time.time() - t0 < a.seconds:
                s = ir.latest()
                if s is not None:
                    print("       " + "  ".join(f"{v:7.3f}" if ir.valid(v) else "    ---" for v in s[1]), flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        if a.stats > 0:
            print(f"statistics over {a.stats:.0f} s ...")
            t1 = time.time()
            time.sleep(a.stats)
            rows = [v for t, v in list(ir.samples) if t > t1]
            for i in range(4):
                v = np.array([r[i] for r in rows if i < len(r)], float)
                ok = v[(v >= ir.min_m) & (v <= ir.max_m)]
                print(f"  index {i}: {len(ok)}/{len(v)} valid" + (
                    f", median {np.median(ok):.3f} m, sd {np.std(ok) * 1000:.1f} mm, "
                    f"range {ok.min():.3f}-{ok.max():.3f} m" if len(ok) else ""))


if __name__ == "__main__":
    main()
