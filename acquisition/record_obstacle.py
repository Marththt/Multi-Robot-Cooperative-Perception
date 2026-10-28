"""Records where the OCCLUDERS are (OptiTrack rigid bodies on the cartons), once per run.

The occluders are NOT boxes of the layout: launch_all.py only reads the rigid bodies of the boxes
(--box / rb_id in the ground truth), so the occluders never enter the ground truth and are never
counted as targets. This script saves their positions apart, in <run>/obstacles.json, to know exactly
where they stood (to explain the results, and to reproduce them in the simulator).

Run it right after launch_all.py has started the run (or just before it, with the cartons already in
place and not moved afterwards), with the run folder as --out:

    python record_obstacles.py --rb 20 21 22 --out survey_rounds/L12_20261026-153000/ --size 0.30 0.30 0.30

--size: side of each carton [m], same order as --rb (used by the analysis as the radius of the blocked
region; default 0.30).
"""
import argparse
import json
import os
import time

from vf.localization.mocap_pose import MocapHub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rb", type=int, nargs="+", required=True, help="OptiTrack rigid body ids of the occluders")
    ap.add_argument("--size", type=float, nargs="*", default=None, help="side of each carton [m]")
    ap.add_argument("--out", required=True, help="run folder (obstacles.json is written inside)")
    ap.add_argument("--seconds", type=float, default=2.0, help="averaging time")
    ap.add_argument("--optitrack-server", default="127.0.0.1")
    ap.add_argument("--optitrack-local", default="127.0.0.1")
    a = ap.parse_args()
    sizes = a.size or [0.30] * len(a.rb)
    if len(sizes) != len(a.rb):
        raise SystemExit("--size needs one value per --rb")

    hub = MocapHub(a.rb, a.optitrack_server, a.optitrack_local)
    samples = {rb: [] for rb in a.rb}
    try:
        try:
            hub.wait_ready()
        except RuntimeError as e:
            raise SystemExit(f"OptiTrack: {e}")
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            for rb in a.rb:
                q = hub.get_pose(rb)
                if q is not None:
                    samples[rb].append((q[0], q[1]))
            time.sleep(0.02)
    finally:
        hub.close()

    obstacles = []
    for rb, size in zip(a.rb, sizes):
        s = samples[rb]
        if not s:
            raise SystemExit(f"no OptiTrack reading for rigid body {rb}: check it in Motive")
        x = sum(p[0] for p in s) / len(s)
        y = sum(p[1] for p in s) / len(s)
        obstacles.append(dict(rb_id=rb, x=round(x, 4), y=round(y, 4), size_m=size))
        print(f"obstacle rb {rb}: x={x:+.3f} y={y:+.3f} (side {size:g} m)")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "obstacles.json")
    with open(path, "w") as f:
        json.dump(obstacles, f, indent=2)
    print(f"written {path}")


if __name__ == "__main__":
    main()