"""Records the TRUE positions of the boxes of one layout from OptiTrack and
writes them into the ground truth file read by survey.py, build_scans.py
and localize_trials.py (--ground-truth).

Every box needs its own OptiTrack rigid body (a few reflective markers on
the box, defined as a rigid body in Motive; note its Streaming ID). Its
position is read directly from OptiTrack, without going through any robot
camera.

Place the boxes, record them, then run the survey (the boxes must not move
until the survey of that layout is over):

    python -m tools.record_ground_truth --layout-id L1 \\
        --object 20:box --object 21:box --object 22:box

--object is STREAMING_ID:CLASS, one per box in the scene. The position of
each box is averaged over --seconds (default 2 s); the spread of the
readings is printed as a check (it should be a few millimetres: more means
the rigid body is poorly tracked).

The file (default config/ground_truth_positions.json) is updated, not
rewritten: the other layouts already in it are kept. A layout already in
the file is only replaced with --overwrite. The key can be a survey layout
(--layout-id) or a static_trial.py trial (--trial-id, same thing): both
are "one fixed scene". An empty scene:

    python -m tools.record_ground_truth --layout-id L0 --empty

Output format (rb_id is extra information, ignored by the readers):
    {"L1": [{"class_name": "box", "x": 0.52, "y": -1.10, "rb_id": 20},
            ...],
     "L0": []}
"""

import argparse
import json
import math
import os
import sys
import time


def parse_objects(values):
    """["20:box", "21:box"] -> [(20, "box"), (21, "box")]."""
    objects = []
    for v in values:
        if ":" not in v:
            raise ValueError(f"--object {v!r}: expected STREAMING_ID:CLASS, e.g. 20:box")
        rb, class_name = v.split(":", 1)
        objects.append((int(rb), class_name.strip()))
    ids = [rb for rb, _ in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("the same Streaming ID is given twice")
    return objects


def average_positions(samples):
    """{rb_id: [(x, y), ...]} -> {rb_id: (mean_x, mean_y, spread_m)},
    spread = RMS distance of the readings from their mean."""
    out = {}
    for rb, pts in samples.items():
        if not pts:
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        spread = math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in pts) / len(pts))
        out[rb] = (mx, my, spread)
    return out


def update_file(path, key, entries, overwrite):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    if key in data and not overwrite:
        raise SystemExit(f"{key} is already in {path}: use --overwrite to replace it")
    data[key] = entries
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(sorted(data.items())), f, indent=2)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    key = p.add_mutually_exclusive_group(required=True)
    key.add_argument("--layout-id", dest="key", metavar="ID",
                     help="same --layout-id used with survey.py")
    key.add_argument("--trial-id", dest="key", metavar="ID",
                     help="same --trial-id used with static_trial.py")
    p.add_argument("--object", action="append", default=[],
                   help="STREAMING_ID:CLASS of one object, repeat for every object")
    p.add_argument("--empty", action="store_true", help="scene with no objects")
    p.add_argument("--seconds", type=float, default=2.0,
                   help="averaging time (default 2 s)")
    p.add_argument("--out", default="config/ground_truth_positions.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--server", default="127.0.0.1")
    p.add_argument("--local", default="127.0.0.1")
    a = p.parse_args()

    if a.empty == bool(a.object):
        p.error("give either one or more --object, or --empty")
    if a.empty:
        update_file(a.out, a.key, [], a.overwrite)
        print(f"{a.key}: no objects, written to {a.out}")
        return

    try:
        objects = parse_objects(a.object)
    except ValueError as e:
        p.error(str(e))

    from vf.localization.mocap_pose import MocapHub  # needs natnet
    hub = MocapHub([rb for rb, _ in objects], a.server, a.local)
    try:
        hub.wait_ready()
        samples = {rb: [] for rb, _ in objects}
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            for rb, _ in objects:
                pose = hub.get_pose(rb)
                if pose is not None:
                    samples[rb].append((pose[0], pose[1]))
            time.sleep(0.02)
    finally:
        hub.close()

    averaged = average_positions(samples)
    missing = [rb for rb, _ in objects if rb not in averaged]
    if missing:
        sys.exit(f"no valid reading for rigid bodies {missing}: nothing written "
                 f"(check that they are tracked in Motive)")

    entries = []
    for rb, class_name in objects:
        x, y, spread = averaged[rb]
        n = len(samples[rb])
        print(f"  rb {rb:>3} {class_name:<8} x={x:+.3f} y={y:+.3f}  "
              f"({n} readings, spread {1000 * spread:.1f} mm)")
        entries.append(dict(class_name=class_name, x=round(x, 4), y=round(y, 4), rb_id=rb))
    update_file(a.out, a.key, entries, a.overwrite)
    print(f"{a.key}: {len(entries)} object(s) written to {a.out}")


if __name__ == "__main__":
    main()