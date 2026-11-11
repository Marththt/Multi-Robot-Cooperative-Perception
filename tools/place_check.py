"""Live placement check for one layout of the capture plan (docs/capture_plan_v2.json), then the capture.

Reads the boxes and the robots from OptiTrack twice a second and prints, for every target of the layout,
how far the nearest box (and, in the random family, the nearest robot) is and in which direction to move
it. Boxes are matched to targets by the smallest total distance, so any box can go on any target. A box of
the set (rigid bodies 11-17) that is inside the arena but not part of the layout is flagged: take it out.

When everything is within --tolerance, Enter starts the capture: the launch_all command of the layout (the
OptiTrack connection of this script is closed first). Ctrl+C quits without capturing.

    python -m tools.place_check --layout R01
    python -m tools.place_check --layout R01 --no-launch      # only the check
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from itertools import permutations

ALL_BOX_IDS = [11, 12, 13, 14, 15, 16, 17]
ROBOT_IDS = {"P0": 8, "P1": 7, "P3": 10, "P5": 6}
ARENA = (-2.690, 2.135, -2.672, 1.519)


def match(targets, found):
    """Assignment of found points to targets with the smallest total distance:
    list of (target index, found index or None, distance or None)."""
    idx = list(range(len(found)))
    best = None
    k = min(len(targets), len(found))
    for perm in permutations(idx, k):
        cost = sum(math.hypot(targets[i][0] - found[j][0], targets[i][1] - found[j][1]) for i, j in enumerate(perm))
        if best is None or cost < best[0]:
            best = (cost, perm)
    out = []
    for i in range(len(targets)):
        if best and i < len(best[1]):
            j = best[1][i]
            out.append((i, j, math.hypot(targets[i][0] - found[j][0], targets[i][1] - found[j][1])))
        else:
            out.append((i, None, None))
    return out


def direction(dx, dy):
    """Move hint in Motive axes, cm."""
    return f"move x {100 * dx:+5.0f} cm, y {100 * dy:+5.0f} cm"


def key_pressed():
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch() in ("\r", "\n")
        return False
    import select
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline()
        return True
    return False


def report(L, hub, tol):
    """Lines to print and whether every target is within tol."""
    lines, ok_all = [], True
    targets = [b[:2] for b in L["boxes"]]
    ids = list(L["box_ids"])
    found, found_ids = [], []
    for rb in ids:
        q = hub.get_pose(rb)
        if q is not None:
            found.append(q[:2])
            found_ids.append(rb)
    lines.append(f"{L['id']}: {L['n_boxes']} boxes, {L['arrangement']}, robots {L['robots']}   (tolerance "
                 f"{100 * tol:.0f} cm)")
    lines.append("")
    lines.append("BOXES")
    missing = [rb for rb in ids if rb not in found_ids]
    if missing:
        lines.append(f"  not tracked: rigid bodies {missing} (in view of OptiTrack?)")
        ok_all = False
    for i, j, d in match(targets, found):
        t = targets[i]
        if j is None:
            lines.append(f"  box {i + 1}: no box for it")
            ok_all = False
            continue
        good = d <= tol
        ok_all &= good
        f = found[j]
        lines.append(f"  box {i + 1} (rb {found_ids[j]}): {100 * d:5.1f} cm  "
                     + ("OK" if good else direction(t[0] - f[0], t[1] - f[1])))
    for rb in ALL_BOX_IDS:
        if rb in ids:
            continue
        q = hub.get_pose(rb)
        if q is not None and ARENA[0] <= q[0] <= ARENA[1] and ARENA[2] <= q[1] <= ARENA[3]:
            lines.append(f"  rb {rb} is in the arena but not in this layout: take it out")
            ok_all = False
    lines.append("")
    if L["radius_m"] is None:
        lines.append("ROBOTS (round 0, place by hand, facing the boxes)")
        rt = [r[:2] for r in L["robot_targets"]]
        rf, rnames = [], []
        for name, rb in ROBOT_IDS.items():
            q = hub.get_pose(rb)
            if q is not None:
                rf.append(q[:2])
                rnames.append(name)
        for i, j, d in match(rt, rf):
            if j is None:
                lines.append(f"  R{i + 1}: no robot tracked for it")
                ok_all = False
                continue
            good = d <= 2 * tol
            ok_all &= good
            lines.append(f"  R{i + 1} ({rnames[j]}): {100 * d:5.1f} cm  "
                         + ("OK" if good else direction(rt[i][0] - rf[j][0], rt[i][1] - rf[j][1])))
    else:
        lines.append(f"ROBOTS: anywhere near the circle of {L['radius_m']:g} m (they drive to the stations)")
    return lines, ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True, help="layout id of the plan, e.g. R01")
    ap.add_argument("--plan", default="docs/capture_plan_v2.json")
    ap.add_argument("--tolerance", type=float, default=0.05, help="m, boxes (robots: twice this)")
    ap.add_argument("--optitrack-server", default="127.0.0.1")
    ap.add_argument("--optitrack-local", default="127.0.0.1")
    ap.add_argument("--no-launch", action="store_true", help="only the check, never start the capture")
    a = ap.parse_args()
    with open(a.plan, encoding="utf-8") as f:
        plan = {L["id"]: L for L in json.load(f)}
    if a.layout not in plan:
        raise SystemExit(f"{a.layout} is not in {a.plan}")
    L = plan[a.layout]
    from vf.localization.mocap_pose import MocapHub
    hub = MocapHub(sorted(set(ALL_BOX_IDS) | set(ROBOT_IDS.values())), a.optitrack_server, a.optitrack_local)
    start = False
    try:
        while True:
            lines, ok = report(L, hub, a.tolerance)
            os.system("cls" if os.name == "nt" else "clear")
            print("\n".join(lines))
            print()
            if a.no_launch:
                print("Ctrl+C to quit")
            elif ok:
                print("ALL PLACED. Enter: start the capture   |   Ctrl+C: quit")
            else:
                print("Enter: start the capture anyway   |   Ctrl+C: quit")
            if not a.no_launch and key_pressed():
                start = True
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nquit, nothing captured")
    finally:
        hub.close()
    if start:
        cmd = shlex.split(L["command"])
        cmd[0] = sys.executable
        print("\n" + " ".join(cmd) + "\n", flush=True)
        time.sleep(1.0)                                 # let the NatNet port go before launch_all opens it
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
