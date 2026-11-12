"""Capture plan of the third block of real scenes (R81 onwards): layouts chosen by their difficulty level.

The scenes of the real campaigns are split into three levels by their geometry only (tools/scene_levels.py):
  easy    every pair of boxes at least 25 cm apart, every box in clear view of at least two robots,
          no obstacles or white objects
  medium  as easy, but at least one box in clear view of a single robot
  hard    two boxes less than 25 cm apart, a box in clear view of no robot, obstacles or white objects
(a box is in clear view of a robot if it is within 40 deg of the optical axis and 3 m, and the line of sight
to it does not cross another box).

The second campaign has few medium scenes and few scenes of blocks D (white objects) and E (light). This plan
adds layouts of exactly those kinds. Every candidate layout is drawn at random (seed fixed), its robot positions
of the 4 rounds are predicted (circle: the stations of launch_all --radius turning by 22.5 deg per round;
random: the positions placed by hand turning about the boxes), and the level of every round is computed with
the boxes moved by +-4 cm and the robots by +-8 cm, as placement and driving errors. A medium layout is kept
only if at least 3 of its 4 rounds are medium in at least 60% of these draws, whichever way the formation
turns; in practice these are dense blocks of 6-7 boxes, close but not touching.

    python -m tools.capture_plan_v3                    # docs/capture_plan_v3.json, docs/capture_guide_v3.pdf,
                                                       # docs/capture_commands_v3.txt
    python -m tools.place_check --plan docs/capture_plan_v3.json --layout R81
"""

import argparse
import json
import math

import numpy as np

from tools.capture_plan import (BOX_IDS, CENTRE, KEEP_OUT_M, MIN_GAP_M, RADIUS, ROBOTS, STEP_DEG, SWEEP_DEG,
                                inside, random_robots, stations, white_objects, write_pdf)
from tools.scene_levels import scene_level

N_ROUNDS = int(round(SWEEP_DEG / STEP_DEG)) + 1
LENS_M = 0.15                       # camera ahead of the robot's rigid body


def cameras(robots, centre, turn_deg):
    """Camera poses (x, y, yaw) of the robots placed at `robots` [x, y, heading_deg] after the formation
    turned by turn_deg about `centre`, every robot facing `centre`."""
    out = []
    c = np.asarray(centre, float)
    t = math.radians(turn_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    for x, y, _ in robots:
        p = c + R @ (np.array([x, y]) - c)
        yaw = math.atan2(c[1] - p[1], c[0] - p[0])
        out.append((p[0] + LENS_M * math.cos(yaw), p[1] + LENS_M * math.sin(yaw), yaw))
    return out


def round_levels(boxes, robots, centre, rng, draws=40, box_err=0.04, robot_err=0.08):
    """For each round and turning direction: the share of draws (boxes and robots moved by uniform errors)
    in which the round is easy / medium / hard. Returns {direction: [dict(level -> share) per round]}."""
    out = {}
    for sign in (+1, -1):
        rounds = []
        for k in range(N_ROUNDS):
            count = dict(easy=0, medium=0, hard=0)
            for _ in range(draws):
                b = np.asarray(boxes, float) + rng.uniform(-box_err, box_err, (len(boxes), 2))
                r = [(x + rng.uniform(-robot_err, robot_err), y + rng.uniform(-robot_err, robot_err), h)
                     for x, y, h in robots]
                count[scene_level(b, cameras(r, centre, sign * k * STEP_DEG))[0]] += 1
            rounds.append({lv: n / draws for lv, n in count.items()})
        out[sign] = rounds
    return out


def candidate_boxes(n, kind, rng):
    """n box centres (relative to the formation centre) of one arrangement kind, at least 30 cm apart."""
    gap = 0.30
    if kind == "line":                          # a line pointing between two robots: the rear boxes hide
        a = rng.uniform(0, 2 * math.pi)
        pitch = rng.uniform(0.30, 0.38)
        t = (np.arange(n) - (n - 1) / 2) * pitch
        pts = np.stack([t * math.cos(a), t * math.sin(a)], 1) + rng.normal(0, 0.03, (n, 2))
    elif kind == "block":                       # a compact group, not touching
        pitch = rng.uniform(0.32, 0.38)
        lat = np.array([(i * pitch + j * pitch / 2, j * pitch * math.sqrt(3) / 2)
                        for i in range(-3, 4) for j in range(-3, 4)])
        order = np.argsort(np.hypot(*(lat - rng.normal(0, pitch / 3, 2)).T))
        pts = lat[order[:n]]
        pts = pts - pts.mean(0) + rng.normal(0, 0.02, (n, 2))
    else:                                       # a group plus a box behind it
        a = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        m = n - 1
        t = (np.arange(m) - (m - 1) / 2) * rng.uniform(0.32, 0.38)
        pts = np.vstack([np.outer(t, v), [u * rng.uniform(0.32, 0.40)]]) + rng.normal(0, 0.02, (n, 2))
        pts -= pts.mean(0)
    d = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
    d[np.diag_indices(n)] = 9
    if d.min() < max(gap, MIN_GAP_M) or np.max(np.hypot(*pts.T)) > 0.55:
        return None
    return pts


def dense_block(n, rng, lo=0.30, hi=0.34):
    """n boxes on a jittered triangular lattice of pitch lo-hi (centres 30-34 cm apart: close, not touching)."""
    pitch = rng.uniform(lo, hi)
    lat = np.array([(i * pitch + j * pitch / 2, j * pitch * math.sqrt(3) / 2) for i in range(-3, 4) for j in range(-3, 4)])
    order = np.argsort(np.hypot(*(lat - rng.normal(0, pitch / 3, 2)).T))
    pts = lat[order[:n]]
    return pts - pts.mean(0) + rng.normal(0, 0.015, (n, 2))


def plan_medium(rng, n_layouts, start_id, draws=20, tries=3000):
    """Medium layouts: dense blocks of 6-7 boxes (centres 30-34 cm apart), robots on a circle of 1.0 or 1.3 m
    or at random. Kept if at least 3 of the 4 rounds are medium in at least 60% of the draws, whichever way
    the formation turns (a box hidden by the others from all robots but one)."""
    specs = [(7, "cerchio 1.0"), (7, "casuale"), (7, "cerchio 1.3"), (6, "casuale"), (6, "cerchio 1.0")]
    out, t = [], 0
    while len(out) < n_layouts and t < tries:
        n, robots_kind = specs[t % len(specs)]
        t += 1
        pts = dense_block(n, rng)
        boxes = pts + np.asarray(CENTRE)
        if RADIUS[robots_kind]:
            robots, centre = stations(RADIUS[robots_kind]), CENTRE
        else:
            robots = random_robots(boxes, rng, dmin=0.9)
            if robots is None:
                continue
            centre = tuple(boxes.mean(0))
        lv = round_levels(boxes, robots, centre, rng, draws=draws)
        good = min(sum(r["medium"] >= 0.6 for r in lv[sign]) for sign in lv)
        if good < 3:
            continue
        share = float(np.mean([r["medium"] for sign in lv for r in lv[sign]]))
        out.append(dict(boxes=boxes, robots=robots, robots_kind=robots_kind, n=n, kind="block", medium_share=share,
                        rounds=good))
        print(f"  medium layout {len(out)}: {n} boxes, {robots_kind}, {good}/4 rounds medium, after {t} tries",
              flush=True)
    return out


def layout_dict(lid, block, n, arrangement, robots_kind, boxes, robots, note, objects, rng, level):
    from tools.capture_plan import command
    row = dict(id=lid, robots=robots_kind, extra=note)
    cmd = command(row, n)
    return dict(id=lid, day=6, block=block, split="test", n_boxes=n, arrangement=arrangement, robots=robots_kind,
                radius_m=RADIUS[robots_kind], note=note, centre=list(CENTRE), level=level,
                boxes=[[round(float(x), 3), round(float(y), 3), round(float(rng.uniform(0, 120)), 0)] for x, y in boxes],
                box_ids=BOX_IDS[:n], robot_targets=robots, objects=objects, command=cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="docs/capture_plan_v3.json")
    ap.add_argument("--pdf", default="docs/capture_guide_v3.pdf")
    ap.add_argument("--commands", default="docs/capture_commands_v3.txt")
    ap.add_argument("--medium", type=int, default=14)
    ap.add_argument("--white", type=int, default=5)
    ap.add_argument("--light", type=int, default=5)
    ap.add_argument("--first-id", type=int, default=81)
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    layouts, k = [], a.first_id
    names = dict(block="dense block, not touching")
    for m in plan_medium(rng, a.medium, k):
        layouts.append(layout_dict(f"R{k}", "M medium", m["n"], names[m["kind"]], m["robots_kind"], m["boxes"],
                                   m["robots"], f"medium in {m['rounds']}/4 rounds ({100 * m['medium_share']:.0f}% of the draws)",
                                   [], rng, "medium"))
        k += 1
    # white objects (block D) and light (block E): 3-5 sparse boxes, circle or random robots
    for block, count in (("D white objects", a.white), ("E light", a.light)):
        made = 0
        while made < count:
            n = (3, 4, 5)[made % 3]
            robots_kind = ("cerchio 1.3", "casuale")[made % 2]
            pts = []
            for _ in range(4000):
                r, ang = 0.5 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
                p = np.array([r * math.cos(ang), r * math.sin(ang)])
                if all(np.hypot(*(p - q)) >= 0.40 for q in pts):
                    pts.append(p)
                if len(pts) == n:
                    break
            if len(pts) < n:
                continue
            pts = np.array(pts)
            boxes = pts + np.asarray(CENTRE)
            robots = stations(1.3) if RADIUS[robots_kind] else random_robots(boxes, rng, dmin=0.9)
            if robots is None:
                continue
            if block.startswith("D"):
                objs = white_objects(2 if made % 2 else 1, pts, rng, 0.6)
                note = f"{len(objs)} white object(s) that are not boxes (paper bin, white bag, white cup...)"
            else:
                objs = []
                note = ("curtains open" if made % 2 == 0 else "half of the lights off")
            centre = CENTRE if RADIUS[robots_kind] else tuple(boxes.mean(0))
            lv = round_levels(boxes, robots, centre, rng, draws=20)
            lvl = "hard" if block.startswith("D") else max(("easy", "medium", "hard"),
                                                             key=lambda L: sum(r[L] for r in lv[1]))
            layouts.append(layout_dict(f"R{k}", block, n, "sparse", robots_kind, boxes, robots, note, objs, rng, lvl))
            k += 1
            made += 1
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(layouts, f, indent=1)
    write_pdf(layouts, a.pdf)
    with open(a.commands, "w", encoding="utf-8") as f:
        f.write("CAPTURE COMMANDS - third block of real scenes (R%d-R%d)\n" % (a.first_id, k - 1))
        f.write("Run from the project folder on the lab PC, venv active. Capture in this order (medium first).\n")
        f.write("Medium layouts: dense blocks whose boxes must NOT touch (centres 28-34 cm apart): place them\n"
                "within 3 cm (--tolerance 0.03). White objects of block D are not tracked: place them by eye\n"
                "from the guide. Block E: change the light before starting the layout and keep it for all 4 rounds.\n"
                "End of the day: git add data/survey_v2 config/ground_truth_positions.json, commit, push.\n\n")
        for L in layouts:
            f.write(f"{L['id']}  {L['block']}  {L['n_boxes']} boxes, {L['arrangement']}, robots {L['robots']}"
                    f"  [{L['note']}]\n")
            tol = " --tolerance 0.03" if L["level"] == "medium" else ""
            f.write(f"    python -m tools.place_check --plan {a.json} --layout {L['id']}{tol}\n")
            f.write(f"    (it runs: {L['command']})\n\n")
    print(f"{len(layouts)} layouts -> {a.json}, {a.pdf}, {a.commands}")
    for L in layouts:
        print(f"  {L['id']} {L['block']:<16} {L['n_boxes']} boxes {L['arrangement']:<22} {L['robots']:<12} "
              f"{L['level']:<7} {L['note']}")


if __name__ == "__main__":
    main()
