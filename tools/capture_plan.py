"""Capture plan of the second real data set: 80 layouts (R01-R80), each with the box positions, the box
orientations, the robot positions (random family), the obstacles and the launch command, drawn one per page.

The layouts follow docs/capture_plan_v2.csv (blocks, days, test/dev split). Every layout is drawn here once
with a fixed seed, so the plan is reproducible:
  docs/capture_plan_v2.json   the layouts (Motive coordinates, metres; yaw in degrees)
  docs/capture_guide_v2.pdf   one page per layout, in capture order, plus a cover page

The boxes are placed around the arena centre CENTRE (the middle of the OptiTrack bounding box), so that
every robot position, on the circle or at random, stays --margin inside the arena. tools/place_check.py
reads the same JSON and shows live, from OptiTrack, how far each box and robot is from its target.

    python -m tools.capture_plan                      # writes both files
    python -m tools.capture_plan --pdf-only           # redraw the PDF from the JSON
"""

import argparse
import csv
import json
import math
import os

import numpy as np

ARENA = (-2.690, 2.135, -2.672, 1.519)              # x_min, x_max, y_min, y_max (config/optitrack_regions.json)
CORNERS = [(1.918, 1.452), (2.135, -2.672), (-2.690, -2.628), (-2.213, 1.519)]
CENTRE = (round((ARENA[0] + ARENA[1]) / 2, 2), round((ARENA[2] + ARENA[3]) / 2, 2))
MARGIN_M = 0.5                                      # survey --margin
KEEP_OUT_M = 0.45                                   # survey --keep-out-m (robot to box)
BOX_SIDE_M = 0.235
BOX_R_M = BOX_SIDE_M / math.sqrt(3)                 # centre to corner of the triangle
MIN_GAP_M = 2 * BOX_R_M + 0.02                      # centre distance at which two boxes cannot touch
STEP_DEG, SWEEP_DEG = 22.5, 67.5
ROBOTS = [("P0", 8, None), ("P1", 7, None), ("P3", 10, "config/checkerboard_calib_P1.json"),
          ("P5", 6, "config/checkerboard_calib_P1.json")]
BOX_IDS = [11, 12, 13, 14, 15, 16, 17]
RADIUS = {"cerchio 1.0": 1.0, "cerchio 1.3": 1.3, "cerchio 1.5": 1.5, "casuale": None}


def box_limit(robots):
    """Largest distance of a box centre from CENTRE that keeps the robots clear of it."""
    r = RADIUS[robots]
    return (r if r else 1.2) - KEEP_OUT_M - BOX_R_M


def spaced(pts, p, gap):
    return all(math.hypot(p[0] - q[0], p[1] - q[1]) >= gap for q in pts)


def cluster(n, rng, pitch=0.32, centre=(0.0, 0.0)):
    """n boxes packed on a jittered triangular lattice (pitch = centre distance), around centre."""
    lat = [(i * pitch + j * pitch / 2, j * pitch * math.sqrt(3) / 2) for i in range(-4, 5) for j in range(-4, 5)]
    lat = np.array(lat)
    seed = rng.normal(0, pitch / 3, 2)
    order = np.argsort(np.hypot(*(lat - seed).T))
    pts = lat[order[:n]]
    pts = pts - pts.mean(0)
    a = rng.uniform(0, 2 * math.pi)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = pts @ R.T + rng.uniform(-0.015, 0.015, pts.shape)
    return pts + np.asarray(centre)


def arrangement(kind, n, rng, limit):
    """Box centres relative to CENTRE, or None if this draw does not fit."""
    if n == 1:
        return np.array([[rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)]])
    if kind == "fitta":
        pts = cluster(n, rng, pitch=rng.uniform(0.30, 0.34))
    elif kind == "sparsa":
        pts = []
        for _ in range(4000):
            r, a = limit * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            p = (r * math.cos(a), r * math.sin(a))
            if spaced(pts, p, 0.50 if limit >= 0.6 else 0.38):     # small circle: less room
                pts.append(p)
            if len(pts) == n:
                break
        if len(pts) < n:
            return None
        pts = np.array(pts) - np.mean(pts, 0) * 0.5
    elif kind == "fila/arco":
        pitch = 0.36
        if (n - 1) * pitch / 2 <= limit:                # straight line
            a = rng.uniform(0, math.pi)
            t = (np.arange(n) - (n - 1) / 2) * pitch
            pts = np.stack([t * math.cos(a), t * math.sin(a)], 1)
        else:                                           # arc of radius rho through the centre region
            rho = min(0.55 if n < 7 else 0.47, 0.9 * limit)
            dth = pitch / rho
            th0 = rng.uniform(0, 2 * math.pi)
            th = th0 + (np.arange(n) - (n - 1) / 2) * dth
            pts = np.stack([rho * np.cos(th), rho * np.sin(th)], 1)
            pts -= pts.mean(0)
        pts += rng.uniform(-0.02, 0.02, pts.shape)
    else:                                               # two groups
        a = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(a), math.sin(a)])
        n1 = (n + 1) // 2
        sep = (0.42 if n >= 5 else 0.36) if limit >= 0.6 else 0.33
        pts = np.vstack([cluster(n1, rng, 0.32, -sep * u), cluster(n - n1, rng, 0.32, sep * u)])
    if np.max(np.hypot(*pts.T)) > limit:
        return None
    for i in range(n):
        for j in range(i):
            if math.hypot(*(pts[i] - pts[j])) < MIN_GAP_M:
                return None
    return pts


def inside(p, extra=0.0):
    return (ARENA[0] + MARGIN_M + extra <= p[0] <= ARENA[1] - MARGIN_M - extra and
            ARENA[2] + MARGIN_M + extra <= p[1] <= ARENA[3] - MARGIN_M - extra)


def random_robots(boxes, rng, tries=20000, dmin=0.9):
    """4 robot positions at dmin-1.8 m from the box centre, at least 40 deg apart, clear of the boxes, and
    inside the arena also after the formation has turned by SWEEP_DEG (counter-clockwise) around it."""
    c = boxes.mean(0)
    for _ in range(tries):
        ang = np.sort(rng.uniform(0, 2 * math.pi, 4))
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
        if gaps.min() < math.radians(40):
            continue
        dist = rng.uniform(dmin, 1.8, 4)
        ok = True
        for a, d in zip(ang, dist):
            for turn in np.radians(np.arange(0, SWEEP_DEG + 1e-6, STEP_DEG)):
                p = c + d * np.array([math.cos(a + turn), math.sin(a + turn)])
                if not inside(p, 0.05) or np.min(np.hypot(*(boxes - p).T)) < KEEP_OUT_M + 0.12:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out = []
            for a, d in zip(ang, dist):
                p = c + d * np.array([math.cos(a), math.sin(a)])
                out.append([round(float(p[0]), 3), round(float(p[1]), 3),
                            round(math.degrees(math.atan2(c[1] - p[1], c[0] - p[0])), 1)])
            return out
    return None


def stations(radius):
    """Round 0 positions on the circle (launch_all --radius, --first-angle-deg 0), facing CENTRE."""
    out = []
    for i in range(4):
        a = math.radians(90.0 * i)
        p = (CENTRE[0] + radius * math.cos(a), CENTRE[1] + radius * math.sin(a))
        out.append([round(p[0], 3), round(p[1], 3), round(math.degrees(a) + 180.0, 1)])
    return out


def obstacles(n, boxes, rng, robots_round0, kind="obstacle"):
    """n objects (about 30 cm) between the boxes and the robots of round 0: on the line from CENTRE to a
    robot, 0.75-0.82 m from CENTRE, clear of the boxes."""
    out = []
    for k in range(200):
        if len(out) == n:
            break
        r = robots_round0[int(rng.integers(len(robots_round0)))]
        u = np.array(r[:2]) - np.asarray(CENTRE)
        u = u / np.linalg.norm(u)
        p = np.asarray(CENTRE) + u * rng.uniform(0.75, 0.82) + rng.normal(0, 0.03, 2)
        rel = boxes - (p - np.asarray(CENTRE))
        if np.min(np.hypot(*rel.T)) < 0.30 or not spaced([o[:2] for o in out], p, 0.6):
            continue
        out.append([round(float(p[0]), 3), round(float(p[1]), 3), kind])
    return out


def white_objects(n, boxes, rng, limit):
    """n white objects that are not boxes, among the boxes (away from them by 30 cm)."""
    out = []
    for _ in range(2000):
        if len(out) == n:
            break
        r, a = limit * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if np.min(np.hypot(*(boxes - p).T)) < 0.32 or not spaced([np.array(o[:2]) - CENTRE for o in out], p, 0.35):
            continue
        q = p + np.asarray(CENTRE)
        out.append([round(float(q[0]), 3), round(float(q[1]), 3), "white object"])
    return out


def command(row, n_boxes):
    parts = ["python -m acquisition.launch_all"]
    for rid, rb, cal in ROBOTS:
        parts.append(f"--robot {rid}:{rb}" + (f":{cal}" if cal else ""))
    parts.append(f"--layout-id {row['id']}")
    parts += [f"--box {b}:box" for b in BOX_IDS[:n_boxes]]
    parts += ["--ground-truth config/ground_truth_positions.json", "--regions config/optitrack_regions.json",
              "--out data/survey_v2/", "--aruco-marker-size 0.077", "--range-fit 1.064 -0.080",
              "--camera-offset-forward-m 0.12", "--instance-threshold-m 0.2",
              f"--step-deg {STEP_DEG:g} --sweep-deg {SWEEP_DEG:g}",
              f"--circle-center {CENTRE[0]:g} {CENTRE[1]:g}", "--no-robot-avoidance"]
    r = RADIUS[row["robots"]]
    if r:
        parts.append(f"--radius {r:g}")
    if "yaw-offsets" in row["extra"]:
        parts.append("--yaw-offsets-deg -20 0 20")
    return " ".join(parts)


def build(csv_path, seed):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    layouts = []
    for row in rows:
        n = int(row["boxes"])
        kind = row["arrangement"].split(" ")[0]
        limit = box_limit(row["robots"])
        if "ostacoli" in row["extra"]:
            limit = min(limit, 0.50)                    # leave room for the obstacles before the robots
        for _ in range(500):
            pts = arrangement(kind, n, rng, limit)
            if pts is None:
                continue
            boxes = pts + np.asarray(CENTRE)
            dmin = 1.3 if "ostacoli" in row["extra"] else 0.9      # robots clear of the obstacles
            robots = (random_robots(boxes, rng, dmin=dmin) if RADIUS[row["robots"]] is None
                      else stations(RADIUS[row["robots"]]))
            if robots is not None:
                break
        else:
            raise SystemExit(f"{row['id']}: no layout found")
        extra = []
        if "ostacoli" in row["extra"] or "ostacoli e" in row["extra"]:
            extra += obstacles(int(row["extra"][0]) if row["extra"][0].isdigit() else 2, pts, rng, robots)
        if "bianc" in row["extra"]:
            k = 2 if row["extra"].startswith("2") or "sparsi" in row["extra"] else 1
            extra += white_objects(k, pts, rng, min(limit, 0.6))
        layouts.append(dict(
            id=row["id"], day=int(row["day"]), block=row["block"], split=row["split"], n_boxes=n,
            arrangement=row["arrangement"], robots=row["robots"], radius_m=RADIUS[row["robots"]],
            note=row["extra"], centre=list(CENTRE),
            boxes=[[round(float(x), 3), round(float(y), 3), round(float(rng.uniform(0, 120)), 0)] for x, y in boxes],
            box_ids=BOX_IDS[:n], robot_targets=robots, objects=extra, command=command(row, n)))
    return layouts


def triangle(x, y, yaw_deg):
    a = math.radians(yaw_deg) + np.array([0, 2 * math.pi / 3, 4 * math.pi / 3])
    return np.stack([x + BOX_R_M * np.cos(a), y + BOX_R_M * np.sin(a)], 1)


def draw_scene(ax, L, zoom):
    """The layout on ax: the whole arena (zoom False) or the box area at a larger scale (zoom True)."""
    from matplotlib.patches import Circle, Polygon
    cx, cy = L["centre"]
    if not zoom:
        ax.add_patch(Polygon(CORNERS, closed=True, fill=False, ec="0.5", lw=1.5))
        ax.add_patch(Polygon([(ARENA[0] + MARGIN_M, ARENA[2] + MARGIN_M), (ARENA[1] - MARGIN_M, ARENA[2] + MARGIN_M),
                              (ARENA[1] - MARGIN_M, ARENA[3] - MARGIN_M), (ARENA[0] + MARGIN_M, ARENA[3] - MARGIN_M)],
                             closed=True, fill=False, ec="0.8", ls="--", lw=0.8))
        for r in (0.5, 1.0, 1.5):
            ax.add_patch(Circle((cx, cy), r, fill=False, ec="0.88", lw=0.8))
        if L["radius_m"]:
            ax.add_patch(Circle((cx, cy), L["radius_m"], fill=False, ec="tab:blue", lw=1, ls=":"))
            for i in range(4):
                for t in np.arange(STEP_DEG, SWEEP_DEG + 1e-6, STEP_DEG):
                    a = math.radians(90 * i + t)
                    ax.plot(cx + L["radius_m"] * math.cos(a), cy + L["radius_m"] * math.sin(a), "o", c="tab:blue",
                            ms=3, alpha=0.35)
        for k, (x, y, h) in enumerate(L["robot_targets"]):
            ax.add_patch(Circle((x, y), 0.17, fc="#cfe2f3", ec="tab:blue", lw=1.2))
            ax.annotate("", xy=(x + 0.28 * math.cos(math.radians(h)), y + 0.28 * math.sin(math.radians(h))),
                        xytext=(x, y), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5))
            ax.text(x, y - 0.30, f"R{k + 1}" if L["radius_m"] is None else f"S{k + 1}", ha="center", va="top",
                    fontsize=8, color="tab:blue")
    s = 0.25 if zoom else 0.12
    ax.plot([cx - s, cx + s], [cy, cy], c="k", lw=1)
    ax.plot([cx, cx], [cy - s, cy + s], c="k", lw=1)
    for k, (x, y, yaw) in enumerate(L["boxes"]):
        ax.add_patch(Polygon(triangle(x, y, yaw), closed=True, fc="white", ec="k", lw=1.6 if zoom else 1.2))
        ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=12 if zoom else 7, fontweight="bold")
    for x, y, kind in L["objects"]:
        if kind == "obstacle":
            ax.add_patch(Circle((x, y), 0.15, fc="#f4cccc", ec="tab:red", lw=1.2, hatch="//"))
        else:
            ax.add_patch(Circle((x, y), 0.10, fc="#fff2cc", ec="tab:orange", lw=1.2))
    ax.set_aspect("equal")
    if zoom:
        h = 0.95
        ax.set_xlim(cx - h, cx + h)
        ax.set_ylim(cy - h, cy + h)
        ticks = np.arange(-0.9, 0.91, 0.1)
        ax.set_xticks(cx + ticks, [f"{100 * t:+.0f}" if abs(round(10 * t)) % 3 == 0 else "" for t in ticks],
                      fontsize=7)
        ax.set_yticks(cy + ticks, [f"{100 * t:+.0f}" if abs(round(10 * t)) % 3 == 0 else "" for t in ticks],
                      fontsize=7)
        ax.grid(alpha=0.35)
        ax.set_xlabel("dx from the centre + [cm] (Motive x)")
        ax.set_ylabel("dy [cm] (Motive y)")
        ax.set_title("boxes (10 cm grid)", fontsize=10)
    else:
        pad = 0.15
        ax.set_xlim(ARENA[0] - pad, ARENA[1] + pad)
        ax.set_ylim(ARENA[2] - pad, ARENA[3] + pad)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Motive x [m]")
        ax.set_ylabel("Motive y [m]")
        ax.set_title("arena and robots", fontsize=10)


def draw_page(fig, L):
    fig.clf()
    cx, cy = L["centre"]
    draw_scene(fig.add_axes([0.05, 0.24, 0.33, 0.64]), L, zoom=False)
    draw_scene(fig.add_axes([0.42, 0.24, 0.33, 0.64]), L, zoom=True)
    title = f"{L['id']}   day {L['day']}   {L['block']}   ({L['split']})"
    fig.text(0.05, 0.955, title, fontsize=16, fontweight="bold")
    fig.text(0.05, 0.925, f"{L['n_boxes']} boxes, {L['arrangement']}, robots: {L['robots']}"
             + (f"   |   {L['note']}" if L["note"] else ""), fontsize=10)
    x0, y0 = 0.785, 0.87
    fig.text(x0, y0, "Boxes from the centre +", fontsize=10, fontweight="bold")
    fig.text(x0, y0 - 0.022, "#   dx [cm]  dy [cm]", fontsize=8, family="monospace")
    for k, (x, y, yaw) in enumerate(L["boxes"]):
        fig.text(x0, y0 - 0.042 - 0.019 * k, f"{k + 1:<2} {100 * (x - cx):+7.0f} {100 * (y - cy):+8.0f}",
                 fontsize=8, family="monospace")
    y1 = y0 - 0.06 - 0.019 * len(L["boxes"]) - 0.02
    if L["radius_m"] is None:
        fig.text(x0, y1, "Robots, round 0", fontsize=10, fontweight="bold")
        fig.text(x0, y1 - 0.022, "by hand, facing the boxes", fontsize=8)
        fig.text(x0, y1 - 0.042, "#   x [m]   y [m]", fontsize=8, family="monospace")
        for k, (x, y, h) in enumerate(L["robot_targets"]):
            fig.text(x0, y1 - 0.062 - 0.019 * k, f"R{k + 1} {x:+.2f}  {y:+.2f}", fontsize=8, family="monospace")
        y1 -= 0.08 + 0.019 * 4 + 0.02
    else:
        fig.text(x0, y1, "Robots", fontsize=10, fontweight="bold")
        fig.text(x0, y1 - 0.022, f"anywhere near the {L['radius_m']:g} m circle:", fontsize=8)
        fig.text(x0, y1 - 0.040, "each drives to the nearest", fontsize=8)
        fig.text(x0, y1 - 0.058, "station S1-S4", fontsize=8)
        y1 -= 0.09
    if L["objects"]:
        fig.text(x0, y1, "Other objects", fontsize=10, fontweight="bold")
        for k, (x, y, kind) in enumerate(L["objects"]):
            fig.text(x0, y1 - 0.022 - 0.019 * k, f"{kind:<12}{100 * (x - cx):+5.0f}{100 * (y - cy):+5.0f}",
                     fontsize=8, family="monospace")
        y1 -= 0.04 + 0.019 * len(L["objects"])
    fig.text(x0, max(y1 - 0.01, 0.27), "Rotate the boxes as drawn,\nby eye. Unused boxes out\nof the arena.",
             fontsize=8, va="top")
    fig.text(0.05, 0.16, f"Place and capture:   python -m tools.place_check --layout {L['id']}", fontsize=12,
             family="monospace", fontweight="bold")
    import textwrap
    fig.text(0.05, 0.125, "(it runs: " + "\n".join(textwrap.wrap(L["command"], 150)) + ")", fontsize=6,
             family="monospace", va="top", color="0.4")
    fig.text(0.05, 0.04, "[ ] boxes placed   [ ] robots placed   [ ] BOXES rows show yaw   [ ] 4 rounds done   "
             "[ ] done in the CSV", fontsize=10)


def cover(fig, layouts):
    fig.clf()
    lines = [
        ("Capture plan, second real data set", 18, "bold"),
        ("", 10, None),
        (f"{len(layouts)} layouts x 4 rounds = {4 * len(layouts)} scenes (5 readings each). One page per layout, "
         "in capture order.", 10, None),
        (f"Arena centre (the + on every page): Motive x = {CENTRE[0]:g} m, y = {CENTRE[1]:g} m. "
         "Mark it on the floor with tape.", 10, None),
        ("Box positions are given from that centre in cm (dx along Motive x, dy along Motive y) and in Motive "
         "metres.", 10, None),
        ("Fastest way: place the boxes roughly, then run  python -m tools.place_check --layout Rxx : it reads "
         "OptiTrack and", 10, None),
        ("says, live, how far every box (and, for the random family, every robot) is from its target. Within "
         "5 cm is fine.", 10, None),
        ("Triangles: box orientation, by eye. Blue circles: robots with heading. Red hatched: obstacle "
         "(about 30 cm).", 10, None),
        ("Yellow: white object that is not a box. Dotted blue circle: the circle the robots stand on.", 10, None),
        ("", 10, None),
        ("Days", 12, "bold"),
    ]
    for d in sorted({L["day"] for L in layouts}):
        ids = [L["id"] for L in layouts if L["day"] == d]
        boxes = sorted({L["n_boxes"] for L in layouts if L["day"] == d})
        lines.append((f"day {d}: {ids[0]}-{ids[-1]} ({len(ids)} layouts; boxes {', '.join(map(str, boxes))})",
                      10, None))
    lines += [("", 10, None), ("Rules", 12, "bold"),
              ("- Once, before starting: find which way Motive +x and +y point on the floor (slide a box and watch "
               "place_check) and mark them with tape arrows at the centre.", 10, None),
              ("- After the first 2-3 layouts of day 1: commit and push, and wait for the check before going on.",
               10, None),
              ("- place_check: Enter starts the capture of that layout (launch_all with the right options).", 10,
               None),
              ("- --box lists only the boxes on the floor (ids 11-17 in this order); the others out of the arena.",
               10, None),
              ("- Keep --camera-offset-forward-m 0.12 (the pipeline corrects the lens offset).", 10, None),
              ("- Obstacle layouts: the robots drive along the circle; keep an eye on them near the obstacles.",
               10, None),
              ("- End of each day: git add data/survey_v2 docs/capture_plan_v2.csv, commit, push.", 10, None)]
    y = 0.95
    for text, size, weight in lines:
        fig.text(0.06, y, text, fontsize=size, fontweight=weight or "normal")
        y -= 0.032 if size <= 10 else 0.045


def write_pdf(layouts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fig = plt.figure(figsize=(11.69, 8.27))             # A4 landscape
    with PdfPages(path) as pdf:
        cover(fig, layouts)
        pdf.savefig(fig)
        for L in layouts:
            draw_page(fig, L)
            pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="docs/capture_plan_v2.csv")
    ap.add_argument("--json", default="docs/capture_plan_v2.json")
    ap.add_argument("--pdf", default="docs/capture_guide_v2.pdf")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pdf-only", action="store_true")
    a = ap.parse_args()
    if a.pdf_only:
        with open(a.json, encoding="utf-8") as f:
            layouts = json.load(f)
    else:
        layouts = build(a.csv, a.seed)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(layouts, f, indent=1)
        print(f"{len(layouts)} layouts -> {a.json}")
    write_pdf(layouts, a.pdf)
    print(f"guide -> {a.pdf}")


if __name__ == "__main__":
    main()
