"""Does the free space of the IR beams (vf/fusion/freespace.py) help the fusion? Two modes, nothing else changed:

  sim     random scenes in the arena: robots range the boxes with the IR as acquisition/ir_scan.py does (one
          reading per visible box and sweep, beam pitched to mid-face), with the faults seen on the robots:
            aim error        the gimbal points off the box centre (sd --aim-sd-deg; a share --bad-aim of large
                             errors, uniform within --bad-aim-deg: a camera that aims wrong)
            edge mix         a beam grazing a box edge reads in between the face and what is behind it
            slipped beam     a beam that misses reads the next box, a robot, the wall, or the floor (at twice
                             the box distance; the height test of ir_scan lets about half of these through)
          then the fusion of demo/live.py (demo.methods: pooling, MHT with sensor model, graph network model A)
          on: base (as now), pre (detections other beams went through dropped before the fusion), post (fused
          boxes the beams contradict dropped), pre+post.
  replay  the same four on the log of a real run (<live dir>/log/*.jsonl), in windows, and a picture with the
          occupancy grid of the beams.

    python -m tools.ir_freespace sim --scenes 200 --out results/ir_freespace/sim.csv
    python -m tools.ir_freespace sim --scenes 200 --bad-aim 0.2 --out results/ir_freespace/sim_badaim.csv
    python -m tools.ir_freespace replay --live-dir data/live_ir/ --png results/ir_freespace/replay.png
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import tempfile
import time

import numpy as np

from vf.fusion import freespace as fs

BOX_R = 0.075                  # a beam within this of a box centre meets it (triangle of apothem 6.8 cm)
EDGE_W = 0.02                  # edge band around BOX_R where the reading mixes the two surfaces
ROBOT_R = 0.18
IR_H, AIM_H, BOX_H = 0.30, 0.15, 0.30
VARIANTS = ("base", "pre", "post", "pre+post")


# ====================================================================
# Simulated IR scan
# ====================================================================

def cast(o, u, boxes, robots, arena, skip_robot=None):
    """Surfaces met by the ray o + t u, sorted: [(t, kind, index, offset)], offset = distance of the centre from
    the axis (edges). Walls of the arena last."""
    hits = []
    for kind, pts, r in (("box", boxes, BOX_R + EDGE_W), ("robot", robots, ROBOT_R)):
        for k, p in enumerate(pts):
            if kind == "robot" and k == skip_robot:
                continue
            w = np.asarray(p[:2]) - o
            t = float(w @ u)
            c = abs(float(w[0] * u[1] - w[1] * u[0]))
            if t > 0 and c < r:
                hits.append((t - math.sqrt(max(r * r - c * c, 0.0)), kind, k, c))
    x0, x1, y0, y1 = arena
    tw = []
    for i, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
        if abs(u[i]) > 1e-9:
            tw += [(lo - o[i]) / u[i], (hi - o[i]) / u[i]]
    hits.append((min(t for t in tw if t > 0), "wall", -1, 0.0))
    return sorted(hits)


def read_ir(o, u, target_horiz, boxes, robots, arena, me, rng, sd):
    """(slant, pitch, kind) of one reading: the beam pitched down to hit the target at mid-face height."""
    pitch = math.atan2(IR_H - AIM_H, max(target_horiz, 0.1))
    floor = IR_H / math.tan(pitch)
    hits = cast(o, u, boxes, robots, arena, skip_robot=me)
    t, kind, k, c = hits[0]
    if kind == "box" and c > BOX_R:                           # grazing an edge: in between the two surfaces
        nxt = hits[1][0] if len(hits) > 1 else floor
        t, kind = t + rng.uniform(0, 1) * (min(nxt, floor) - t), "edge"
    if t > floor:
        t, kind = floor, "floor"
    t += rng.normal(0, sd)
    return t / math.cos(pitch), pitch, kind


def random_scene(rng, arena, n_boxes, n_robots, margin=0.35):
    x0, x1, y0, y1 = arena
    pts = []

    def place(clear):
        for _ in range(2000):
            p = np.array([rng.uniform(x0 + margin, x1 - margin), rng.uniform(y0 + margin, y1 - margin)])
            if all(np.hypot(*(p - q[:2])) > clear for q in pts):
                return p
        return None

    boxes, robots = [], []
    for _ in range(n_boxes):
        p = place(0.35)
        if p is not None:
            boxes.append(p)
            pts.append(p)
    for _ in range(n_robots):
        p = place(0.5)
        if p is not None:
            c = np.mean(boxes, 0) if boxes else np.zeros(2)
            yaw = math.atan2(c[1] - p[1], c[0] - p[0]) + rng.normal(0, 0.5)
            robots.append(np.array([p[0], p[1], yaw]))
            pts.append(p)
    return boxes, robots


def simulate_states(rng, a, arena, boxes, robots):
    """IR states of every robot, as acquisition/ir_scan.py writes them, plus the kind of surface each read."""
    from acquisition.ir_scan import hidden
    bdicts = [dict(id=k, x=float(b[0]), y=float(b[1])) for k, b in enumerate(boxes)]
    states = collections.defaultdict(list)
    now = time.time()
    for i, body in enumerate(robots):
        others = [(float(r[0]), float(r[1])) for j, r in enumerate(robots) if j != i]
        vis = [b for b in bdicts if math.hypot(b["x"] - body[0], b["y"] - body[1]) <= a.max_range
               and (a.aim_hidden or not hidden(body, b, bdicts, others, 0.10, 0.20))]
        for sweep in range(a.sweeps):
            for b in vis:
                o = body[:2] + 0.10 * np.array([math.cos(body[2]), math.sin(body[2])])
                ang = math.atan2(b["y"] - o[1], b["x"] - o[0])
                if rng.uniform() < a.bad_aim:
                    err = rng.uniform(-a.bad_aim_deg, a.bad_aim_deg)
                else:
                    err = rng.normal(0, a.aim_sd_deg)
                head = ang + math.radians(err)
                u = np.array([math.cos(head), math.sin(head)])
                target = math.hypot(b["x"] - o[0], b["y"] - o[1]) - fs.APOTHEM_M
                slant, pitch, kind = read_ir(o, u, target, boxes, [r[:2] for r in robots], arena, i, rng, a.ir_sd)
                horiz, hh = slant * math.cos(pitch), IR_H - slant * math.sin(pitch)
                dets = []
                if 0.0 <= hh <= 0.6:                           # the height test of ir_scan.range_box
                    d = horiz + (fs.APOTHEM_M if a.report == "centre" else 0.0)
                    dets.append(dict(class_name="box", confidence=1.0, distance=d, bearing_rad=0.0,
                                     viewing_angle_rad=0.0 if a.view == "zero" else None,
                                     position=[o[0] + d * u[0], o[1] + d * u[1]], bbox=None, kind=kind))
                states[f"S{i}"].append(dict(robot_id=f"S{i}", t=now - 30 + 0.1 * len(states[f"S{i}"]),
                                            pose=[float(o[0]), float(o[1]), float(head)], detections=dets,
                                            ir_slant_m=slant, gimbal=[0.0, math.degrees(pitch)], kind=kind))
        for _ in range(a.probe_rays):                      # free space only: a lidar / a gimbal sweep
            o = body[:2]
            head = rng.uniform(-math.pi, math.pi)
            u = np.array([math.cos(head), math.sin(head)])
            t = cast(o, u, boxes, [r[:2] for r in robots], arena, skip_robot=i)[0][0] + rng.normal(0, a.ir_sd)
            states[f"S{i}"].append(dict(robot_id=f"S{i}", t=now - 30 + 0.1 * len(states[f"S{i}"]),
                                        pose=[float(o[0]), float(o[1]), float(head)], detections=[],
                                        ir_slant_m=float(t), gimbal=[0.0, 0.0], kind="probe"))
        # the fan of the gimbal: --probe-fan beams evenly spread over +-probe_fan_deg around the chassis heading
        # (what the camera sees), pitched down probe_pitch_deg: the beam meets the floor at IR_H / tan(pitch)
        if a.probe_fan > 0:
            pitch = math.radians(a.probe_pitch_deg)
            floor = IR_H / math.tan(pitch) if pitch > 1e-6 else float("inf")
            for off in np.linspace(-a.probe_fan_deg, a.probe_fan_deg, a.probe_fan):
                head = body[2] + math.radians(off + rng.normal(0, a.aim_sd_deg))
                u = np.array([math.cos(head), math.sin(head)])
                o = body[:2] + 0.10 * u
                t = min(cast(o, u, boxes, [r[:2] for r in robots], arena, skip_robot=i)[0][0], floor)
                t += rng.normal(0, a.ir_sd)
                states[f"S{i}"].append(dict(robot_id=f"S{i}", t=now - 30 + 0.1 * len(states[f"S{i}"]),
                                            pose=[float(o[0]), float(o[1]), float(head)], detections=[],
                                            ir_slant_m=float(t / math.cos(pitch)),
                                            gimbal=[float(off), math.degrees(pitch)], kind="fan"))
    return states


# ====================================================================
# The fusion of demo/live.py, with and without the free space
# ====================================================================

class Fusion:
    def __init__(self, a):
        from demo.methods import Classical, Gnn
        self.classical = Classical(a.sensor_calib, a.sensor_model, a.regions)
        self.gnn = None if a.no_gnn else Gnn(a.gnn, a.sensor_calib)
        self.work = tempfile.mkdtemp(prefix="ir_freespace_")

    def run(self, states, truth):
        """{method: [[x, y], ...]} of the pooled states ({robot: [state, ...]})."""
        from demo.methods import Recent, write_round
        rec = Recent(self.work, keep=1000000)
        for q, sts in states.items():
            for s in sts:
                rec.states[q].append(s)
        pooled = rec.pooled(1e9, 1e9)
        write_round(self.work, pooled, [[float(b[0]), float(b[1]), 0.0] for b in truth])
        out = {"mht": self.classical.run(self.work)[0]["mht_sensor"]}
        if self.gnn is not None:
            out["gnn"] = self.gnn.run(self.work)
        return out


def gate(states, arena, margin):
    """Detections on the border of the arena dropped (walls), the states kept (the beam still went that far)."""
    if arena is None:
        return states, 0
    n, out = 0, {}
    for q, sts in states.items():
        out[q] = []
        for s in sts:
            dets = [d for d in s.get("detections", []) if fs.in_arena(d["position"], arena, margin)]
            n += len(s.get("detections", [])) - len(dets)
            out[q].append(dict(s, detections=dets))
    return out, n


def variants(fusion, states, truth, threshold=0.0):
    """{variant: ({method: estimates}, info)}."""
    flat = [s for sts in states.values() for s in sts]
    ray_of = {id(s): fs.ray_from_state(s) for s in flat}
    rays = [r for r in ray_of.values() if r is not None]
    # pre: detections that the beams of the others went through
    pre_states, dropped = {}, []
    for q, sts in states.items():
        out = []
        for s in sts:
            dets = [dict(d, ray=ray_of.get(id(s))) for d in s.get("detections", [])]
            kept, drop = fs.filter_detections(dets, rays, threshold)
            dropped += drop
            out.append(dict(s, detections=[{k: v for k, v in d.items() if k not in ("ray", "n_free", "n_hit")}
                                           for d in kept]))
        pre_states[q] = out
    base = fusion.run(states, truth)
    pre = fusion.run(pre_states, truth)
    res = {"base": base, "pre": pre,
           "post": {m: fs.filter_objects(e, rays, threshold)[0] for m, e in base.items()},
           "pre+post": {m: fs.filter_objects(e, rays, threshold)[0] for m, e in pre.items()}}
    return res, dict(rays=rays, dropped=dropped)


def score(est, truth):
    from demo.methods import judge
    j = judge(est, [list(map(float, b[:2])) for b in truth])
    return dict(verdict=j["verdict"], n_est=j["n_est"], n_true=j["n_true"], n_matched=j["n_matched"],
                false=j["n_est"] - j["n_matched"], missed=j["n_true"] - j["n_matched"], rmse_cm=j["rmse_cm"])


def summarise(rows):
    print(f"\n{'method':6} {'variant':9} {'scenes':>6} {'correct':>8} {'false/scene':>12} {'missed/scene':>13}")
    keys = sorted({(r["method"], r["variant"]) for r in rows}, key=lambda k: (k[0], VARIANTS.index(k[1])))
    for m, v in keys:
        rs = [r for r in rows if r["method"] == m and r["variant"] == v]
        print(f"{m:6} {v:9} {len(rs):6d} {100 * np.mean([r['verdict'] == 'correct' for r in rs]):7.1f}% "
              f"{np.mean([r['false'] for r in rs]):12.2f} {np.mean([r['missed'] for r in rs]):13.2f}")


def write_csv(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {path}")


# ====================================================================
# Modes
# ====================================================================

def run_sim(a):
    from vf.fusion.sensor_model import arena_from_regions
    rng = np.random.default_rng(a.seed)
    arena = arena_from_regions(a.regions)
    fusion = Fusion(a)
    rows, kinds = [], collections.Counter()
    for sc in range(a.scenes):
        nb = int(rng.integers(a.boxes[0], a.boxes[1] + 1))
        nr = int(rng.integers(a.robots[0], a.robots[1] + 1))
        boxes, robots = random_scene(rng, arena, nb, nr)
        states = simulate_states(rng, a, arena, boxes, robots)
        if a.arena_gate_m is not None:
            states, _ = gate(states, arena, a.arena_gate_m)
        res, info = variants(fusion, states, boxes, a.threshold)
        dets = [d for sts in states.values() for s in sts for d in s["detections"]]
        for d in dets:
            kinds[("all", d["kind"])] += 1
        for d in info["dropped"]:
            kinds[("dropped", d["kind"])] += 1
        bad = sum(1 for d in dets if d["kind"] != "box")
        for v, est in res.items():
            for m, e in est.items():
                rows.append(dict(scene=sc, n_boxes=len(boxes), n_robots=len(robots), detections=len(dets),
                                 wrong_detections=bad, method=m, variant=v, **score(e, boxes)))
        if (sc + 1) % 20 == 0:
            print(f"{sc + 1}/{a.scenes} scenes", flush=True)
    print("\ndetections by surface read (all / dropped by the free-space test):")
    for k in sorted({k for _, k in kinds}):
        print(f"  {k:6} {kinds[('all', k)]:6d} / {kinds[('dropped', k)]:5d}")
    summarise(rows)
    write_csv(a.out, rows)


def read_log(live_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(live_dir, "log", "*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["t"])


def run_replay(a):
    from vf.fusion.sensor_model import arena_from_regions
    log = read_log(a.live_dir)
    if not log:
        raise SystemExit(f"no log in {a.live_dir}/log")
    fusion = Fusion(a)
    rows, t0, t_end = [], log[0]["t"], log[-1]["t"]
    t = t0 + a.window_s
    last = None
    while t <= t_end + 1e-6:
        win = [r for r in log if t - a.window_s <= r["t"] <= t]
        truth = next((r["boxes"] for r in reversed(win) if r.get("boxes")), None)
        if win and truth:
            states = collections.defaultdict(list)
            for r in win:
                states[r["robot_id"]].append(r)
            if a.arena_gate_m is not None:
                states, _ = gate(states, arena_from_regions(a.regions), a.arena_gate_m)
            tb = [[b["x"], b["y"]] for b in truth]
            res, info = variants(fusion, states, tb, a.threshold)
            for v, est in res.items():
                for m, e in est.items():
                    rows.append(dict(t=round(t - t0, 1), readings=len(win), method=m, variant=v,
                                     dropped=len(info["dropped"]), **score(e, tb)))
            last = (win, tb, res, info)
        t += a.step_s
    summarise(rows)
    write_csv(a.out, rows)
    if a.png and last:
        picture(a.png, last, arena_from_regions(a.regions))


def picture(path, last, arena):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    win, truth, res, info = last
    g, ext = fs.occupancy_grid(info["rays"], arena)
    fig, ax = plt.subplots(figsize=(8, 7))
    lim = max(1.0, np.abs(g).max())
    im = ax.imshow(g, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, label="log-odds (blue: beams went through, red: beams ended)")
    for r in info["rays"]:
        ax.plot([r.o[0], r.end[0]], [r.o[1], r.end[1]], color="0.3", lw=0.5, alpha=0.5)
    for b in truth:
        ax.add_patch(plt.Circle(b, 0.136, fill=False, color="k", lw=1.5))
    for d in info["dropped"]:
        ax.plot(*d["position"], "x", color="orange", ms=9, mew=2)
    for (m, mk, col) in (("mht", "+", "red"), ("gnn", "x", "purple")):
        for v, ls in (("base", 1.0), ("pre+post", 0.5)):
            e = np.array(res[v].get(m, [])).reshape(-1, 2)
            if len(e):
                ax.plot(e[:, 0], e[:, 1], mk, color=col, ms=14 * ls + 6, mew=2, alpha=0.9,
                        label=f"{m.upper()} {v} ({len(e)})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("free space of the IR beams (last window); orange x: detections dropped", fontsize=10)
    ax.set_aspect("equal")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"written {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sim", "replay"])
    ap.add_argument("--out", default=None, help="CSV of every scene / window, method and variant")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="a point is empty when log P(box) / P(nothing) of the beams is below this")
    ap.add_argument("--sensor-calib", default="config/sensor_calib_ir.json")
    ap.add_argument("--sensor-model", default="config/sensor_model_ir.json")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--gnn", default="results/plan_v2/M4/gnn/detection_full_full_h2")
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--arena-gate-m", type=float, default=None,
                    help="before everything, drop the detections within this of the arena border (the beam read a "
                         "wall): prior knowledge of the arena, not free space; default: off")
    g = ap.add_argument_group("sim")
    g.add_argument("--scenes", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--boxes", type=int, nargs=2, default=[3, 7], metavar=("MIN", "MAX"))
    g.add_argument("--robots", type=int, nargs=2, default=[2, 4], metavar=("MIN", "MAX"))
    g.add_argument("--sweeps", type=int, default=2, help="readings per visible box and robot")
    g.add_argument("--max-range", type=float, default=3.5)
    g.add_argument("--aim-sd-deg", type=float, default=1.5)
    g.add_argument("--bad-aim", type=float, default=0.1, help="share of readings with a large aim error")
    g.add_argument("--bad-aim-deg", type=float, default=8.0)
    g.add_argument("--ir-sd", type=float, default=0.024, help="range noise [m] (real run: 2.4 cm)")
    g.add_argument("--aim-hidden", action="store_true", help="also aim at boxes hidden behind others")
    g.add_argument("--probe-fan", type=int, default=0,
                   help="per robot, a fan of this many beams evenly spread over +-probe-fan-deg around the chassis "
                        "heading (the field of view of the camera), free space only: a gimbal sweep")
    g.add_argument("--probe-fan-deg", type=float, default=40.0)
    g.add_argument("--probe-pitch-deg", type=float, default=4.0,
                   help="the fan is pitched down this much (the IR is at the height of the box tops): it meets the "
                        "floor at 0.30 / tan(pitch), 4.3 m at 4 deg")
    g.add_argument("--probe-rays", type=int, default=0,
                   help="per robot, this many extra beams in random directions that only give free space (no "
                        "detection): what a 2D lidar or a gimbal sweep would add")
    g.add_argument("--report", choices=["face", "centre"], default="face",
                   help="face: the point the beam hits, as acquisition/ir_scan.py reports now; centre: that point "
                        "moved one apothem along the beam (the box centre, the same for every robot)")
    g.add_argument("--view", choices=["zero", "none"], default="none",
                   help="viewing angle given to the fusion: zero (as ir_scan now) or none (unknown: the MHT then "
                        "does not use the orientation of the box)")
    g = ap.add_argument_group("replay")
    g.add_argument("--live-dir", default="data/live_ir/")
    g.add_argument("--window-s", type=float, default=36.0)
    g.add_argument("--step-s", type=float, default=12.0)
    g.add_argument("--png", default=None)
    a = ap.parse_args()
    (run_sim if a.mode == "sim" else run_replay)(a)


if __name__ == "__main__":
    main()
