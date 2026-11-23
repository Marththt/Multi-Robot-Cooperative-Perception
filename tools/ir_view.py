"""Top view of the IR live run: what every robot is doing and what the fusion makes of it.

Drawn from the files of acquisition/ir_live_robots.py (<live dir>/boxes.json, <live dir>/log/<robot>.jsonl) and,
if the demo page runs, from its results (http://localhost:8000/state):

  grey circles     the true boxes (OptiTrack), with their rigid body id
  triangles        the robots (OptiTrack pose), coloured; a thin line shows where the gimbal points now
  coloured lines   the IR beams of the last --window-s seconds, from the sensor to the point the reading gives;
                   a dashed short line with an x: a reading with no valid range
  dots             those points: what each robot reports to the fusion
  red +            boxes found by MHT;  purple x: boxes found by the graph network (model A)
  table            per robot: readings in the window, median error of the range against the box it aimed at
  --freespace      also the free space of the beams (vf/fusion/freespace.py), computed here on the log of the window:
                   blue/red background = log-odds occupancy (blue: beams went through), thin grey lines = the
                   beams of the gimbal fans (acquisition/ir_scan.py --fan), orange x = detections the free space
                   drops, red circle / purple square = MHT / graph network on what is left (pre+post of
                   tools/ir_freespace.py), next to the results of the demo page (as it runs now)

    python -m tools.ir_view --live-dir data/live_ir/                    # window that refreshes every second
    python -m tools.ir_view --live-dir data/live_ir/ --png view.png    # one picture and exit
    python -m tools.ir_view --live-dir data/live_ir/ --png view.png --at -30   # as it was 30 s before the end of the log
    python -m tools.ir_view --live-dir data/live_ir/ --window-s 60 --freespace
"""

import argparse
import glob
import json
import math
import os
import time
import urllib.request

import numpy as np

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
BOX_R = 0.136                  # footprint radius of a box (2 x apothem)


def read_log(live_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(live_dir, "log", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return sorted(rows, key=lambda r: r["t"])


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def demo_results(url):
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=0.5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def free_space(fusion, recent, boxes):
    """MHT / graph network with the free space of the beams (tools/ir_freespace.variants) on the window."""
    from tools.ir_freespace import variants
    if not recent or not boxes:
        return None
    states = {}
    for r in recent:
        states.setdefault(r["robot_id"], []).append(r)
    try:
        res, info = variants(fusion, states, [[b["x"], b["y"]] for b in boxes])
    except Exception as e:
        print(f"free space: {e}")
        return None
    return dict(res=res, rays=info["rays"], dropped=info["dropped"])


def draw(ax, rows, now, window_s, truth, demo, arena, fusion=None):
    from matplotlib.patches import Circle, Polygon
    ax.clear()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if arena:
        x0, x1, y0, y1 = arena
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="0.6", lw=1)
    recent_all = [r for r in rows if now - window_s <= r["t"] <= now]
    fans = [r for r in recent_all if r.get("kind") == "fan"]
    recent = [r for r in recent_all if r.get("kind") != "fan"]
    robots = sorted({r["robot_id"] for r in rows})
    colour = {q: COLOURS[k % len(COLOURS)] for k, q in enumerate(robots)}
    # true boxes: from boxes.json (live) or from the last log row (replay)
    boxes = truth if truth is not None else next((r.get("boxes") for r in reversed(recent) if r.get("boxes")), []) or []
    fsr = free_space(fusion, recent_all, boxes) if fusion is not None else None
    if fsr is not None:
        from vf.fusion import freespace as fs
        if arena:
            g, ext = fs.occupancy_grid(fsr["rays"], arena)
            lim = max(1.0, float(np.abs(g).max()))
            ax.imshow(g, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.6, zorder=0)
        for r in fans:
            ray = fs.ray_from_state(r)
            if ray is not None:
                ax.plot([ray.o[0], ray.end[0]], [ray.o[1], ray.end[1]], color="0.5", lw=0.4, alpha=0.5)
        for d in fsr["dropped"]:
            ax.plot(*d["position"], "x", color="orange", ms=10, mew=2.5)
    for b in boxes:
        ax.add_patch(Circle((b["x"], b["y"]), BOX_R, fill=False, color="0.4", lw=1.5))
        ax.annotate(str(b.get("id", b.get("rb_id", ""))), (b["x"], b["y"]), ha="center", va="center", fontsize=8,
                    color="0.3")
    # beams
    lines = []
    for q in robots:
        mine = [r for r in recent if r["robot_id"] == q]
        errs = []
        for r in mine:
            o = r["pose"]
            if r.get("detections"):
                p = r["detections"][0]["position"]
                ax.plot([o[0], p[0]], [o[1], p[1]], color=colour[q], lw=1, alpha=0.55)
                ax.plot(p[0], p[1], "o", color=colour[q], ms=4)
                if r.get("true_horiz_m") is not None:
                    errs.append(r["detections"][0]["distance"] - r["true_horiz_m"])
            else:
                e = (o[0] + 0.4 * math.cos(o[2]), o[1] + 0.4 * math.sin(o[2]))
                ax.plot([o[0], e[0]], [o[1], e[1]], "--", color=colour[q], lw=1)
                ax.plot(e[0], e[1], "x", color=colour[q], ms=6)
        last = next((r for r in reversed(rows) if r["robot_id"] == q and r["t"] <= now), None)
        if last is not None and last.get("body"):
            x, y, yaw = last["body"]
            tri = np.array([[0.18, 0], [-0.12, 0.1], [-0.12, -0.1]])
            c, s = math.cos(yaw), math.sin(yaw)
            pts = tri @ np.array([[c, s], [-s, c]]) + [x, y]
            ax.add_patch(Polygon(pts, color=colour[q], alpha=0.8))
            o = last["pose"]
            ax.plot([o[0], o[0] + 0.5 * math.cos(o[2])], [o[1], o[1] + 0.5 * math.sin(o[2])], color=colour[q], lw=0.8)
            ax.annotate(q, (x, y - 0.22), ha="center", fontsize=9, color=colour[q], weight="bold")
        med = f"{np.median(errs) * 100:+.1f} cm" if errs else "-"
        info = ""
        if last is not None and last.get("yaw_offset_deg") is not None:
            info = f" (IR slot {last.get('ir_slot')}, aim offset {last['yaw_offset_deg']:+.1f} deg)"
        lines.append(f"{q}: {len(mine)} readings, {sum(1 for r in mine if not r.get('detections'))} invalid, "
                     f"median range error {med}{info}")
    # fusion
    title = f"last {window_s:.0f} s"
    if fsr is not None:
        from demo.methods import judge
        tb = [[b["x"], b["y"]] for b in boxes]
        for key, mk, col, lab in (("mht", "o", "red", "MHT+free space"), ("gnn", "s", "purple", "GNN+free space")):
            e = np.array(fsr["res"]["pre+post"].get(key, [])).reshape(-1, 2)
            if key not in fsr["res"]["pre+post"]:
                continue
            v = judge(e.tolist(), tb)["verdict"]
            ax.plot(e[:, 0], e[:, 1], mk, mfc="none", color=col, ms=16, mew=1.8, label=f"{lab}: {v}")
            title += f" | {lab} {len(e)}"
        title += f" | {len(fans)} fan beams, {len(fsr['dropped'])} detections dropped"
    if demo:
        res = demo.get("results", {})
        for key, mk, col, lab in (("mht_sensor", "+", "red", "MHT"), ("gnn", "x", "purple", "GNN")):
            r = res.get(key)
            if r and r.get("boxes") is not None:
                bx = np.array(r["boxes"]).reshape(-1, 2)
                if len(bx):
                    ax.plot(bx[:, 0], bx[:, 1], mk, color=col, ms=14, mew=2.5, label=f"{lab}: {r.get('verdict')}")
                else:
                    ax.plot([], [], mk, color=col, label=f"{lab}: no box")
                title += f" | {lab} {len(bx)} box(es), {r.get('verdict')}"
    else:
        title += " | demo page not reachable"
    if demo or fsr is not None:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.text(0.01, 0.01, "\n".join(lines), transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"))
    xs = [b["x"] for b in boxes] + [r["body"][0] for r in recent if r.get("body")]
    ys = [b["y"] for b in boxes] + [r["body"][1] for r in recent if r.get("body")]
    if xs:
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-dir", default="data/live_ir/")
    ap.add_argument("--demo", default="http://localhost:8000/state", help="'' = do not ask the demo page")
    ap.add_argument("--window-s", type=float, default=20.0, help="readings of the last seconds")
    ap.add_argument("--regions", default="config/optitrack_regions.json")
    ap.add_argument("--png", default=None, help="write one picture here and exit")
    ap.add_argument("--freespace", action="store_true",
                    help="also the free space of the beams and the fusion with it (computed here, on the log)")
    ap.add_argument("--sensor-calib", default="config/sensor_calib_ir.json")
    ap.add_argument("--sensor-model", default="config/sensor_model_ir.json")
    ap.add_argument("--gnn", default="results/plan_v2/M4/gnn/detection_full_full_h2", help="'' = no graph network")
    ap.add_argument("--at", type=float, default=None,
                    help="with --png: time of the picture, seconds from the start of the log (negative: from its end)")
    a = ap.parse_args()
    import matplotlib
    if a.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arena = None
    try:
        from vf.fusion.sensor_model import arena_from_regions
        arena = arena_from_regions(a.regions)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 8))
    fusion = None
    if a.freespace:
        import types
        from tools.ir_freespace import Fusion
        fusion = Fusion(types.SimpleNamespace(sensor_calib=a.sensor_calib, sensor_model=a.sensor_model,
                                              regions=a.regions, gnn=a.gnn, no_gnn=not a.gnn))

    def frame(_=None):
        rows = read_log(a.live_dir)
        if a.png and a.at is not None and rows:
            now = (rows[-1]["t"] + a.at) if a.at < 0 else (rows[0]["t"] + a.at)
            truth, demo = None, None
        else:
            now = time.time() if not a.png else (rows[-1]["t"] if rows else time.time())
            b = read_json(os.path.join(a.live_dir, "boxes.json"))
            truth = None if not b else [dict(id=x.get("rb_id"), x=x["x"], y=x["y"]) for x in b.get("boxes", [])]
            demo = demo_results(a.demo)
        draw(ax, rows, now, a.window_s, truth, demo, arena, fusion)

    if a.png:
        frame()
        fig.savefig(a.png, dpi=130, bbox_inches="tight")
        print(f"written {a.png}")
        return
    from matplotlib.animation import FuncAnimation
    frame.anim = FuncAnimation(fig, frame, interval=1000, cache_frame_data=False)   # kept alive while shown
    plt.show()


if __name__ == "__main__":
    main()
