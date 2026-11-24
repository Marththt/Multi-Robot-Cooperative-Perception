"""Simulation of the fleet: a captain and its followers scan the arena, then say how many targets there are and where.

Every robot: holonomic chassis (the RoboMaster EP has mecanum wheels) driven by the controller of the captures
(vf/control.goto.compute_command, same gains, v_max 0.3 m/s) towards the point its brain goal gives, waiting when
another robot is close and in the way (goto.blocked); camera frames at --fps (fleet/sensing.look). The brain
(fleet/brain.py) gets the poses and the frames and gives new goals every --tick-s. A move that would touch an
obstacle, a target, the border or another robot is refused and counted as a contact (should stay 0).

Sensors (--sensors): "gimbal" (default) = camera with YOLO only + the ToF on the gimbal, which fans for the map
and is aimed at the objects to range them (gimbal_step; fleet/brain.py); the report is the captain's online
object list (fleet/fusion.Tracker), the batch fusion of all the detections at the end is given as a check
(batch_*). "camera" = the first version (the camera also sees free space, batch fusion only).

At the end (coverage goal reached and nothing tentative, nothing left to explore, or --t-max-s): the objects,
scored against the truth.

    python -m fleet.sim --mode coverage --seed 0 --live         # watch it in real time (--speed 3: faster)
    python -m fleet.sim --mode formation --seed 0 --gif
    python -m fleet.sim --mode coverage --seed 0 --gif
    python -m fleet.sim --mode both --seeds 0-9                 # comparison table, no pictures
    python -m fleet.sim --mode coverage --seeds 0-9 --no-aim    # ablations: --no-aim, --no-revisit, --captain lanes
    python -m fleet.sim --sensors camera --mode both --seeds 0-9    # the first version
    python -m fleet.sim --world my_scene.json --robots 3         # a scene from a file (fleet/world.World.save)

Output: results/fleet_sim/<mode>_s<seed>/ summary.json, world.json, detections.jsonl, map.png (anim.gif).
"""

import argparse
import json
import math
import os
import time

import numpy as np

from fleet import fusion
from fleet.brain import Brain, BrainParams
from fleet.sensing import CameraModel, GimbalIR, ir_beam, look, proximity
from fleet.world import ROBOT_R_M, Grid, World, lab_arena
from vf.control.goto import compute_command, wrap_pi

V_MAX = 0.3
STOP_M = 2 * ROBOT_R_M + 0.08     # another robot this close, within IN_WAY_DEG of where we go: wait
IN_WAY_DEG = 60.0


def in_the_way(pose, target, others, stop_m=STOP_M, cone_deg=IN_WAY_DEG):
    """As vf/control/goto.blocked, but only a robot in a cone ahead stops us (goto.blocked: any robot in the half
    plane ahead, so two robots side by side at the stop distance never move again)."""
    ex, ey = target[0] - pose[0], target[1] - pose[1]
    ne = math.hypot(ex, ey)
    if ne < 1e-6:
        return False
    cos_c = math.cos(math.radians(cone_deg))
    for ox, oy in others:
        dx, dy = ox - pose[0], oy - pose[1]
        d = math.hypot(dx, dy)
        if d < stop_m and (dx * ex + dy * ey) > cos_c * d * ne:
            return True
    return False


def gimbal_step(a, ir, gim, brain, grid, true_ids, pose, poses, ids_r, rng, t, dt):
    """Every robot's gimbal for one time step: fan, or turn to the object of its task, settle, take n_aim IR
    readings and report their median; and the IR reading of this step (20 Hz = dt 0.05 s) for the map.
    Returns how many aimed measurements were reported."""
    reported = 0
    for r in ids_r:
        st = gim[r]
        x, y, yaw = pose[r]
        aiming = False
        if st["task"] is not None:
            k = brain.tracker.get(st["task"])
            if k is None or k.state not in ("tentative", "confirmed"):
                st["task"] = None
            else:
                tx, ty = k.xy
                want = wrap_pi(math.atan2(ty - y, tx - x) - yaw)
                err = wrap_pi(want - st["gy"])
                step = math.radians(ir.aim_dps) * dt
                st["gy"] = wrap_pi(st["gy"] + max(-step, min(step, err)))
                aiming = abs(err) < math.radians(1.0)
                st["settle"] = st["settle"] + dt if aiming else 0.0
        if st["task"] is None:
            lim = math.radians(ir.fan_deg)
            st["gy"] += st["dir"] * math.radians(ir.fan_dps) * dt
            if abs(st["gy"]) > lim:
                st["gy"], st["dir"] = math.copysign(lim, st["gy"]), -st["dir"]
        h = wrap_pi(yaw + st["gy"])
        o = (x + ir.arm_m * math.cos(h), y + ir.arm_m * math.sin(h))
        others = [pose[q][:2] for q in ids_r if q != r]
        d, _ = ir_beam(grid, true_ids, o, h, ir.r_max_m, others, rng, ir)
        brain.add_ir(r, o, h, d, ir.r_max_m, poses)
        if st["task"] is not None and aiming and st["settle"] >= ir.settle_s:
            st["samples"].append(d)
            if len(st["samples"]) >= ir.n_aim:
                vals = [v for v in st["samples"] if v is not None]
                dm = float(np.median(vals)) if len(vals) * 2 > len(st["samples"]) else None
                brain.ir_measure(r, st["task"], o, h, dm, ir)
                reported += 1
                st.update(task=None, samples=[], settle=0.0)
    return reported


def start_poses(arena, n, spacing=0.7):
    x0, x1, y0, y1 = arena
    along_x = (x1 - x0) >= (y1 - y0)
    out = []
    for k in range(n):
        o = (k - (n - 1) / 2) * spacing
        if along_x:
            out.append((x0 + 0.45, (y0 + y1) / 2 + o, 0.0))
        else:
            out.append(((x0 + x1) / 2 + o, y0 + 0.45, math.pi / 2))
    return out


def run(a, mode, seed, pictures=True):
    rng = np.random.default_rng(seed)
    arena = tuple(a.arena) if a.arena else lab_arena()
    ids_r = [f"R{k}" for k in range(a.robots)]
    starts = start_poses(arena, a.robots)
    order = sorted(range(a.robots), key=lambda k: abs(k - (a.robots - 1) / 2))   # captain in the middle
    captain = ids_r[order[0]]
    if a.world:
        world = World.load(a.world)
        arena = world.arena
    else:
        world = World.random(rng, arena, a.targets, a.obstacles, keep_free=[s[:2] for s in starts])
    cam = CameraModel(r_max_m=a.cam_range_m, hfov_deg=a.hfov_deg, fp_rate=a.fp_rate, p_d_near=a.p_d,
                      p_d_far=a.p_d * 2 / 3, sd_range_m=(0.02, a.sd_range_rel), sd_bearing_deg=a.sd_bearing_deg)
    p = BrainParams(mode=mode, coverage_goal=a.coverage_goal, sensors=a.sensors, captain_plan=a.captain,
                    revisit_m=0.0 if a.no_revisit else BrainParams.revisit_m)
    ir = GimbalIR(fan_dps=a.fan_dps, fan_deg=a.fan_deg, r_max_m=a.ir_range_m)
    gim = {r: dict(gy=0.0, dir=1.0, task=None, settle=0.0, samples=[]) for r in ids_r}
    n_aims = 0
    brain = Brain(arena, ids_r, captain, cam, p)
    grid = Grid(arena, p.fine_res)
    true_ids = world.ids(grid)
    pose = {r: list(s) for r, s in zip(ids_r, starts)}
    contacts = {r: 0 for r in ids_r}
    touching = {r: False for r in ids_r}
    contact_log = []
    dist = {r: 0.0 for r in ids_r}
    traj = {r: [tuple(pose[r])] for r in ids_r}
    snaps = []
    dt, t, next_look, next_tick, next_snap = a.dt, 0.0, 0.0, 0.0, 0.0
    frames = 0
    wall = time.time()
    view, next_view, next_est, est_live = None, 0.0, 0.0, []
    if a.live:
        from fleet.live import LiveView
        view = LiveView(world, grid, cam, ids_r, captain, mode, show_truth=not a.hide_truth)
    while t < a.t_max_s:
        if view is not None and t >= next_view:
            if not view.alive():
                break
            if a.sensors == "gimbal":
                est_live = [tuple(k.xy) for k in brain.tracker.confirmed()]
            elif t >= next_est:
                est_live = [o["xy"] for o in fusion.objects(brain.detections, brain.looks, brain.fine,
                                                            cam.sd_range_m, cam.sd_bearing_deg) if o["kept"]]
                next_est += 2.0
            view.update(t, pose, brain, est_live, gim)
            next_view += 0.1 * a.speed
            lag = wall + t / a.speed - time.time()
            if lag > 0:
                time.sleep(lag)
        poses_now = {r: tuple(pose[r]) for r in ids_r}
        if a.sensors == "gimbal":
            n_aims += gimbal_step(a, ir, gim, brain, grid, true_ids, pose, poses_now, ids_r, rng, t, dt)
            if int(t / 0.25) != int((t + dt) / 0.25):
                for r in ids_r:
                    brain.add_footprint(pose[r])
        if t >= next_look:
            for r in ids_r:
                others = [pose[q][:2] for q in ids_r if q != r]
                if a.sensors == "gimbal":           # the camera is on the gimbal: YOLO only
                    x, y, yaw = pose[r]
                    fr = look(cam, world, grid, true_ids, r, (x, y, wrap_pi(yaw + gim[r]["gy"])), others, rng, t,
                              depth=False)
                    brain.camera_frame(fr, poses_now)
                else:
                    fr = look(cam, world, grid, true_ids, r, tuple(pose[r]), others, rng, t)
                    brain.add_frame(fr)
                    brain.add_proximity(*proximity(grid, true_ids, pose[r], cam.prox_r_m))
                frames += 1
            next_look += 1.0 / a.fps
        if t >= next_tick:
            brain.tick(t, poses_now)
            next_tick += a.tick_s
            if a.sensors == "gimbal" and not a.no_aim:
                for r, tid in brain.aim_tasks(poses_now, [r for r in ids_r if gim[r]["task"] is None]).items():
                    gim[r].update(task=tid, settle=0.0, samples=[])
            if a.sensors == "gimbal":
                if brain.finished():
                    break
            elif brain.done or brain.coverage() >= p.coverage_goal:
                break
        if pictures and a.gif and t >= next_snap:
            snaps.append(dict(t=t, pose={r: tuple(pose[r]) for r in ids_r},
                              goals={r: brain.goals[r]["final"] for r in brain.goals},
                              views=brain.views.copy(), phase=brain.phase))
            next_snap += a.gif_every_s
        for r in ids_r:
            g = brain.goals.get(r)
            if g is None:
                continue
            vx, vy, wz, d, _ = compute_command(tuple(pose[r]), g["target"], g["yaw"], g["v_max"] or V_MAX)
            if d < 0.03:
                vx = vy = 0.0
            others = [pose[q][:2] for q in ids_r if q != r]
            if in_the_way(pose[r], g["target"], others, STOP_M):
                vx = vy = 0.0
            x, y, yaw = pose[r]
            c, s = math.cos(yaw), math.sin(yaw)
            nx, ny = x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt
            hit = world.collides(nx, ny) or (
                "robot" if any(math.hypot(nx - o[0], ny - o[1]) < 2 * ROBOT_R_M and
                               math.hypot(nx - o[0], ny - o[1]) < math.hypot(x - o[0], y - o[1]) for o in others)
                else None)
            if hit:
                if not touching[r]:
                    contacts[r] += 1
                    contact_log.append(dict(t=round(t, 2), robot=r, what=hit, x=round(x, 2), y=round(y, 2)))
                touching[r] = True
                nx, ny = x, y
            else:
                touching[r] = False
            dist[r] += math.hypot(nx - x, ny - y)
            pose[r] = [nx, ny, wrap_pi(yaw + wz * dt)]
        t += dt
        if int(t / 0.25) != int((t - dt) / 0.25):
            for r in ids_r:
                traj[r].append(tuple(pose[r]))

    objs = fusion.objects(brain.detections, brain.looks, brain.fine, cam.sd_range_m, cam.sd_bearing_deg)
    batch = [o["xy"] for o in objs if o["kept"]]
    online_extra = {}
    if a.sensors == "gimbal":                       # the report is the online object list; the batch is a check
        tr = brain.tracker
        objs = [dict(xy=tuple(float(v) for v in k.xy), kept=k.state == "confirmed", state=k.state, n_det=k.n_cam,
                     n_views=k.looks, ratio=k.ratio, n_ir=k.n_ir, ir_miss=k.ir_miss, robots=sorted(k.robots))
                for k in tr.tracks if k.state != "merged"]
        est = [o["xy"] for o in objs if o["kept"]]
        sb = fusion.score(batch, world.targets)
        online_extra = dict(batch_n_est=len(batch), batch_tp=sb["tp"], batch_fp=sb["fp"], batch_fn=sb["fn"],
                            batch_ospa_m=sb["ospa_m"], tentative_left=len(tr.tentative()), aims=n_aims,
                            ir_ranged=sum(1 for k in tr.confirmed() if k.n_ir))
    else:
        est = batch
    sc = fusion.score(est, world.targets)
    phi = (1.0 - cam.p_d_near) ** brain.views
    true_free = true_ids == 0
    summary = dict(mode=mode, seed=seed, robots=a.robots, captain=captain, t_s=round(t, 1),
                   wall_s=round(time.time() - wall, 1), phase=brain.phase, done=brain.done,
                   coverage_true=float(np.mean(phi[true_free] <= p.explored_phi)),
                   coverage_brain=brain.coverage(), frames=frames, detections=len(brain.detections),
                   targets_ever_detected=len({d["true"] for d in brain.detections if d["true"] is not None}),
                   contacts=sum(contacts.values()), contact_log=contact_log, path_m={r: round(v, 2) for r, v in dist.items()},
                   **{k: v for k, v in sc.items() if k != "matches"},
                   **online_extra, estimates=est, clusters=objs, truth=world.targets.tolist())
    if view is not None and view.alive():
        view.update(t, pose, brain, est, gim)
        print("simulation over: close the window to go on", flush=True)
        view.hold()
    out = os.path.join(a.out, f"{mode}_s{seed}")
    if pictures:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        world.save(os.path.join(out, "world.json"))
        with open(os.path.join(out, "detections.jsonl"), "w") as f:
            for d in brain.detections:
                f.write(json.dumps(dict(d, xy=list(d["xy"]), lens=list(d["lens"]))) + "\n")
        draw(os.path.join(out, "map.png"), world, grid, phi, traj, objs, summary, brain)
        if a.gif and snaps:
            animate(os.path.join(out, "anim.gif"), world, grid, cam, snaps, traj, captain)
        summary["out"] = out
    return summary


# ---------------------------------------------------------------------- pictures
def _scene(ax, world):
    import matplotlib.patches as mp
    for (x0, x1, y0, y1) in world.obstacles:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.2"))
    ax.plot(world.targets[:, 0], world.targets[:, 1], "s", ms=6, mfc="none", mec="tab:green", mew=1.5,
            label="target (truth)")
    a = world.arena
    ax.set_xlim(a[0], a[1])
    ax.set_ylim(a[2], a[3])
    ax.set_aspect("equal")


def draw(path, world, grid, phi, traj, objs, s, brain):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7.2))
    a = world.arena
    ax.imshow((1 - phi).T, origin="lower", extent=(a[0], a[1], a[2], a[3]), cmap="Blues", vmin=0, vmax=1.6,
              interpolation="nearest")
    _scene(ax, world)
    cols = plt.cm.tab10.colors
    for k, (r, tr) in enumerate(traj.items()):
        tr = np.array(tr)
        ax.plot(tr[:, 0], tr[:, 1], "-", color=cols[(k + 1) % 10], lw=1.3,
                label=f"{r}{' (captain)' if r == s['captain'] else ''}")
        ax.plot(*tr[-1, :2], "o", color=cols[(k + 1) % 10], ms=9)
    rej = [o["xy"] for o in objs if not o["kept"]]
    if rej:
        rej = np.array(rej)
        ax.plot(rej[:, 0], rej[:, 1], ".", color="0.5", ms=5, label="rejected cluster")
    est = [o["xy"] for o in objs if o["kept"]]
    if est:
        est = np.array(est)
        ax.plot(est[:, 0], est[:, 1], "x", color="tab:red", ms=8, mew=2, label="estimated target")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax.set_title(f"{s['mode']}, seed {s['seed']}: {s['n_est']} targets found of {s['n_true']} "
                 f"(TP {s['tp']}, FP {s['fp']}, FN {s['fn']}), coverage {100 * s['coverage_true']:.0f}% "
                 f"in {s['t_s']:.0f} s, contacts {s['contacts']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate(path, world, grid, cam, snaps, traj, captain):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(6, 5.4))
    a = world.arena
    im = ax.imshow(np.zeros(grid.shape).T, origin="lower", extent=(a[0], a[1], a[2], a[3]), cmap="Blues",
                   vmin=0, vmax=1.6, interpolation="nearest")
    _scene(ax, world)
    rids = list(snaps[0]["pose"])
    cols = plt.cm.tab10.colors
    dots = [ax.plot([], [], "o", color=cols[(k + 1) % 10], ms=8)[0] for k in range(len(rids))]
    heads = [ax.plot([], [], "-", color=cols[(k + 1) % 10], lw=2)[0] for k in range(len(rids))]
    goals = [ax.plot([], [], "+", color=cols[(k + 1) % 10], ms=10)[0] for k in range(len(rids))]
    title = ax.set_title("")

    def upd(i):
        s = snaps[i]
        im.set_data((1 - (1 - cam.p_d_near) ** s["views"]).T)
        for k, r in enumerate(rids):
            x, y, yaw = s["pose"][r]
            dots[k].set_data([x], [y])
            heads[k].set_data([x, x + 0.35 * math.cos(yaw)], [y, y + 0.35 * math.sin(yaw)])
            g = s["goals"].get(r)
            goals[k].set_data(([g[0]], [g[1]]) if g else ([], []))
        title.set_text(f"t = {s['t']:.0f} s   {s['phase']}   captain {captain}")
        return [im, *dots, *heads, *goals, title]

    FuncAnimation(fig, upd, frames=len(snaps), blit=False).save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


# ---------------------------------------------------------------------- CLI
def seeds_arg(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["formation", "coverage", "both"], default="formation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=seeds_arg, default=None, help="e.g. 0-9: many scenes, a table at the end")
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--world", default=None, help="scene from a JSON file (World.save)")
    ap.add_argument("--arena", type=float, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1"),
                    help="default: the lab arena (config/optitrack_regions.json)")
    ap.add_argument("--cam-range-m", type=float, default=2.0)
    ap.add_argument("--hfov-deg", type=float, default=88.0)
    ap.add_argument("--sensors", choices=["gimbal", "camera"], default="gimbal",
                    help="gimbal: camera (YOLO only) + IR on the gimbal, map from the IR, online fusion with IR "
                         "aiming; camera: the first version (the camera also sees free space, batch fusion)")
    ap.add_argument("--captain", choices=["frontier", "lanes"], default="frontier")
    ap.add_argument("--no-aim", action="store_true", help="gimbal: never aim the IR at objects (fan only): "
                                                           "no active perception")
    ap.add_argument("--no-revisit", action="store_true", help="gimbal: tentative objects do not draw the fleet back")
    ap.add_argument("--fan-dps", type=float, default=30.0, help="gimbal fan speed [deg/s]")
    ap.add_argument("--fan-deg", type=float, default=40.0, help="gimbal fan half width [deg]")
    ap.add_argument("--ir-range-m", type=float, default=2.5)
    ap.add_argument("--fp-rate", type=float, default=None, help="false detections per frame (default 0.1 gimbal, "
                                                                "0.03 camera)")
    ap.add_argument("--p-d", type=float, default=None, help="detection probability up to 70%% of the range "
                                                            "(default 0.85 gimbal, 0.9 camera)")
    ap.add_argument("--sd-range-rel", type=float, default=None,
                    help="camera range error sd = 2 cm + this x range (default 0.15 gimbal: YOLO monocular; "
                         "0.03 camera)")
    ap.add_argument("--sd-bearing-deg", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=2.0, help="camera frames per second per robot")
    ap.add_argument("--tick-s", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t-max-s", type=float, default=300.0)
    ap.add_argument("--coverage-goal", type=float, default=0.98)
    ap.add_argument("--live", action="store_true", help="watch it in a window, in real time (see --speed)")
    ap.add_argument("--speed", type=float, default=1.0, help="--live: times real time (2 = twice as fast)")
    ap.add_argument("--hide-truth", action="store_true", help="--live: do not draw the true targets")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-every-s", type=float, default=1.0)
    ap.add_argument("--out", default="results/fleet_sim")
    a = ap.parse_args()
    g = a.sensors == "gimbal"
    a.fp_rate = a.fp_rate if a.fp_rate is not None else (0.1 if g else 0.03)
    a.p_d = a.p_d if a.p_d is not None else (0.85 if g else 0.9)
    a.sd_range_rel = a.sd_range_rel if a.sd_range_rel is not None else (0.15 if g else 0.03)
    if not a.live:
        import matplotlib
        matplotlib.use("Agg")
    modes = ["formation", "coverage"] if a.mode == "both" else [a.mode]
    seeds = a.seeds or [a.seed]
    rows = []
    for seed in seeds:
        for m in modes:
            s = run(a, m, seed, pictures=len(seeds) == 1)
            rows.append(s)
            print(f"{m:9s} seed {seed:3d}: {s['n_est']:2d} found / {s['n_true']} true  TP {s['tp']:2d} FP {s['fp']} "
                  f"FN {s['fn']}  err {100 * (s['mean_err_m'] or 0):.1f} cm  OSPA {100 * s['ospa_m']:.1f} cm  "
                  f"coverage {100 * s['coverage_true']:.1f}%  t {s['t_s']:.0f} s  contacts {s['contacts']}  "
                  f"({s['wall_s']:.0f} s)" + (f"  aims {s['aims']}, batch {s['batch_n_est']} (FP {s['batch_fp']} "
                                               f"FN {s['batch_fn']})" if "aims" in s else "")
                  + (f"  -> {s['out']}" if "out" in s else ""), flush=True)
    if len(rows) > 1:
        print("\nmean per mode:")
        for m in modes:
            r = [s for s in rows if s["mode"] == m]
            f = lambda k: np.mean([s[k] for s in r])
            print(f"  {m:9s} exact count {np.mean([s['count_err'] == 0 for s in r]) * 100:5.1f}%  "
                  f"|count err| {np.mean([abs(s['count_err']) for s in r]):.2f}  F1 {f('f1'):.3f}  "
                  f"OSPA {100 * f('ospa_m'):.1f} cm  coverage {100 * f('coverage_true'):.1f}%  "
                  f"time {f('t_s'):.0f} s  path {np.mean([sum(s['path_m'].values()) for s in r]):.1f} m  "
                  f"contacts {f('contacts'):.2f}")
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "compare.json"), "w") as fh:
            json.dump([{k: v for k, v in s.items() if k not in ("clusters",)} for s in rows], fh, indent=1)


if __name__ == "__main__":
    main()
