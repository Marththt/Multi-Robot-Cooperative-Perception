"""The brain of the fleet: the shared map and the goal of every robot, a few times per second.

It runs in one place (the captain's computer) and sees only what the robots report: their poses (OptiTrack) and
their camera frames (fleet/sensing.Frame). It never sees the true scene.

Sensors "gimbal" (the default; the robots have only the camera and the ToF on the gimbal):
  - the MAP comes from the gimbal IR alone (add_ir): every reading frees the cells along the beam and marks the
    end cell occupied (L_HIT_IR, so that beams grazing an edge do not erase it); a beam ending on another robot
    marks nothing there. Where robots have stood is free (add_footprint). The arena is not known in advance.
  - the robots move only where the IR has seen free space: unknown cells within unknown_m of a robot centre are
    blocked like obstacles (no driving into an unseen box);
  - the CAMERA gives YOLO detections only; where it looked is worked out on the brain's own map (camera_frame),
    for the coverage (phi) and for the negative evidence of the online fusion;
  - ONLINE FUSION (fleet/fusion.Tracker): every frame updates the object list (tentative / confirmed /
    deleted); an object is confirmed only if the IR agrees (an aimed IR range on it, or a box-sized blob of the
    IR map where it is: occ_near);
  - ACTIVE PERCEPTION: an idle gimbal is sent to aim the IR at the nearest object not yet ranged (aim_tasks), and
    tentative objects raise phi around them (revisit_m), so the coverage brings the fleet back to look again;
  - the captain explores by frontiers (captain_plan "frontier": the largest unexplored blob, a viewpoint it can
    reach on known free space, or the nearest frontier cell); the mission ends when the coverage goal is reached
    and nothing is tentative (finished).

Map (fine grid, 5 cm)
  occupancy   log-odds: + on a cell where a camera ray stopped (or the proximity sensor found something), - on a
              cell a ray went through. Occupied cells are closed (binary_fill_holes): the inside of an obstacle is
              never seen but is not free space.
  views       how many frames saw the cell free, between r_min and r_max of the camera
  miss        phi = (1 - p_d) ** views: the probability that a target standing there would still be undetected.
              It is the density of the coverage problem (what is left to look at), and 1 - phi is the coverage.

Plans (coarse grid, 10 cm): occupied cells (and the other robots) inflated by the robot radius + a margin, A*
8-connected, the robot follows a point lookahead_m ahead on its path.

The captain drives lanes across the arena (boustrophedon, along the longer side, lanes spaced so that the fleet's
swath covers the arena), then goes to the largest part still unexplored (mop-up) until the coverage goal or the
time limit. It slows down when a follower is left behind: the fleet stays together.

The followers, two strategies (--mode):
  formation   formation control: every follower holds a slot of a line abreast centred on the captain, offset
              across the lanes (a fixed direction, so that turning at the end of a lane never makes them cross) and
              a little behind, the outer cameras turned outwards (splay_deg).
  coverage    sensor coverage (Cortes et al. 2004, Lloyd's algorithm): the cells within leash_m of the captain are
              split among the robots by Voronoi (nearest robot); every follower goes to the centroid of its cell
              weighted by phi, facing where phi is (phi-weighted mean direction). As the cells are seen, phi drops
              and the followers move on to what is left; the captain drags the whole partition along the arena.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from fleet.fusion import Tracker
from fleet.sensing import _march
from fleet.world import ROBOT_R_M, Grid
from vf.control.goto import wrap_pi

L_HIT, L_FREE, L_MAX = 0.9, -0.4, 4.0
L_HIT_IR = 1.7                 # one IR echo outweighs a few beams grazing the cell (the edge of an obstacle)


@dataclass
class BrainParams:
    mode: str = "formation"           # formation | coverage
    fine_res: float = 0.05
    coarse_res: float = 0.10
    inflate_m: float = ROBOT_R_M + 0.10
    robot_clear_m: float = 2 * ROBOT_R_M + 0.10
    lookahead_m: float = 0.35
    captain_v: float = 0.18           # m/s, the followers get v_max of the controller
    captain_slow_v: float = 0.04      # when a follower is behind
    behind_m: float = 0.9             # a follower farther than this from its goal is "behind"
    spacing_m: float = 0.8            # formation: distance between neighbours across the lanes
    back_m: float = 0.25              # formation: followers this much behind per slot
    splay_deg: float = 25.0           # formation: outer cameras turned outwards
    leash_m: float = 1.6              # coverage: cells within this of the captain are shared out
    lane_extra_m: float = 1.0         # lane spacing = swath of the fleet + this (what the cameras see beside it)
    margin_m: float = 0.40            # lanes stay this far from the border
    coverage_goal: float = 0.98
    explored_phi: float = 0.1         # a cell with phi below this is covered
    waypoint_tol_m: float = 0.25
    stuck_s: float = 20.0             # a captain goal not approached by 10 cm in this long is dropped
    ir_across_m: float = 0.0          # extra sd across the beam of an aimed IR reading (where it hit the face; lab: 0.04)
    cam_floor_m: float = 0.0          # floor of an object's camera position error (fusion.Track._cam; lab: 0.08)
    boxed_cells: int = 30             # captain reaches fewer coarse cells than this: it turns on the spot first
    sensors: str = "gimbal"           # gimbal: map from the gimbal IR, camera = YOLO only | camera: the camera
                                      # also gives free space (depth), the first version
    captain_plan: str = "frontier"    # frontier: next best viewpoint on the map | lanes: boustrophedon first
    unknown_m: float = 0.22           # gimbal: a robot centre stays this far from space never seen free
    aim_range_m: float = 2.2          # gimbal: objects the IR is aimed at, at most this far
    aim_cooldown_s: float = 6.0       # the same object is not aimed at again before this
    revisit_m: float = 0.3            # tentative objects make phi high within this: the fleet goes back to them
    box_max_m: float = 0.30           # an occupied blob wider than this is not a box


def downsample(a, f, how):
    """Coarse cells of f x f fine cells (padded at the far edges)."""
    nx, ny = -(-a.shape[0] // f), -(-a.shape[1] // f)
    pad = np.zeros((nx * f, ny * f), a.dtype)
    if how == "mean":
        pad[:] = 0
    pad[:a.shape[0], :a.shape[1]] = a
    b = pad.reshape(nx, f, ny, f)
    return b.any(axis=(1, 3)) if how == "any" else b.mean(axis=(1, 3))


def disk(r_cells):
    r = int(math.ceil(r_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r_cells * r_cells


def astar(blocked, start, goal):
    """8-connected A* on a boolean grid; start is allowed even if blocked. Returns a list of (ix, iy) or None."""
    nx, ny = blocked.shape
    if start == goal:
        return [start]
    g = {start: 0.0}
    came = {}
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
    openq = [(h(start), 0.0, start)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while openq:
        _, gc, c = heapq.heappop(openq)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return path[::-1]
        if gc > g.get(c, 1e18):
            continue
        for dx, dy, w in steps:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or blocked[n]:
                continue
            if dx and dy and (blocked[c[0] + dx, c[1]] or blocked[c[0], c[1] + dy]):
                continue                    # no corner cutting
            ng = gc + w
            if ng < g.get(n, 1e18):
                g[n], came[n] = ng, c
                heapq.heappush(openq, (ng + h(n), ng, n))
    return None


class Brain:
    def __init__(self, arena, robots, captain, cam, p=BrainParams()):
        self.arena, self.robots, self.captain, self.cam, self.p = arena, list(robots), captain, cam, p
        self.fine, self.coarse = Grid(arena, p.fine_res), Grid(arena, p.coarse_res)
        self.f = int(round(p.coarse_res / p.fine_res))
        self.logodds = np.zeros(self.fine.shape)
        self.views = np.zeros(self.fine.shape, int)
        self.looks = np.zeros(self.fine.shape, int)     # frames that reached the cell (free or stopped there)
        self.detections = []                # dict(robot, t, xy, ...) of every frame
        self.goals = {}                     # rid -> dict(target=(x, y), yaw, path=[(x, y)...], final=(x, y))
        self.followers = [r for r in self.robots if r != captain]
        self.phase = "lanes" if p.captain_plan == "lanes" else "explore"
        self.wp, self.wp_t = 0, None
        self.trail = np.zeros(self.fine.shape, bool)      # where robots have stood: free space for sure
        self.tracker = Tracker(cam.sd_range_m, cam.sd_bearing_deg, cam_floor_m=p.cam_floor_m)
        self.aimed = {}                                   # track id -> time of the last aim
        self.ir_last = {}                                 # rid -> (origin, end, hit) of the last IR reading
        self.lanes, self.across = None, None
        self.slots = None
        self.mop_blacklist = []
        self.done = False
        self.retries = 0                                  # blacklist resets while coverage is low
        self.spins, self.spin_pts = 0, []           # turns on the spot while boxed in by unknown space
        self.t = 0.0
        self.unreachable = 0
        self.best_d = float("inf")
        self.mop, self.mop_at = None, float("inf")

    # ------------------------------------------------------------------ map
    def add_frame(self, fr):
        g = self.fine
        lo = self.logodds.ravel()
        if fr.seen is not None and len(fr.seen):
            lo[fr.seen] = np.maximum(lo[fr.seen] + L_FREE, -L_MAX)
            d = np.hypot(g.P[fr.seen, 0] - fr.lens[0], g.P[fr.seen, 1] - fr.lens[1])
            inr = fr.seen[(d >= self.cam.r_min_m) & (d <= self.cam.r_max_m)]
            self.views.ravel()[inr] += 1
        if fr.hits is not None and len(fr.hits):
            lo[fr.hits] = np.minimum(lo[fr.hits] + L_HIT, L_MAX)
        reached = np.concatenate([a for a in (fr.seen, fr.hits) if a is not None and len(a)] or [np.zeros(0, int)])
        if len(reached):
            reached = np.unique(reached)
            d = np.hypot(g.P[reached, 0] - fr.lens[0], g.P[reached, 1] - fr.lens[1])
            self.looks.ravel()[reached[(d >= self.cam.r_min_m) & (d <= self.cam.r_max_m)]] += 1
        for d in fr.detections:
            self.detections.append(dict(robot=fr.robot, t=fr.t, lens=fr.lens, **d))

    def add_footprint(self, pose):
        g = self.fine
        m = np.hypot(g.X - pose[0], g.Y - pose[1]) <= ROBOT_R_M
        self.trail |= m
        self.logodds[m] = np.minimum(self.logodds[m], -L_MAX / 2)

    def add_ir(self, robot, origin, heading, d, r_max, poses):
        """One gimbal IR reading: the cells along the beam are free, the end cell occupied (unless the beam ended
        on another robot: the brain knows where the robots are)."""
        g = self.fine
        end_d = r_max if d is None else d
        hit = d is not None
        if hit:
            ex, ey = origin[0] + d * math.cos(heading), origin[1] + d * math.sin(heading)
            if any(math.hypot(ex - q[0], ey - q[1]) < ROBOT_R_M + 0.08 for rid, q in poses.items() if rid != robot):
                hit, end_d = False, d - 0.05
        s = np.arange(g.res / 2, max(end_d - g.res, 0.0), g.res / 2)
        px, py = origin[0] + np.cos(heading) * s, origin[1] + np.sin(heading) * s
        ok = (px >= g.x0) & (px < g.x1) & (py >= g.y0) & (py < g.y1)
        flat = np.unique(((px[ok] - g.x0) / g.res).astype(int) * g.ny + ((py[ok] - g.y0) / g.res).astype(int))
        lo = self.logodds.ravel()
        lo[flat] = np.maximum(lo[flat] + L_FREE, -L_MAX)
        end = (origin[0] + end_d * math.cos(heading), origin[1] + end_d * math.sin(heading))
        if hit and g.inside(*end):
            ix, iy = g.index(*end)
            self.logodds[ix, iy] = min(self.logodds[ix, iy] + L_HIT_IR, L_MAX)
        self.ir_last[robot] = (tuple(origin), end, hit)

    def camera_frame(self, fr, poses):
        """A YOLO frame (gimbal sensors): where it looked, from the brain's own map (rays stop on cells known to be
        occupied and on the other robots; unknown space counts as seen through), then the tracker."""
        g, cam = self.fine, self.cam
        half = math.radians(cam.hfov_deg) / 2
        angles = fr.lens[2] + np.linspace(-half, half, int(round(2 * half / math.radians(2.0))) + 1)
        occ = (self.logodds > 0.5).astype(int)
        others = [q[:2] for rid, q in poses.items() if rid != fr.robot]
        seen, hits = _march(fr.lens[:2], angles, cam.r_max_m, g, occ, others)
        looked = np.concatenate([seen, hits]).astype(int)
        if len(looked):
            d = np.hypot(g.P[looked, 0] - fr.lens[0], g.P[looked, 1] - fr.lens[1])
            looked = np.unique(looked[(d >= cam.r_min_m) & (d <= cam.r_max_m)])
            self.views.ravel()[looked] += 1
            self.looks.ravel()[looked] += 1
        mask = np.zeros(g.shape, bool)
        mask.ravel()[looked] = True

        def was_looked(xy):
            ix, iy = g.index(*xy)
            return bool(mask[max(0, ix - 1):ix + 2, max(0, iy - 1):iy + 2].any())

        dets = []
        for d in fr.detections:
            row = dict(robot=fr.robot, t=fr.t, lens=fr.lens, **d)
            self.detections.append(row)
            dets.append(row)
        self.tracker.camera(fr.robot, fr.t, dets, was_looked, self.occ_near)

    def occ_near(self, xy, r=0.12):
        """The IR map has a BOX-SIZED blob within r of xy: occupied cells there, in a connected blob no wider
        than box_max_m (a long blob is an obstacle or a wall, whatever YOLO says is in front of it). Returns the
        centre (x, y) of that blob (one blob confirms one object only: fusion.Tracker.states), or None."""
        g = self.fine
        if not g.inside(*xy):
            return False
        ix, iy = g.index(*xy)
        w = int(math.ceil(0.6 / g.res))
        x0, y0 = max(0, ix - w), max(0, iy - w)
        win = self.logodds[x0:ix + w + 1, y0:iy + w + 1] > 0.5
        if not win.any():
            return False
        lab, _ = ndimage.label(win, structure=np.ones((3, 3)))
        k = int(math.ceil(r / g.res))
        cx, cy = ix - x0, iy - y0
        near = lab[max(0, cx - k):cx + k + 1, max(0, cy - k):cy + k + 1]
        for c in np.unique(near[near > 0]):
            cells = np.argwhere(lab == c)
            ext = (cells.max(0) - cells.min(0) + 1) * g.res
            if ext.max() <= self.p.box_max_m:
                c = cells.mean(0) + (x0, y0)
                return (g.x0 + (c[0] + 0.5) * g.res, g.y0 + (c[1] + 0.5) * g.res)
        return None

    def ir_measure(self, robot, tid, origin, heading, d, ir):
        """An aimed IR reading on track tid: the box centre one apothem behind the face the beam hit."""
        if d is None:
            k = self.tracker.get(tid)
            if k is not None and k.state in ("tentative", "confirmed"):
                k.ir_miss += 1
                self.tracker.states(self.occ_near)
            return
        u = np.array([math.cos(heading), math.sin(heading)])
        z = np.asarray(origin, float) + (d + ir.apothem_m) * u
        face = np.asarray(origin, float) + d * u
        if not self.occ_near(tuple(face), r=0.08):      # the beam hit an obstacle, a wall, or nothing box-like
            k = self.tracker.get(tid)
            if k is not None and k.state in ("tentative", "confirmed"):
                k.ir_miss += 1
                self.tracker.states(self.occ_near)
            return
        # along the beam: the IR error; across: the aim error and where on the face the beam hit (up to half a
        # box off the centre line: the centre one apothem behind the hit point is right along the beam only)
        sa = ir.sd_m
        sc = math.hypot((d + ir.apothem_m) * math.radians(ir.aim_sd_deg), self.p.ir_across_m)
        R = np.outer(u, u) * sa * sa + np.outer([-u[1], u[0]], [-u[1], u[0]]) * sc * sc
        ok = self.tracker.ir(tid, z, R, self.occ_near)
        self.detections.append(dict(robot=robot, t=self.t, lens=(origin[0], origin[1], heading), xy=tuple(z),
                                    range=d + ir.apothem_m, bearing=0.0, true=None, ir=True, sd_along=sa,
                                    sd_across=sc, ir_ok=ok))

    def aim_tasks(self, poses, idle):
        """Gimbal IR tasks: every idle robot (gimbal fanning) gets the nearest live object it can aim at (within
        aim_range_m, in line of sight on the map, not yet ranged by the IR, not aimed at in the last
        aim_cooldown_s, not given to another robot). Returns {rid: track id}."""
        out, taken = {}, set()
        occ = self.logodds > 0.5
        for r in idle:
            q = poses[r]
            best, bd = None, self.p.aim_range_m
            for k in self.tracker.live():
                if k.n_ir > 0 or k.id in taken or self.t - self.aimed.get(k.id, -1e9) < self.p.aim_cooldown_s:
                    continue
                xy = k.xy
                d = math.hypot(xy[0] - q[0], xy[1] - q[1])
                if d < bd and self.fine_line(q[:2], xy, occ, skip_end_m=0.15):
                    best, bd = k, d
            if best is not None:
                out[r] = best.id
                taken.add(best.id)
                self.aimed[best.id] = self.t
        return out

    def fine_line(self, a, b, occ, skip_end_m=0.0):
        g = self.fine
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        for t in np.arange(0.3, max(d - skip_end_m, 0.3), g.res / 2):
            ix, iy = g.index(a[0] + (b[0] - a[0]) * t / d, a[1] + (b[1] - a[1]) * t / d)
            if occ[ix, iy]:
                return False
        return True

    def finished(self):
        """Coverage goal reached and nothing tentative left, or nothing left to explore."""
        return self.done or (self.coverage() >= self.p.coverage_goal and not self.tracker.tentative())

    def add_proximity(self, free, occ):
        lo = self.logodds.ravel()
        lo[free] = np.maximum(lo[free] + L_FREE, -L_MAX)
        lo[occ] = np.minimum(lo[occ] + L_HIT, L_MAX)

    def occupied(self):
        return ndimage.binary_fill_holes(self.logodds > 0.5)

    def phi(self):
        """Probability that a target in the cell is still undetected (0 on occupied cells)."""
        ph = (1.0 - self.cam.p_d_near) ** self.views
        ph[self.occupied()] = 0.0
        g = self.fine
        for k in self.tracker.tentative():           # an object still to confirm or reject: look there again
            xy = k.xy
            m = np.hypot(g.X - xy[0], g.Y - xy[1]) < self.p.revisit_m
            ph[m] = np.maximum(ph[m], 0.9)
        return ph

    def coverage(self):
        """Share of the cells not known to be occupied that are covered (the brain's own estimate)."""
        occ = self.occupied()
        ph = (1.0 - self.cam.p_d_near) ** self.views
        return float(np.mean(ph[~occ] <= self.p.explored_phi))

    # ------------------------------------------------------------------ planning grid
    def inflated(self):
        """Coarse cells a robot centre must not enter: occupied cells inflated by inflate_m on the fine grid, then
        a coarse cell is blocked when most of it is (no bloat from the coarser grid). Cached per tick."""
        if getattr(self, "_infl_t", None) != self.t:
            occ = self.occupied()
            if occ.any():
                occ = ndimage.binary_dilation(occ, disk(self.p.inflate_m / self.fine.res))
            c = self.coarse
            b = downsample(occ.astype(float), self.f, "mean")[:c.nx, :c.ny] > 0.5
            if self.p.sensors == "gimbal":               # only where the IR has seen free space (or a robot stood)
                unknown = (self.logodds >= 0) & ~self.trail
                unknown = ndimage.binary_dilation(unknown, disk(self.p.unknown_m / self.fine.res))
                b |= downsample(unknown.astype(float), self.f, "mean")[:c.nx, :c.ny] > 0.5
            m = int(math.ceil(self.p.inflate_m / c.res))
            b[:m, :] = b[-m:, :] = True
            b[:, :m] = b[:, -m:] = True
            self._infl, self._infl_t = b, self.t
        return self._infl

    def blocked(self, poses, me):
        c, p = self.coarse, self.p
        b = self.inflated().copy()
        for rid, q in poses.items():
            if rid == me or q is None:
                continue
            b |= np.hypot(c.X - q[0], c.Y - q[1]) < p.robot_clear_m
        return b

    def plan(self, pose, goal_xy, blocked):
        """A* path (points) from the robot to goal_xy, or to the reachable cell nearest to it. Returns (path,
        how far the end of the path is from goal_xy)."""
        c = self.coarse
        s = c.index(pose[0], pose[1])
        b = blocked.copy()
        b[s] = False
        lab, _ = ndimage.label(~b, structure=np.ones((3, 3)))
        reach = np.argwhere(lab == lab[s])
        gi = np.array(c.index(*goal_xy))
        k = int(np.argmin(((reach - gi) ** 2).sum(1)))
        g = (int(reach[k, 0]), int(reach[k, 1]))
        path = astar(b, s, g)
        if path is None:
            return [(pose[0], pose[1])], math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])
        pts = [c.centre(*q) for q in path]
        pts[0] = (pose[0], pose[1])
        if g == tuple(gi):
            pts[-1] = tuple(goal_xy)
        end = pts[-1]
        return pts, math.hypot(goal_xy[0] - end[0], goal_xy[1] - end[1])

    def lookahead(self, pose, path, blocked=None):
        """The farthest point of the path within lookahead_m that the robot reaches in a straight line without
        entering a blocked cell (no corner cutting next to an obstacle); at least the next point."""
        best = path[1] if len(path) > 1 else path[0]
        for q in path[1:]:
            if math.hypot(q[0] - pose[0], q[1] - pose[1]) > self.p.lookahead_m:
                break
            if blocked is not None and not self.clear_line(pose, q, blocked):
                break
            best = q
        return best

    def clear_line(self, a, b, blocked):
        c = self.coarse
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        s0 = c.index(a[0], a[1])
        for t in np.arange(0.0, d, c.res / 3):
            ix, iy = c.index(a[0] + (b[0] - a[0]) * t / d, a[1] + (b[1] - a[1]) * t / d)
            if (ix, iy) != s0 and blocked[ix, iy]:
                return False
        return True

    # ------------------------------------------------------------------ captain
    def make_lanes(self):
        x0, x1, y0, y1 = self.arena
        p, n = self.p, len(self.robots)
        swath = (n - 1) * p.spacing_m if p.mode == "formation" else 2 * p.leash_m * 0.8
        m = p.margin_m
        along_x = (x1 - x0) >= (y1 - y0)
        lo, hi = (y0, y1) if along_x else (x0, x1)
        a0, a1 = (x0 + m, x1 - m) if along_x else (y0 + m, y1 - m)
        c0, c1 = lo + m + swath / 2, hi - m - swath / 2
        if c1 <= c0:
            cs = [(lo + hi) / 2]
        else:
            k = max(1, int(math.ceil((c1 - c0) / (swath + p.lane_extra_m))) + 1)
            cs = list(np.linspace(c0, c1, k)) if k > 1 else [(c0 + c1) / 2]
        wps = []
        for i, cc in enumerate(cs):
            ends = (a0, a1) if i % 2 == 0 else (a1, a0)
            for e in ends:
                wps.append((e, cc) if along_x else (cc, e))
        self.across = np.array([0.0, 1.0]) if along_x else np.array([1.0, 0.0])
        return wps

    def start_lanes(self, poses):
        """Lanes starting from the end nearest the captain."""
        wps = self.make_lanes()
        cap = poses[self.captain]
        flip = [wps[i ^ 1] for i in range(len(wps))] if len(wps) % 2 == 0 else wps
        cands = [wps, wps[::-1], flip, flip[::-1]]
        self.lanes = min(cands, key=lambda w: math.hypot(w[0][0] - cap[0], w[0][1] - cap[1]))

    def mop_goal(self, poses):
        """A viewpoint for the largest unexplored blob: a cell the captain can reach, view_m from the blob's centre
        (inside the camera range), the nearest to the captain; and the direction of the blob. None when nothing is
        left (or only blobs already tried)."""
        c = self.coarse
        ph = downsample(self.phi(), self.f, "mean")[:c.nx, :c.ny]
        un = ph > 0.5
        for q in self.mop_blacklist:
            un[np.hypot(c.X - q[0], c.Y - q[1]) < 0.3] = False
        lab, n = ndimage.label(un)
        if n == 0:
            return None
        cap = poses[self.captain]
        b = self.inflated().copy()                 # the map only: the followers move away
        s = c.index(cap[0], cap[1])
        b[s] = False
        reach_lab, _ = ndimage.label(~b, structure=np.ones((3, 3)))
        reach = reach_lab == reach_lab[s]
        boxed = reach.sum() < self.p.boxed_cells
        if not boxed:
            self.spins = 0
        occ = downsample(self.occupied().astype(float), self.f, "mean")[:c.nx, :c.ny] > 0.25
        best, score = None, -1.0
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < 4:
                continue
            ctr = (float(c.X[m].mean()), float(c.Y[m].mean()))
            dc = np.hypot(c.X - ctr[0], c.Y - ctr[1])
            ring = reach & (dc >= 0.6) & (dc <= 0.6 * self.cam.r_max_m)
            if not ring.any():
                continue
            dcap = np.hypot(c.X - cap[0], c.Y - cap[1])
            dcap[~ring] = np.inf
            vp = None
            for flat in np.argsort(dcap, axis=None)[:200]:          # nearest viewpoint that sees the blob
                i = np.unravel_index(flat, dcap.shape)
                if not np.isfinite(dcap[i]):
                    break
                if self.clear_line(c.centre(*i), ctr, occ):
                    vp = c.centre(*i)
                    break
            if vp is None:
                continue
            sc = m.sum() / (1.0 + dcap[i])
            if sc > score:
                best, score = (vp, ctr), sc
        if best is None:
            # no viewpoint within camera range of any blob (e.g. at the start, little is known free): go to the
            # frontier, the reachable cell nearest to the largest blob, facing it
            R = np.argwhere(reach)
            for k in np.argsort([-(lab == k).sum() for k in range(1, n + 1)])[:3] + 1:
                B = np.argwhere(lab == k)
                if len(B) < 4 or not len(R):
                    continue
                d2 = ((R[:, None, :] - B[None, ::max(1, len(B) // 300), :]) ** 2).sum(-1)
                i, j = np.unravel_index(np.argmin(d2), d2.shape)
                vp = c.centre(*R[i])
                if math.hypot(vp[0] - cap[0], vp[1] - cap[1]) < 0.15 and len(self.mop_blacklist) and \
                        any(math.hypot(vp[0] - q[0], vp[1] - q[1]) < 0.3 for q in self.mop_blacklist):
                    continue
                ctr = (float(c.X[lab == k].mean()), float(c.Y[lab == k].mean()))
                bq = c.centre(*B[::max(1, len(B) // 300)][j])
                return vp, bq if math.hypot(bq[0] - vp[0], bq[1] - vp[1]) > 0.05 else ctr
        if best is None and boxed and self.spins < 4:
            # nothing reachable, but the captain is boxed in by space the IR has not seen yet (at the start, beside
            # the robot: the fan covers +-fan_deg ahead only): turn on the spot, a quarter at a time, before giving up
            self.spins += 1
            a = cap[2] + math.pi / 2
            q = (cap[0] + math.cos(a), cap[1] + math.sin(a))
            self.spin_pts.append(q)
            return (cap[0], cap[1]), q
        return best

    def captain_goal(self, poses):
        cap = poses[self.captain]
        p = self.p
        if self.lanes is None:
            self.start_lanes(poses)
        if self.wp_t is None:
            self.wp_t = self.t
        if self.phase == "lanes":
            if self.wp >= len(self.lanes):
                self.phase, self.mop = "mop", None
            else:
                w = self.lanes[self.wp]
                prev = self.lanes[self.wp - 1] if self.wp else (cap[0], cap[1])
                heading = math.atan2(w[1] - prev[1], w[0] - prev[0])
                dw = math.hypot(w[0] - cap[0], w[1] - cap[1])
                if dw < self.best_d - 0.1:
                    self.best_d, self.wp_t = dw, self.t
                if dw < p.waypoint_tol_m or self.t - self.wp_t > p.stuck_s:
                    self.wp += 1
                    self.wp_t, self.best_d = self.t, float("inf")
                    return self.captain_goal(poses)
                return w, heading
        if self.phase in ("mop", "explore"):
            if self.mop is None or self.t - self.wp_t > p.stuck_s or \
                    (math.hypot(self.mop[0][0] - cap[0], self.mop[0][1] - cap[1]) < p.waypoint_tol_m and
                     abs(wrap_pi(math.atan2(self.mop[1][1] - cap[1], self.mop[1][0] - cap[0]) - cap[2])) < 0.2 and
                     self.t - self.mop_at > 2.0):
                if self.mop is not None and self.mop[1] not in self.spin_pts:
                    self.mop_blacklist.append(self.mop[1])        # looked at it: whatever is left is hidden
                blob = self.mop_goal(poses)
                if blob is None and self.coverage() < p.coverage_goal and self.retries < 3:
                    # every place left was tried once (blacklisted) but much is still unseen: the map has grown
                    # since, try them again rather than stop
                    self.retries += 1
                    self.mop_blacklist = []
                    blob = self.mop_goal(poses)
                if blob is None:
                    self.done = True
                    return (cap[0], cap[1]), cap[2]
                self.mop, self.wp_t, self.mop_at = blob, self.t, float("inf")
            vp, ctr = self.mop
            if math.hypot(vp[0] - cap[0], vp[1] - cap[1]) < p.waypoint_tol_m:
                self.mop_at = min(self.mop_at, self.t)
            return vp, math.atan2(ctr[1] - vp[1], ctr[0] - vp[0])
        return (cap[0], cap[1]), cap[2]

    # ------------------------------------------------------------------ followers
    def formation_goals(self, poses, cap_xy, heading):
        p = self.p
        n = len(self.robots)
        lat = [(k - (n - 1) / 2) * p.spacing_m for k in range(n)]
        if self.slots is None:                   # captain in the middle slot, followers by their side, once
            ci = int(np.argmin(np.abs(lat)))
            rest = [lat[k] for k in range(n) if k != ci]
            cap = poses[self.captain]
            side = sorted(self.followers, key=lambda r: (np.array(poses[r][:2]) - cap[:2]) @ self.across)
            self.slots = dict(zip(side, sorted(rest)))
            self.slots[self.captain] = lat[ci]
        fwd = np.array([math.cos(heading), math.sin(heading)])
        if abs(fwd @ self.across) > 0.7:          # mop-up across the lanes: the line turns with the captain
            self.across = np.array([-fwd[1], fwd[0]]) * np.sign(np.array([-fwd[1], fwd[0]]) @ self.across or 1)
        out = {}
        x0, x1, y0, y1 = self.arena
        m = p.inflate_m
        for r in self.followers:
            o = self.slots[r] - self.slots[self.captain]
            xy = np.asarray(cap_xy) + o * self.across - p.back_m * abs(o) / p.spacing_m * fwd
            xy = (min(x1 - m, max(x0 + m, xy[0])), min(y1 - m, max(y0 + m, xy[1])))
            v = o * self.across
            left = float(np.sign(fwd[0] * v[1] - fwd[1] * v[0])) if o else 0.0
            out[r] = (xy, heading + left * math.radians(p.splay_deg) * min(1.0, abs(o) / p.spacing_m))
        return out

    def coverage_goals(self, poses, cap_xy):
        p, c = self.p, self.coarse
        ph = downsample(self.phi(), self.f, "mean")[:c.nx, :c.ny]
        occ = ndimage.binary_dilation(downsample(self.occupied(), self.f, "any")[:c.nx, :c.ny],
                                      disk(p.inflate_m / c.res))
        region = (np.hypot(c.X - cap_xy[0], c.Y - cap_xy[1]) <= p.leash_m) & ~occ
        gens = [self.captain] + self.followers
        G = np.array([poses[r][:2] for r in gens])
        d = np.hypot(c.X[..., None] - G[:, 0], c.Y[..., None] - G[:, 1])
        owner = np.argmin(d, axis=-1)
        out = {}
        for k, r in enumerate(gens[1:], start=1):
            cell = region & (owner == k)
            w = ph[cell] + 1e-3
            if not cell.any() or w.sum() < 0.05:
                # nothing left near: back towards the captain, at a robot spacing
                v = np.asarray(poses[r][:2]) - np.asarray(cap_xy)
                nv = float(np.hypot(*v)) or 1.0
                xy = tuple(np.asarray(cap_xy) + v / nv * min(nv, p.spacing_m))
                out[r] = (xy, poses[self.captain][2])
                continue
            X, Y = c.X[cell], c.Y[cell]
            xy = (float((w * X).sum() / w.sum()), float((w * Y).sum() / w.sum()))
            q = np.array(poses[r][:2])
            v = np.stack([X - q[0], Y - q[1]], 1)
            nv = np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-6
            u = (w[:, None] * v / nv).sum(0)
            out[r] = (xy, math.atan2(u[1], u[0]))
        return out

    def keep_m(self, r):
        """A follower farther than this from the captain is left behind: the captain slows down."""
        if self.p.mode == "formation" and self.slots:
            return abs(self.slots[r] - self.slots[self.captain]) + self.p.behind_m
        return self.p.leash_m + self.p.behind_m

    @staticmethod
    def path_ahead(pose, path, length):
        """Points every 10 cm along the first `length` metres of a path."""
        pts, left, a = [np.array(pose[:2], float)], length, np.array(pose[:2], float)
        for q in path[1:]:
            b = np.array(q, float)
            d = float(np.hypot(*(b - a)))
            if d < 1e-9:
                continue
            k = int(min(d, left) / 0.1)
            pts += [a + (b - a) * (i + 1) * 0.1 / d for i in range(k)]
            left -= d
            if left <= 0:
                break
            a = b
        return np.array(pts)

    def make_way(self, pose, goal_xy, ahead):
        """A follower on (or with its goal on) the captain's way steps aside: if it or its goal is closer than
        robot_clear_m to the first metres of the captain's path, its goal becomes the nearest free point (map,
        arena) at least that far from the whole stretch: beside the path, on either side, or behind the captain."""
        clear = self.p.robot_clear_m + 0.05
        g = np.array(goal_xy, float)
        dmin = lambda q: float(np.min(np.hypot(ahead[:, 0] - q[0], ahead[:, 1] - q[1])))
        if dmin(pose[:2]) >= clear and dmin(g) >= clear:
            return goal_xy
        inf, c = self.inflated(), self.coarse
        cands = []
        for k in range(0, len(ahead), 2):
            a, b = ahead[max(0, k - 1)], ahead[min(len(ahead) - 1, k + 1)]
            t = b - a
            nt = float(np.hypot(*t))
            if nt < 1e-9:
                continue
            n = np.array([-t[1], t[0]]) / nt
            for side in (1.0, -1.0):
                for dist in (clear, clear + 0.25, clear + 0.5):
                    cands.append(ahead[k] + side * n * dist)
        if len(ahead) > 1:
            back = ahead[0] - ahead[1]
            back /= float(np.hypot(*back)) or 1.0
            cands += [ahead[0] + back * clear * f for f in (1.0, 1.4, 1.8)]
        best, bd = None, np.inf
        for q in cands:
            if not c.inside(*q) or inf[c.index(*q)] or dmin(q) < clear - 1e-6:
                continue
            d = float(np.hypot(*(q - np.asarray(pose[:2]))))
            if d < bd:
                best, bd = q, d
        if best is None:
            return goal_xy
        return float(best[0]), float(best[1])

    # ------------------------------------------------------------------ tick
    def tick(self, t, poses):
        """New goals for every robot (self.goals). poses: rid -> (x, y, yaw) or None."""
        self.t = t
        if any(poses.get(r) is None for r in self.robots):
            return self.goals
        cap = poses[self.captain]
        cap_goal, heading = self.captain_goal(poses)
        if self.p.mode == "formation":
            fg = self.formation_goals(poses, cap[:2], heading if self.phase == "lanes" else cap[2])
        else:
            fg = self.coverage_goals(poses, cap[:2])
        behind = any(math.hypot(poses[r][0] - cap[0], poses[r][1] - cap[1]) > self.keep_m(r) for r in self.followers)
        # the captain first, on the map only: the followers make way for it (no deadlock of a captain waiting for
        # a follower that waits for the captain)
        path, miss = self.plan(cap, cap_goal, self.inflated())
        if miss > 0.5 and self.phase == "lanes":
            self.unreachable += 1                         # the waypoint itself cannot be reached: next one
            if self.unreachable >= 4:
                self.wp, self.wp_t, self.unreachable, self.best_d = self.wp + 1, self.t, 0, float("inf")
        else:
            self.unreachable = 0
        self.goals[self.captain] = dict(target=self.lookahead(cap, path, self.inflated()), yaw=wrap_pi(heading),
                                        path=path, final=tuple(cap_goal),
                                        v_max=self.p.captain_slow_v if behind else self.p.captain_v)
        ahead = self.path_ahead(cap, path, 1.5)
        for r, (xy, yaw) in fg.items():
            xy = self.make_way(poses[r], xy, ahead)
            path, _ = self.plan(poses[r], xy, self.blocked(poses, r))
            self.goals[r] = dict(target=self.lookahead(poses[r], path, self.inflated()), yaw=wrap_pi(yaw),
                                 path=path, final=tuple(xy), v_max=None)
        return self.goals
