"""Live window of the fleet simulation (python -m fleet.sim --live): what the brain knows, while it happens.

  map         white = never seen, light grey = seen free by the IR (gimbal sensors), blue = looked at by a
              camera (darker = more often), black = seen occupied
  robots      dot + chassis heading, the camera field of view as a wedge (it turns with the gimbal), the IR beam
              (red when it hit something), the goal of the brain (+); the captain larger, its path dashed; a
              dotted line from a robot to the object its gimbal is aiming the IR at
  objects     green squares = the truth (--hide-truth to hide them); online list of the captain: orange o =
              tentative, red x = confirmed (the report), grey dot = deleted
"""

import math

import numpy as np


class LiveView:
    def __init__(self, world, grid, cam, robots, captain, mode, show_truth=True):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mp
        plt.ion()
        self.plt, self.cam, self.captain, self.robots = plt, cam, captain, robots
        self.fig, self.ax = plt.subplots(figsize=(8, 7.4))
        a = world.arena
        ext = (a[0], a[1], a[2], a[3])
        self.im = self.ax.imshow(np.zeros(grid.shape).T, origin="lower", extent=ext, cmap="Blues", vmin=0,
                                 vmax=1.6, interpolation="nearest")
        self.occ = self.ax.imshow(np.ma.masked_all(grid.shape).T, origin="lower", extent=ext, cmap="Greys",
                                  vmin=0, vmax=1, interpolation="nearest")
        for (x0, x1, y0, y1) in world.obstacles:
            self.ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.35", ls=":", lw=1))
        if show_truth and len(world.targets):
            self.ax.plot(world.targets[:, 0], world.targets[:, 1], "s", ms=7, mfc="none", mec="tab:green", mew=1.5)
        cols = plt.cm.tab10.colors
        self.dot, self.head, self.goal, self.fov = {}, {}, {}, {}
        for k, r in enumerate(robots):
            c = cols[(k + 1) % 10]
            big = r == captain
            self.fov[r] = mp.Polygon(np.zeros((3, 2)), closed=True, fc=c, ec=c, alpha=0.12)
            self.ax.add_patch(self.fov[r])
            self.dot[r] = self.ax.plot([], [], "o", color=c, ms=12 if big else 8, mec="k" if big else c,
                                       label=f"{r}{' (captain)' if big else ''}")[0]
            self.head[r] = self.ax.plot([], [], "-", color=c, lw=2.5)[0]
            self.goal[r] = self.ax.plot([], [], "+", color=c, ms=12, mew=2)[0]
        self.path = self.ax.plot([], [], "--", color="k", lw=1)[0]
        self.beam = {r: self.ax.plot([], [], "-", color="tab:red", lw=1, alpha=0.8)[0] for r in robots}
        self.aim = {r: self.ax.plot([], [], ":", color="k", lw=1.2)[0] for r in robots}
        self.tent = self.ax.plot([], [], "o", mfc="none", mec="tab:orange", ms=10, mew=2, label="tentative")[0]
        self.dele = self.ax.plot([], [], ".", color="0.5", ms=6, label="deleted")[0]
        self.est = self.ax.plot([], [], "x", color="tab:red", ms=10, mew=2.5, label="confirmed / reported")[0]
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        self.ax.set_xlim(a[0], a[1])
        self.ax.set_ylim(a[2], a[3])
        self.ax.set_aspect("equal")
        self.mode = mode
        self.fig.tight_layout()

    def update(self, t, pose, brain, est, gim=None):
        cam = self.cam
        cov = 1 - (1 - cam.p_d_near) ** brain.views
        if brain.p.sensors == "gimbal":
            free = (brain.logodds < 0) | brain.trail
            cov = np.where(free & (cov < 0.15), 0.15, cov)
        self.im.set_data(cov.T)
        occ = brain.logodds > 0.5
        self.occ.set_data(np.ma.masked_where(~occ, np.ones(occ.shape)).T)
        half = math.radians(cam.hfov_deg) / 2
        for r in self.robots:
            x, y, yaw = pose[r]
            if gim is not None and brain.p.sensors == "gimbal":
                cy = yaw + gim[r]["gy"]
                last = brain.ir_last.get(r)
                if last:
                    (ox, oy), (ex, ey), hit = last
                    self.beam[r].set_data([ox, ex], [oy, ey])
                    self.beam[r].set_color("tab:red" if hit else "0.6")
                k = brain.tracker.get(gim[r]["task"]) if gim[r]["task"] is not None else None
                if k is not None:
                    self.aim[r].set_data([x, k.xy[0]], [y, k.xy[1]])
                else:
                    self.aim[r].set_data([], [])
            else:
                cy = yaw
            self.dot[r].set_data([x], [y])
            self.head[r].set_data([x, x + 0.3 * math.cos(yaw)], [y, y + 0.3 * math.sin(yaw)])
            arc = cy + np.linspace(-half, half, 15)
            lx, ly = x + cam.forward_m * math.cos(cy), y + cam.forward_m * math.sin(cy)
            pts = [(lx, ly)] + [(lx + cam.r_max_m * math.cos(b), ly + cam.r_max_m * math.sin(b)) for b in arc]
            self.fov[r].set_xy(pts)
            g = brain.goals.get(r)
            self.goal[r].set_data(([g["final"][0]], [g["final"][1]]) if g else ([], []))
        g = brain.goals.get(self.captain)
        if g:
            p = np.array(g["path"])
            self.path.set_data(p[:, 0], p[:, 1])
        if est:
            e = np.array(est)
            self.est.set_data(e[:, 0], e[:, 1])
        else:
            self.est.set_data([], [])
        if brain.p.sensors == "gimbal":
            for line, ks in ((self.tent, brain.tracker.tentative()),
                             (self.dele, [k for k in brain.tracker.tracks if k.state == "deleted"])):
                xy = np.array([k.xy for k in ks]).reshape(-1, 2)
                line.set_data(xy[:, 0], xy[:, 1])
        n_t = len(brain.tracker.tentative()) if brain.p.sensors == "gimbal" else 0
        self.ax.set_title(f"{self.mode}   t = {t:5.1f} s   {brain.phase}   coverage {100 * brain.coverage():.0f}%   "
                          f"confirmed: {len(est)}   tentative: {n_t}", fontsize=10)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def alive(self):
        return self.plt.fignum_exists(self.fig.number)

    def hold(self):
        """Keeps the window open at the end, until it is closed."""
        self.plt.ioff()
        self.plt.show()
