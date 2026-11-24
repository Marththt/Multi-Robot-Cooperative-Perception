"""Live window of the fleet simulation (python -m fleet.sim --live): what the brain knows, while it happens.

  map         white = never seen, blue = seen (darker = more often), black = seen occupied
  robots      dot + heading, the camera field of view as a wedge, the goal of the brain (+); the captain larger,
              its planned path dashed
  targets     green squares = the truth (--hide-truth to hide them), red x = what the fleet would report now
              (fleet/fusion.objects on the detections so far, every 2 s)
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
        self.est = self.ax.plot([], [], "x", color="tab:red", ms=10, mew=2.5, label="reported target")[0]
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        self.ax.set_xlim(a[0], a[1])
        self.ax.set_ylim(a[2], a[3])
        self.ax.set_aspect("equal")
        self.mode = mode
        self.fig.tight_layout()

    def update(self, t, pose, brain, est):
        cam = self.cam
        self.im.set_data((1 - (1 - cam.p_d_near) ** brain.views).T)
        occ = brain.logodds > 0.5
        self.occ.set_data(np.ma.masked_where(~occ, np.ones(occ.shape)).T)
        half = math.radians(cam.hfov_deg) / 2
        for r in self.robots:
            x, y, yaw = pose[r]
            self.dot[r].set_data([x], [y])
            self.head[r].set_data([x, x + 0.3 * math.cos(yaw)], [y, y + 0.3 * math.sin(yaw)])
            arc = yaw + np.linspace(-half, half, 15)
            lx, ly = x + cam.forward_m * math.cos(yaw), y + cam.forward_m * math.sin(yaw)
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
        self.ax.set_title(f"{self.mode}   t = {t:5.1f} s   {brain.phase}   coverage {100 * brain.coverage():.0f}%   "
                          f"targets reported: {len(est)}", fontsize=10)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def alive(self):
        return self.plt.fignum_exists(self.fig.number)

    def hold(self):
        """Keeps the window open at the end, until it is closed."""
        self.plt.ioff()
        self.plt.show()
