"""A RoboMaster EP seen from above, for the figures: its chassis (32 x 24 cm) as a rectangle, turned with the
robot, the camera as a dot on the front edge and a short line along its optical axis."""

import math

from matplotlib.patches import Polygon

LENGTH_M, WIDTH_M = 0.32, 0.24
CAMERA_AHEAD_M = 0.15             # the lens is this far ahead of the chassis centre (bevnet/real.py)


def draw_robot(ax, x, y, yaw, colour="#2a78d6", fill="#d6e4f7", axis_m=0.22, label=None, zorder=5):
    """(x, y, yaw) = the camera (lens) pose, as recorded by the captures."""
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = x - CAMERA_AHEAD_M * c, y - CAMERA_AHEAD_M * s            # chassis centre behind the lens
    corners = []
    for dl, dw in ((LENGTH_M / 2, WIDTH_M / 2), (LENGTH_M / 2, -WIDTH_M / 2), (-LENGTH_M / 2, -WIDTH_M / 2),
                   (-LENGTH_M / 2, WIDTH_M / 2)):
        corners.append((cx + dl * c - dw * s, cy + dl * s + dw * c))
    ax.add_patch(Polygon(corners, closed=True, fc=fill, ec=colour, lw=1.6, zorder=zorder))
    fx, fy = cx + LENGTH_M / 2 * c, cy + LENGTH_M / 2 * s
    ax.plot([fx, fx + axis_m * c], [fy, fy + axis_m * s], color=colour, lw=2, zorder=zorder)
    ax.plot(fx, fy, "o", ms=4, color=colour, zorder=zorder + 1)
    if label:
        ax.text(cx - 0.30 * c, cy - 0.30 * s, label, ha="center", va="center", fontsize=9, color=colour,
                fontweight="bold", zorder=zorder)
