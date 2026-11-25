"""Gimbal heading correction of one robot, from its own fan (IR beams) and YOLO rays against the boxes OptiTrack
tracks, measured at the start of every lab run.

Why: the gimbal yaw read back from the SDK is not tied to the rigid body by a fixed offset. YAW_OFFSET_DEG
(+13.3 deg, measured on G11 on 2026-11-23) was 10-17 deg off for all three robots on 2026-11-25 (same correction
for the IR and the camera, constant over the run): 25 cm sideways at 1 m, every box seen twice.

estimate(ir_rows, cam_rows, boxes) tries corrections from -max_deg to +max_deg: an IR beam votes when its end (one
apothem behind the face) falls within tol_m of a box centre, a YOLO ray when its bearing is within tol_deg of a
box; the votes are normalised per sensor and summed; the best correction is refined to 0.25 deg. Returns
(correction [deg], IR votes, camera votes), or None when there are fewer than min_votes votes or the peak is not
clear (min_peak times any other correction more than 5 deg away; with few boxes in view a wrong correction can
line some rays up with some box by chance).
"""

import math

import numpy as np

APOTHEM_M = 0.068


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def estimate(ir_rows, cam_rows, boxes, max_deg=35.0, tol_m=0.07, tol_deg=2.0, min_votes=30, min_peak=1.25):
    B = np.asarray(boxes, float).reshape(-1, 2)
    if not len(B):
        return None
    ir = [r for r in ir_rows if r.get("d") is not None]
    O = np.array([r["o"] for r in ir], float).reshape(-1, 2)
    H = np.array([r["h"] for r in ir], float)
    D = np.array([r["d"] for r in ir], float) + APOTHEM_M
    L, A = [], []
    for r in cam_rows:
        for ang, _ in r["rays"]:
            L.append(r["lens"][:2])
            A.append(ang)
    L, A = np.asarray(L, float).reshape(-1, 2), np.asarray(A, float)
    bear = np.arctan2(B[None, :, 1] - L[:, None, 1], B[None, :, 0] - L[:, None, 0]) if len(A) else None

    def votes(deg):
        c = math.radians(deg)
        n_ir = n_cam = 0
        if len(H):
            ex, ey = O[:, 0] + D * np.cos(H + c), O[:, 1] + D * np.sin(H + c)
            n_ir = int((np.hypot(ex[:, None] - B[None, :, 0], ey[:, None] - B[None, :, 1]).min(1) < tol_m).sum())
        if len(A):
            n_cam = int((np.abs(_wrap(A[:, None] + c - bear)).min(1) < math.radians(tol_deg)).sum())
        return n_ir, n_cam

    def score(v):
        return v[0] / max(1, len(H)) + v[1] / max(1, len(A))

    coarse = [(score(votes(d)), d) for d in np.arange(-max_deg, max_deg + 0.5, 1.0)]
    s0, d0 = max(coarse)
    other = max([sc for sc, d in coarse if abs(d - d0) > 5.0] or [0.0])
    if s0 <= 0 or s0 < min_peak * other or abs(d0) >= max_deg:
        return None                     # no clear peak: a chance alignment with a few boxes, not the heading
    fine = [(score(votes(d)), d) for d in np.arange(d0 - 1.0, d0 + 1.01, 0.25)]
    s, d = max(fine)
    v = votes(d)
    if v[0] + v[1] < min_votes:
        return None
    return float(d), v[0], v[1]


def rotate_row(row, deg):
    """A report with every heading turned by deg (the IR beam, the camera lens and its rays)."""
    if not deg:
        return row
    c = math.radians(deg)
    row = dict(row)
    if row.get("h") is not None:
        row["h"] = row["h"] + c
    if row.get("type") == "cam":
        row["lens"] = [row["lens"][0], row["lens"][1], row["lens"][2] + c]
        row["rays"] = [[a + c, d] for a, d in row["rays"]]
    return row
