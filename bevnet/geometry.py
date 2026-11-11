"""Floor grid and camera projection of the image-based network (NumPy, run in the data loader).

The network reasons on a grid of the floor in the world frame: GRID_HALF_M around the arena centre, CELL_M per
cell (80 x 80 cells of 5 cm). For every image, the centre of every cell is lifted to a few heights on the boxes
(HEIGHTS_M: near the floor, the marker band, near the top) and projected into the image with the known camera
pose (OptiTrack, or GPS / odometry on another platform), the downward pitch and the real lens model; the
network then reads its image features at those pixels (torch grid_sample). Nothing is measured from the image
here: the geometry only says where on the floor each pixel can come from.
"""

import math

import numpy as np

GRID_HALF_M = 2.0
CELL_M = 0.05
HEIGHTS_M = (0.05, 0.15, 0.25)          # on the 30 cm boxes: near the floor, the marker, near the top
MAX_RANGE_M = 3.5                      # beyond this a cell is outside the camera's useful range


def grid_size():
    return int(round(2 * GRID_HALF_M / CELL_M))


def cell_centres():
    """(G, G, 2) world (x, y) of the cell centres; row i <-> y, column j <-> x."""
    G = grid_size()
    c = -GRID_HALF_M + CELL_M * (np.arange(G) + 0.5)
    X, Y = np.meshgrid(c, c)
    return np.stack([X, Y], -1)


def camera_rotation(yaw, pitch, roll):
    """World -> camera rotation (rows: camera x right, y down, z forward), as imgsim/render.py."""
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    d = np.cross(f, r)
    r2 = math.cos(roll) * r + math.sin(roll) * d
    d2 = -math.sin(roll) * r + math.cos(roll) * d
    return np.stack([r2, d2, f])


def project(P, pose, pitch, roll, height, K, D, size):
    """World points (..., 3) -> pixels (..., 2) and a validity mask (in front, inside the image, within the lens
    model's range). OpenCV's radial-tangential distortion (k1, k2, p1, p2, k3)."""
    x, y, yaw = pose
    R = camera_rotation(yaw, pitch, roll)
    pc = (np.asarray(P, np.float64) - np.array([x, y, height])) @ R.T
    z = pc[..., 2]
    zs = np.where(z > 1e-3, z, 1e-3)
    xn, yn = pc[..., 0] / zs, pc[..., 1] / zs
    k1, k2, p1, p2, k3 = (list(np.asarray(D, float).reshape(-1)) + [0.0] * 5)[:5]
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    W, H = size
    r2_max = ((W / 2) / K[0, 0]) ** 2 + ((H / 2) / K[1, 1]) ** 2       # image corner, undistorted
    valid = (z > 0.05) & (r2 < 1.3 * r2_max) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    return np.stack([u, v], -1), valid


def sampling_grid(pose, pitch, roll, height, K, D, size, cells=None):
    """For one image: normalised grid_sample coordinates (len(HEIGHTS_M), G, G, 2) in [-1, 1]
    (align_corners=False), their validity (len(HEIGHTS_M), G, G), and per-cell view features (3, G, G): the
    unit direction from the camera to the cell and its distance / MAX_RANGE_M."""
    cells = cell_centres() if cells is None else cells
    W, H = size
    grids, valid = [], []
    dist = np.hypot(cells[..., 0] - pose[0], cells[..., 1] - pose[1])
    for h in HEIGHTS_M:
        P = np.concatenate([cells, np.full(cells.shape[:2] + (1,), h)], -1)
        uv, ok = project(P, pose, pitch, roll, height, K, D, size)
        ok &= dist <= MAX_RANGE_M
        g = np.stack([(2 * uv[..., 0] + 1) / W - 1, (2 * uv[..., 1] + 1) / H - 1], -1)
        grids.append(np.where(ok[..., None], g, -2.0))
        valid.append(ok)
    d = np.maximum(dist, 1e-6)
    view = np.stack([(cells[..., 0] - pose[0]) / d, (cells[..., 1] - pose[1]) / d, dist / MAX_RANGE_M])
    return np.array(grids, np.float32), np.array(valid, bool), view.astype(np.float32)


def heatmap_targets(boxes, sigma_cells=1.5):
    """Training targets of the boxes (B, 2): a Gaussian per box on the grid (G, G) with 1 at its cell, the
    sub-cell offset (2, G, G) and the mask of the box cells (G, G)."""
    G = grid_size()
    heat = np.zeros((G, G), np.float32)
    off = np.zeros((2, G, G), np.float32)
    mask = np.zeros((G, G), np.float32)
    ii, jj = np.mgrid[0:G, 0:G]
    for bx, by in boxes:
        fj, fi = (bx + GRID_HALF_M) / CELL_M - 0.5, (by + GRID_HALF_M) / CELL_M - 0.5
        j, i = int(round(fj)), int(round(fi))
        if not (0 <= i < G and 0 <= j < G):
            continue
        heat = np.maximum(heat, np.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * sigma_cells ** 2)))
        heat[i, j] = 1.0
        off[:, i, j] = (fj - j, fi - i)
        mask[i, j] = 1.0
    return heat.astype(np.float32), off, mask


def decode(heat, off, threshold, min_dist_m=0.0):
    """Box positions (K, 2) from a probability map (G, G) and its offsets (2, G, G): the local maxima (3 x 3)
    above threshold; with min_dist_m > 0, a maximum closer than that to a stronger one is dropped."""
    G = heat.shape[0]
    pad = np.pad(heat, 1, constant_values=-1)
    neigh = np.max(np.stack([pad[1 + di:G + 1 + di, 1 + dj:G + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)]),
                   axis=0)
    ii, jj = np.nonzero((heat >= neigh) & (heat > threshold))
    x = -GRID_HALF_M + CELL_M * (jj + 0.5 + off[0, ii, jj])
    y = -GRID_HALF_M + CELL_M * (ii + 0.5 + off[1, ii, jj])
    if not len(ii):
        return np.zeros((0, 2))
    pts = np.stack([x, y], -1)
    if min_dist_m > 0:
        keep = []
        for k in np.argsort(-heat[ii, jj]):
            if all(np.hypot(*(pts[k] - pts[q])) >= min_dist_m for q in keep):
                keep.append(k)
        pts = pts[sorted(keep)]
    return pts
