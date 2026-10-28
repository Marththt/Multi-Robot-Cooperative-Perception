"""Geometry of the scene: triangular-prism boxes seen from above, obstacles, ray casting.

A box is an equilateral triangle in the floor plane. Its pose is the centre (x, y) and the orientation
theta = direction of the normal of face 0; faces 1 and 2 have normals theta + 120 deg and theta + 240 deg.
The ArUco marker of a face sits at the centre of that face, APOTHEM metres from the box centre.
"""

import math

import numpy as np

APOTHEM_M = 0.087                       # box centre -> face centre (side about 0.30 m)
MIN_CENTER_DISTANCE_M = 2 * APOTHEM_M   # two boxes touching face to face; closer centres are impossible
FACE_ANGLES = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)


def wrap(a):
    """Angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def face_normal(theta, face):
    return theta + FACE_ANGLES[face]


def marker_point(center, theta, face):
    n = face_normal(theta, face)
    return np.asarray(center, float) + APOTHEM_M * np.array([math.cos(n), math.sin(n)])


def box_polygon(center, theta):
    """Footprint vertices of a box (the vertices lie between the faces, 2 * APOTHEM from the centre)."""
    c = np.asarray(center, float)
    return np.array([c + 2 * APOTHEM_M * np.array([math.cos(theta + math.pi / 3 + k * 2 * math.pi / 3),
                                                   math.sin(theta + math.pi / 3 + k * 2 * math.pi / 3)])
                     for k in range(3)])


def polygons_overlap(a, b, margin=0.0):
    """True if two convex polygons overlap (separating-axis test); margin [m] keeps them apart by at
    least that gap."""
    for poly in (a, b):
        n = len(poly)
        for k in range(n):
            edge = poly[(k + 1) % n] - poly[k]
            axis = np.array([-edge[1], edge[0]]) / (np.hypot(*edge) + 1e-12)
            pa, pb = a @ axis, b @ axis
            if pa.max() + margin <= pb.min() or pb.max() + margin <= pa.min():
                return False
    return True


def rectangle(center, width, depth, angle):
    """Footprint of a rectangular obstacle (a wall or a column)."""
    c, s = math.cos(angle), math.sin(angle)
    hw, hd = width / 2, depth / 2
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return np.array([[center[0] + c * x - s * y, center[1] + s * x + c * y] for x, y in corners])


def segments_cross(p, q, a, b):
    """True if the open segment p-q crosses the segment a-b."""
    d1 = q - p
    d2 = b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return False
    w = a - p
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    u = (w[0] * d1[1] - w[1] * d1[0]) / den
    return 1e-9 < t < 1 - 1e-9 and 0.0 <= u <= 1.0


def blocked(p, q, polygons):
    """True if the line of sight p -> q crosses an edge of any polygon."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    for poly in polygons:
        n = len(poly)
        for k in range(n):
            if segments_cross(p, q, poly[k], poly[(k + 1) % n]):
                return True
    return False


def faces_seen(camera, center, theta, eps=0.05):
    """Faces whose marker side looks towards the camera."""
    out = []
    for f in range(3):
        m = marker_point(center, theta, f)
        to_cam = math.atan2(camera[1] - m[1], camera[0] - m[0])
        if math.cos(float(wrap(to_cam - face_normal(theta, f)))) > eps:
            out.append(f)
    return out


def viewing_angle(camera_yaw, theta, face):
    """Angle between the marker normal and the reversed optical axis of the camera, as the real
    detector reports it (0 = marker squarely facing the camera)."""
    return abs(float(wrap(camera_yaw + math.pi - face_normal(theta, face))))