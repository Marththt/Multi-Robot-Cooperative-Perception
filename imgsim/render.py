"""Renders what the camera of one robot sees in a scene of sim/plan.py.

The scene (boxes, yaws, obstacles, camera poses) comes from sim.plan.plan_scene, so every rendered scene has a
twin in the measurement-level sets. The image is built like a pinhole camera and then distorted with the real
lens model, so the real detector (vf/perception/aruco_detector.py) and solvePnP see the same geometry as on
the robot:

  camera      intrinsics and distortion of the checkerboard calibration (config/checkerboard_calib_P*.json),
              1280 x 720; height above the floor and downward pitch measured on the real frames (1,669
              detections of marker 0: pitch 10-13 deg, 22 deg on one robot; marker centre 9 cm below the
              camera), with a small random pitch and roll per capture
  boxes       triangular prisms, 30 cm faces, 40 cm tall, white with a blue tape band near the top and the
              bottom, the ArUco marker (DICT_4X4_50, id 0, 77 mm) on a white paper square at mid-height,
              as on the real boxes; each face lit by one random light direction
  obstacles   the cylinders of scenario E3, grey, 55 cm tall; with obstacle_style "random" every obstacle gets
              its own shape (obstacle_shapes), height, colour and surface pattern (obstacle_look)
  robots      the other robots of the scene, as dark boxes with two blue lights; with visual_style "random"
              chassis, four wheels, body and arm with gripper (sizes and greys drawn per robot)
  room        dark grey floor with texture and tile joints, white walls with grey cabinets, at a random
              distance around the arena; with visual_style "random" also bright windows, light spots on the
              floor (sun, lamps), desks with screens along the walls
  own robot   the gripper that covers the bottom of every real frame (imgsim/assets, extracted from the
              real frames of P5)
  image       exposure and white-balance jitter, vignetting, defocus, sometimes motion blur, sensor noise and
              JPEG compression, set so that the detection probability against distance and viewing angle
              matches the sensor file fitted on the real runs (imgsim/compare.py)

Visibility is resolved by drawing far objects first (painter's algorithm) and by back-face culling of the
box faces. Every marker drawn in the image is returned with its box, face and projected centre, so the
detections can be labelled.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List

import cv2
import numpy as np

from sim.geometry import face_normal

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
OBSTACLE_SHAPES = ("cylinder", "box", "prism", "hexagon", "blob")


@dataclass
class Appearance:
    """Everything that sets the look of the images. Ranges [lo, hi] are drawn per scene or per capture."""
    camera_height_m: float = 0.25                        # lens above the floor (measured 24-26 cm)
    pitch_deg: List[float] = field(default_factory=lambda: [9.0, 14.0])      # downwards, per capture
    roll_deg: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    box_height_m: float = 0.30                           # measured faces: 23-24 cm wide, 30 cm high
    box_side_m: float = 0.235
    marker_size_m: float = 0.077
    marker_centre_height_m: float = 0.15
    paper_size_m: float = 0.11
    tape_bands_m: List[List[float]] = field(default_factory=lambda: [[0.005, 0.035], [0.235, 0.265]])
    tape_bgr: List[float] = field(default_factory=lambda: [200.0, 105.0, 25.0])
    box_bgr: List[float] = field(default_factory=lambda: [236.0, 238.0, 238.0])
    plain_obstacle_height_m: float = 0.55                # grey cylinders of obstacle_style "plain"
    robot_size_m: List[float] = field(default_factory=lambda: [0.32, 0.24, 0.27])    # length, width, height
    robot_body_behind_m: float = 0.15                    # body centre behind the camera
    room_half_m: List[float] = field(default_factory=lambda: [3.6, 5.0])
    wall_height_m: float = 2.8
    floor_grey: List[float] = field(default_factory=lambda: [78.0, 100.0])
    wall_grey: List[float] = field(default_factory=lambda: [200.0, 228.0])
    light_ambient: List[float] = field(default_factory=lambda: [0.70, 0.85])
    exposure: List[float] = field(default_factory=lambda: [0.85, 1.12])
    white_balance: float = 0.04
    vignetting: float = 0.25
    defocus_sigma_px: List[float] = field(default_factory=lambda: [0.2, 0.6])
    motion_blur_share: float = 0.05
    motion_blur_px: List[int] = field(default_factory=lambda: [3, 9])
    noise_sd: List[float] = field(default_factory=lambda: [1.0, 3.0])
    jpeg_quality: List[int] = field(default_factory=lambda: [80, 95])
    draw_robots: bool = True
    draw_gripper: bool = True
    draw_markers: bool = True                            # False: plain boxes (white with tape, no paper, no marker)
    obstacle_style: str = "plain"                        # "random": shape, height, colour, pattern per obstacle
    obstacle_shapes: List[str] = field(default_factory=lambda: list(OBSTACLE_SHAPES))
    obstacle_height_m: List[float] = field(default_factory=lambda: [0.15, 1.0])       # with "random"
    light_obstacle_share: float = 0.25                   # with "random": share of white / light grey obstacles
    mask_only: bool = False                              # boxes only: white boxes on black, everything else black
                                                         # (obstacles and robots still hide the boxes), no effects
    visual_style: str = "plain"                          # "random": robots with wheels and arm, windows, light
                                                         # spots on the floor, desks and screens, wider greys
    texture_px_per_m: float = 2000.0

    @classmethod
    def load(cls, path=None):
        if not path:
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


# ====================================================================
# Camera
# ====================================================================

class Camera:
    """Real intrinsics and distortion. The scene is drawn on an undistorted canvas larger than the image
    (margin), then resampled through the lens model into the 1280 x 720 image."""

    def __init__(self, calib_path, margin=110, scale=1.0):
        with open(calib_path) as f:
            c = json.load(f)
        self.scale = float(scale)
        self.K = np.array(c["camera_matrix"], np.float64)
        self.K[:2] *= scale                               # a smaller image of the same camera
        self.D = np.array(c["dist_coeffs"], np.float64).reshape(-1)
        self.W, self.H = [int(round(v * scale)) for v in c.get("image_size", [1280, 720])]
        m = int(round(margin * scale))
        self.Ki = self.K.copy()
        self.Ki[0, 2] += m
        self.Ki[1, 2] += m
        self.Wi, self.Hi = self.W + 2 * m, self.H + 2 * m
        u, v = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        und = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), self.K, self.D, P=self.K)
        und = und.reshape(self.H, self.W, 2)
        self.map_x = (und[..., 0] + m).astype(np.float32)
        self.map_y = (und[..., 1] + m).astype(np.float32)
        uu, vv = np.meshgrid(np.arange(self.Wi, dtype=np.float32), np.arange(self.Hi, dtype=np.float32))
        self.rays = np.stack([(uu - self.Ki[0, 2]) / self.Ki[0, 0], (vv - self.Ki[1, 2]) / self.Ki[1, 1],
                              np.ones_like(uu)], -1)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        r2 = ((xx - self.W / 2) / (self.W / 2)) ** 2 + ((yy - self.H / 2) / (self.H / 2)) ** 2
        self.radius2 = r2 / r2.max()


def camera_rotation(yaw, pitch, roll):
    """World -> camera rotation (rows: camera x right, y down, z forward, in world coordinates)."""
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    d = np.cross(f, r)
    r2 = math.cos(roll) * r + math.sin(roll) * d
    d2 = -math.sin(roll) * r + math.cos(roll) * d
    return np.stack([r2, d2, f])


# ====================================================================
# Textures
# ====================================================================

def face_textures(app, rng, n=6):
    """Front views of a box face (texture px per metre), a few variants of tape and paper placement."""
    s = app.texture_px_per_m
    W, H = int(round(app.box_side_m * s)), int(round(app.box_height_m * s))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    mk = int(round(app.marker_size_m * s))
    marker = cv2.aruco.generateImageMarker(dictionary, 0, mk)
    out = []
    for _ in range(n):
        tex = np.empty((H, W, 3), np.float32)
        tex[:] = np.array(app.box_bgr) * rng.uniform(0.96, 1.0)
        tex += rng.normal(0, 2.0, (H, W, 1)).astype(np.float32)
        for lo, hi in app.tape_bands_m:          # heights from the floor; image rows from the top
            tilt = rng.uniform(-0.006, 0.006)
            for x in range(0, W, 8):
                dz = tilt * (x / W - 0.5)
                r0 = int((app.box_height_m - (hi + dz)) * s)
                r1 = int((app.box_height_m - (lo + dz)) * s)
                tex[max(r0, 0):max(r1, 0), x:x + 8] = np.array(app.tape_bgr) * rng.uniform(0.93, 1.03)
        ps = int(round(app.paper_size_m * s))
        cx = W / 2 + rng.uniform(-0.01, 0.01) * s
        cy = (app.box_height_m - app.marker_centre_height_m) * s + rng.uniform(-0.01, 0.01) * s
        paper = np.full((ps, ps, 3), 246.0, np.float32)
        y0, x0 = (ps - mk) // 2, (ps - mk) // 2
        paper[y0:y0 + mk, x0:x0 + mk] = np.where(marker[..., None] > 127, 240.0, 22.0)
        ang = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((ps / 2, ps / 2), ang, 1.0)
        M[0, 2] += cx - ps / 2
        M[1, 2] += cy - ps / 2
        mask = cv2.warpAffine(np.ones((ps, ps), np.float32), M, (W, H))
        warped = cv2.warpAffine(paper, M, (W, H))
        if app.draw_markers:
            tex = tex * (1 - mask[..., None]) + warped * mask[..., None]
        out.append(np.clip(tex, 0, 255).astype(np.uint8))
    return out


def tileable_noise(rng, size=512, scale=24):
    """Smooth random texture in [-1, 1] that wraps around (floor and walls)."""
    small = rng.normal(0, 1, (size // scale, size // scale)).astype(np.float32)
    big = cv2.resize(np.tile(small, (3, 3)), (3 * size, 3 * size), interpolation=cv2.INTER_CUBIC)[size:2 * size,
                                                                                                    size:2 * size]
    fine = rng.normal(0, 0.35, (size, size)).astype(np.float32)
    t = big + cv2.GaussianBlur(fine, (3, 3), 0)
    return t / (np.abs(t).max() + 1e-6)


def obstacle_look(app, seed, centre, radius):
    """Random look of one obstacle on the footprint circle (centre, radius) of the scene: a convex footprint of
    one of app.obstacle_shapes inscribed in it, a height, a colour and a surface pattern. Never white with blue
    bands (the look of a box)."""
    rng = np.random.default_rng(seed)
    shape = str(rng.choice(app.obstacle_shapes))
    rot = rng.uniform(0, 2 * math.pi)
    if shape == "cylinder":
        ang = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        rad = np.full(20, radius)
    elif shape == "box":
        phi = rng.uniform(0.35, 1.22)
        ang = np.array([phi, math.pi - phi, math.pi + phi, 2 * math.pi - phi])
        rad = np.full(4, radius)
    elif shape == "prism":
        ang, rad = np.array([0.0, 2.0944, 4.1888]), np.full(3, radius)
    elif shape == "hexagon":
        ang, rad = np.linspace(0, 2 * math.pi, 6, endpoint=False), np.full(6, radius)
    else:                                                  # blob: an irregular convex polygon
        n = int(rng.integers(6, 11))
        ang = np.sort(rng.uniform(0, 2 * math.pi, n))
        rad = radius * rng.uniform(0.6, 1.0, n)
    pts = np.stack([centre[0] + rad * np.cos(ang + rot), centre[1] + rad * np.sin(ang + rot)], 1)
    hull = cv2.convexHull(np.float32(pts)).reshape(-1, 2).astype(float)
    area = np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1])
    if area < 0:                                           # counter-clockwise
        hull = hull[::-1]
    height = _u(rng, app.obstacle_height_m)

    def colour(light):
        if light:
            v = rng.uniform(195, 245)
            return np.array([v, v, v]) * (1 + rng.uniform(-0.04, 0.04, 3))
        hsv = np.uint8([[[rng.integers(0, 180), rng.integers(0, 205), rng.integers(35, 225)]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(float)

    light = rng.random() < app.light_obstacle_share
    base = colour(light)
    second = colour(False)
    if light and second[0] > 1.3 * max(second[1], second[2]):      # no blue on white
        second = second[[2, 1, 0]]
    th = max(16, int(round(64 * height / 0.5)))
    tex = np.empty((th, 128, 3), np.float32)
    tex[:] = base
    pattern = str(rng.choice(["plain", "noise", "stripes", "bands", "checker"]))
    yy, xx = np.mgrid[0:th, 0:128]
    if pattern == "noise":
        tex *= (1 + 0.12 * rng.normal(0, 1, (th, 128, 1))).astype(np.float32)
    elif pattern == "stripes":
        period = int(rng.integers(8, 32))
        tex[(xx // period) % 2 == 1] = second
    elif pattern == "bands":
        period = int(rng.integers(8, 40))
        tex[(yy // period) % 2 == 1] = second
    elif pattern == "checker":
        period = int(rng.integers(8, 24))
        tex[((xx // period) + (yy // period)) % 2 == 1] = second
    tex += rng.normal(0, 2.0, tex.shape).astype(np.float32)
    top = base * 1.05
    return dict(footprint=hull, height=height, tex=np.clip(tex, 0, 255), top=top)


def part_look(footprint, z0, height, bgr, rng, pattern="noise", second=None):
    """A solid for draw_solid: convex footprint (counter-clockwise), from z0 to z0 + height, colour and pattern."""
    F = np.asarray(footprint, float)
    area = np.sum(F[:, 0] * np.roll(F[:, 1], -1) - np.roll(F[:, 0], -1) * F[:, 1])
    if area < 0:
        F = F[::-1]
    th = max(8, int(round(64 * height / 0.5)))
    tex = np.empty((th, 64, 3), np.float32)
    tex[:] = bgr
    if pattern == "noise":
        tex *= (1 + 0.10 * rng.normal(0, 1, (th, 64, 1))).astype(np.float32)
    elif pattern == "stripes" and second is not None:
        tex[:, (np.arange(64) // 4) % 2 == 1] = second
    elif pattern == "band" and second is not None:
        tex[th // 3: th // 3 + max(1, th // 6)] = second
    return dict(footprint=F, z0=float(z0), height=float(height), tex=np.clip(tex, 0, 255), top=np.asarray(bgr) * 1.08)


def rect(cx, cy, yaw, x0, x1, y0, y1):
    """Rectangle [x0, x1] x [y0, y1] in a frame at (cx, cy, yaw) -> world corners (4, 2), counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.stack([cx + c * pts[:, 0] - s * pts[:, 1], cy + s * pts[:, 0] + c * pts[:, 1]], 1)


def robot_parts(bc, yaw, rng):
    """A RoboMaster-like robot at body centre bc, heading yaw: chassis, four wheels with rollers, upper body,
    arm and gripper, in dark greys (drawn per robot)."""
    k = rng.uniform(0.9, 1.1)
    dark = lambda lo, hi: np.full(3, rng.uniform(lo, hi)) * (1 + rng.uniform(-0.05, 0.05, 3))
    x, y = bc
    parts = [part_look(rect(x, y, yaw, -0.16 * k, 0.16 * k, -0.095 * k, 0.095 * k), 0.035, 0.085, dark(35, 70), rng)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            wx, wy = sx * 0.105 * k, sy * 0.12 * k
            parts.append(part_look(rect(x, y, yaw, wx - 0.05, wx + 0.05, wy - 0.022, wy + 0.022), 0.0, 0.10,
                                   dark(20, 40), rng, "stripes", dark(60, 90)))
    body = dark(30, 65)
    led = np.array([255.0, 150.0, 40.0]) if rng.random() < 0.6 else dark(70, 110)
    parts.append(part_look(rect(x, y, yaw, -0.15 * k, 0.05 * k, -0.08 * k, 0.08 * k), 0.12, rng.uniform(0.06, 0.10),
                           body, rng, "band", led))
    arm = dark(25, 60)
    up = rng.uniform(0.08, 0.13)
    parts.append(part_look(rect(x, y, yaw, 0.0, 0.035, -0.018, 0.018), 0.19, up, arm, rng))
    parts.append(part_look(rect(x, y, yaw, 0.03, 0.15 * k, -0.018, 0.018), 0.17 + up, 0.035, arm, rng))
    parts.append(part_look(rect(x, y, yaw, 0.13 * k, 0.20 * k, -0.04, 0.04), 0.14 + up * 0.6, rng.uniform(0.06, 0.1),
                           dark(20, 50), rng))
    return parts


# ====================================================================
# Renderer
# ====================================================================

class Renderer:
    """scale < 1 renders a smaller image of the same camera (intrinsics scaled), e.g. 0.2 -> 256 x 144 for an
    image-based network; blur and motion-blur sizes are scaled with it."""

    def __init__(self, calib_path, app=None, seed=0, scale=1.0):
        self.cam = Camera(calib_path, scale=scale)
        self.app = app or Appearance()
        rng = np.random.default_rng(seed)
        self.textures = face_textures(self.app, rng)
        self.noise = tileable_noise(rng)
        self.noise_wall = tileable_noise(rng, scale=48)
        mask = cv2.imread(os.path.join(ASSETS, "gripper_mask.png"), cv2.IMREAD_GRAYSCALE)
        grip = cv2.imread(os.path.join(ASSETS, "gripper.png"))
        if mask is not None and grip is not None:
            self.grip_mask = cv2.resize(mask, (self.cam.W, self.cam.H)).astype(np.float32)[..., None] / 255.0
            self.grip = cv2.resize(grip, (self.cam.W, self.cam.H)).astype(np.float32)
        else:
            self.grip_mask = None

    # ------------------------------------------------------------ room
    def scene_style(self, rng):
        """Appearance drawn once per scene: room, light, cabinets."""
        a = self.app
        L = _u(rng, a.room_half_m)
        cabinets = []
        for wall in range(4):
            for _ in range(int(rng.integers(0, 4))):
                c, w = rng.uniform(-L, L), rng.uniform(0.6, 2.0)
                cabinets.append((wall, c - w / 2, c + w / 2, rng.uniform(0.9, 1.25), rng.uniform(110, 150)))
        light = rng.normal(0, 1, 3)
        light[2] = abs(light[2]) + 0.5
        style = dict(room=L, floor=_u(rng, a.floor_grey), wall=_u(rng, a.wall_grey), cabinets=cabinets,
                     light=light / np.linalg.norm(light), ambient=_u(rng, a.light_ambient))
        if a.obstacle_style == "random":                   # drawn last: "plain" scenes keep their random stream
            style["obstacle_seed"] = int(rng.integers(0, 2 ** 31 - 1))
            style["obstacle_looks"] = {}
        if a.visual_style == "random":
            style["floor"] = rng.uniform(50, 115)
            style["wall"] = rng.uniform(185, 252)
            style["windows"] = []
            for wall in range(4):
                for _ in range(int(rng.integers(0, 4))):
                    c, w = rng.uniform(-L, L), rng.uniform(0.8, 2.5)
                    style["windows"].append((wall, c - w / 2, c + w / 2, rng.uniform(0.7, 1.3), rng.uniform(1.9, 2.6),
                                             rng.uniform(235, 255), rng.uniform(0.4, 1.0)))
            style["spots"] = [(rng.uniform(-L, L), rng.uniform(-L, L), rng.uniform(0.2, 0.9), rng.uniform(0.15, 0.5),
                               rng.uniform(0, math.pi), rng.uniform(60, 190)) for _ in range(int(rng.integers(0, 4)))]
            clutter = []
            for _ in range(int(rng.integers(0, 7))):
                wall = int(rng.integers(0, 4))
                along, depth = rng.uniform(-L + 0.6, L - 0.6), L - rng.uniform(0.35, 0.9)
                cx, cy = [(depth, along), (-depth, along), (along, depth), (along, -depth)][wall]
                yaw = 0.0 if wall < 2 else math.pi / 2
                dw, dd = rng.uniform(0.5, 1.4), rng.uniform(0.3, 0.45)
                top = np.full(3, rng.uniform(90, 230))
                clutter.append(part_look(rect(cx, cy, yaw, -dd, dd, -dw, dw), 0.0, rng.uniform(0.68, 0.8), top * 0.8,
                                         rng))
                if rng.random() < 0.7:
                    clutter.append(part_look(rect(cx, cy, yaw, -0.03, 0.03, -0.25, 0.25), 0.78, rng.uniform(0.3, 0.42),
                                             np.full(3, rng.uniform(15, 40)), rng))
            style["clutter"] = clutter
            style["robot_seed"] = int(rng.integers(0, 2 ** 31 - 1))
            style["robot_looks"] = {}
        return style

    def background(self, C, R, style):
        """Floor, walls and ceiling by ray casting on the canvas."""
        a = self.app
        d = self.cam.rays @ R                              # (Hi, Wi, 3) world directions
        L, h = style["room"], C[2]
        big = 1e9
        tw = np.full(d.shape[:2], big, np.float32)
        wall_id = np.zeros(d.shape[:2], np.int8)
        coord = np.zeros(d.shape[:2], np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1)]):
                t = (sign * L - C[axis]) / d[..., axis]
                ok = (t > 0) & (t < tw)
                tw = np.where(ok, t, tw)
                wall_id = np.where(ok, k, wall_id)
            tf = np.where(d[..., 2] < -1e-6, -h / d[..., 2], big)
        floor = tf < tw
        img = np.empty(d.shape, np.float32)
        # floor
        X = C[0] + tf * d[..., 0]
        Y = C[1] + tf * d[..., 1]
        n = self.noise
        iu = (np.nan_to_num(X) * 100).astype(np.int64) % n.shape[1]
        iv = (np.nan_to_num(Y) * 100).astype(np.int64) % n.shape[0]
        g = style["floor"] * (1 + 0.10 * n[iv, iu])
        joint = (np.abs(((X / 1.0) % 1.0) - 0.5) > 0.497) | (np.abs(((Y / 1.0) % 1.0) - 0.5) > 0.497)
        g = np.where(joint, g * 0.85, g)
        g *= 1 + 0.35 * np.clip((tf - 1.0) / 5.0, 0, 1)          # glancing reflection: far floor looks lighter
        for sx, sy, ra, rb, ang, amp in style.get("spots", []):     # sun or lamp light on the floor
            dx, dy = X - sx, Y - sy
            u = dx * math.cos(ang) + dy * math.sin(ang)
            v = -dx * math.sin(ang) + dy * math.cos(ang)
            g = g + amp * np.exp(-0.5 * ((np.nan_to_num(u) / ra) ** 2 + (np.nan_to_num(v) / rb) ** 2))
        # walls
        Zw = h + tw * d[..., 2]
        along = np.where(wall_id < 2, C[1] + tw * d[..., 1], C[0] + tw * d[..., 0])
        nw = self.noise_wall
        wg = style["wall"] * (1 + 0.04 * nw[(np.nan_to_num(Zw) * 60).astype(np.int64) % nw.shape[0],
                                           (np.nan_to_num(along) * 60).astype(np.int64) % nw.shape[1]])
        for wall, lo, hi, top, grey in style["cabinets"]:
            m = (wall_id == wall) & (along > lo) & (along < hi) & (Zw < top)
            wg = np.where(m, grey, wg)
            wg = np.where(m & (Zw > top - 0.03), grey * 0.7, wg)
        for wall, lo, hi, bottom, top, bright, mull in style.get("windows", []):
            m = (wall_id == wall) & (along > lo) & (along < hi) & (Zw > bottom) & (Zw < top)
            frame = (np.abs(((along - lo) / mull) % 1.0 - 0.5) > 0.47) | (Zw < bottom + 0.04) | (Zw > top - 0.04)
            wg = np.where(m, np.where(frame, 70.0, bright), wg)
        wg = np.where(Zw < 0.08, 150.0, wg)
        wg = np.where(Zw > a.wall_height_m, 228.0, wg)
        val = np.where(floor, g, wg)
        img[..., 0] = val * np.where(floor, 1.0, 0.99)
        img[..., 1] = val
        img[..., 2] = val * np.where(floor, 0.98, 1.0)
        return img

    # ------------------------------------------------------------ objects
    def project(self, P, C, R):
        """World points (N, 3) -> canvas pixels (N, 2) and depth (N,)."""
        pc = (np.asarray(P, float) - C) @ R.T
        z = pc[:, 2]
        uv = (pc[:, :2] / np.maximum(z[:, None], 1e-6)) * [self.cam.Ki[0, 0], self.cam.Ki[1, 1]] + \
            [self.cam.Ki[0, 2], self.cam.Ki[1, 2]]
        return uv, z, pc

    def draw_textured_quad(self, img, tex, quad_uv, shade):
        x0, y0 = np.floor(quad_uv.min(0)).astype(int)
        x1, y1 = np.ceil(quad_uv.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.cam.Wi - 1), min(y1, self.cam.Hi - 1)
        if x1 <= x0 or y1 <= y0:
            return
        h, w = tex.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32(quad_uv - [x0, y0])
        Hm = cv2.getPerspectiveTransform(src, dst)
        size = (x1 - x0 + 1, y1 - y0 + 1)
        warped = cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        mask = np.zeros((size[1], size[0]), np.float32)
        cv2.fillConvexPoly(mask, np.round(dst * 16).astype(np.int32), 1.0, lineType=cv2.LINE_AA, shift=4)
        roi = img[y0:y1 + 1, x0:x1 + 1]
        roi[:] = roi * (1 - mask[..., None]) + warped * shade * mask[..., None]

    def draw_polygon(self, img, uv, bgr):
        hull = cv2.convexHull(np.float32(uv)).reshape(-1, 2)
        cv2.fillConvexPoly(img, np.round(hull * 16).astype(np.int32), [float(c) for c in bgr],
                           lineType=cv2.LINE_AA, shift=4)

    def draw_solid(self, img, look, C, R, style):
        """A vertical prism on a convex footprint: the side faces that face the camera (far ones first),
        textured and lit, then the top if the camera is above it."""
        F, tex = look["footprint"], look["tex"]
        z0 = look.get("z0", 0.0)
        h = z0 + look["height"]
        n = len(F)
        edges = np.linalg.norm(np.roll(F, -1, 0) - F, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(edges)]) / max(edges.sum(), 1e-9)
        W = tex.shape[1]
        faces = []
        for i in range(n):
            p, q = F[i], F[(i + 1) % n]
            d = q - p
            nv = np.array([d[1], -d[0]]) / max(np.hypot(*d), 1e-9)          # outward for counter-clockwise
            mid = (p + q) / 2
            if np.dot(nv, C[:2] - mid) <= 0:
                continue
            faces.append((float(np.hypot(*(mid - C[:2]))), i, p, q, nv))
        faces.sort(key=lambda f: -f[0])
        for _, i, p, q, nv in faces:
            quad = np.array([[*p, h], [*q, h], [*q, z0], [*p, z0]])
            uv, z, _ = self.project(quad, C, R)
            if (z < 0.05).any():
                continue
            c0 = int(cum[i] * W)
            c1 = max(c0 + 2, int(cum[i + 1] * W))
            n3 = np.array([nv[0], nv[1], 0.0])
            shade = style["ambient"] + (1 - style["ambient"]) * max(0.0, float(n3 @ style["light"]))
            self.draw_textured_quad(img, np.ascontiguousarray(tex[:, c0:min(c1, W)]), uv, shade)
        if C[2] > h:
            top = np.column_stack([F, np.full(n, h)])
            uv, z, _ = self.project(top, C, R)
            if (z > 0.05).all():
                shade = style["ambient"] + (1 - style["ambient"]) * max(0.0, float(style["light"][2]))
                self.draw_polygon(img, uv, look["top"] * shade)

    def render(self, scene, k, rng, style=None):
        """Image of capture k of scene (sim.scene.Scene), the markers drawn in it and the camera settings.
        Returns (BGR uint8 image, [dict(box, face, u, v, depth)], dict(pitch_deg, roll_deg, height_m)); with
        mask_only the dict also has box_pixels: per box, how many pixels of it are visible in the image."""
        clean, markers, info = self.render_clean(scene, k, rng, style)
        return self.camera_effects(clean, rng), markers, info

    def render_clean(self, scene, k, rng, style=None):
        """As render, but the image before the camera effects (float BGR): several readings of one stop are
        camera_effects() of the same clean image, as a robot standing still takes them."""
        a = self.app
        style = style or self.scene_style(rng)
        x, y, yaw = scene.captures[k].pose
        pitch, roll = math.radians(_u(rng, a.pitch_deg)), math.radians(_u(rng, a.roll_deg))
        C = np.array([x, y, a.camera_height_m])
        R = camera_rotation(yaw, pitch, roll)
        m = a.mask_only
        img = np.zeros((self.cam.Hi, self.cam.Wi, 3), np.float32) if m else self.background(C, R, style)
        objects = []                       # (depth, kind, data)
        half = a.box_side_m / 2
        apothem = a.box_side_m / (2 * math.sqrt(3))       # centre to face of the triangle of side box_side_m
        for j, (c, th) in enumerate(zip(scene.boxes, scene.thetas)):
            for f in range(3):
                nrm = face_normal(th, f)
                nv = np.array([math.cos(nrm), math.sin(nrm)])
                fc = np.asarray(c, float) + apothem * nv
                if np.dot(nv, np.array([x, y]) - fc) <= 0:          # back face
                    continue
                t = np.array([-nv[1], nv[0]])
                left, right = fc - t * half, fc + t * half
                quad = np.array([[*left, a.box_height_m], [*right, a.box_height_m], [*right, 0.0], [*left, 0.0]])
                depth = float(np.hypot(*(np.asarray(c) - [x, y])))
                objects.append((depth + 0.01 * f, "face", (j, f, quad, nv, fc)))
        for q, poly in enumerate(scene.obstacles):
            poly = np.asarray(poly, float)
            oc, r = poly.mean(0), float(np.max(np.hypot(*(poly - poly.mean(0)).T)))
            if "obstacle_seed" in style:                   # the same look from every robot of the scene
                looks = style["obstacle_looks"]
                if q not in looks:
                    looks[q] = obstacle_look(a, style["obstacle_seed"] + 7_919 * q, oc, r)
                objects.append((float(np.hypot(*(oc - [x, y]))), "solid", looks[q]))
            else:
                objects.append((float(np.hypot(*(oc - [x, y]))), "cylinder", (oc, r)))
        for part in style.get("clutter", []):
            objects.append((float(np.hypot(*(part["footprint"].mean(0) - [x, y]))), "solid", part))
        if a.draw_robots and "robot_seed" in style:
            looks = style["robot_looks"]
            for q, cap in enumerate(scene.captures):
                if q == k:
                    continue
                if q not in looks:
                    qx, qy, qyaw = cap.pose
                    bc = np.array([qx, qy]) - a.robot_body_behind_m * np.array([math.cos(qyaw), math.sin(qyaw)])
                    looks[q] = robot_parts(bc, qyaw, np.random.default_rng(style["robot_seed"] + 104_729 * q))
                for part in looks[q]:
                    objects.append((float(np.hypot(*(part["footprint"].mean(0) - [x, y]))), "solid", part))
        elif a.draw_robots:
            L_, W_, H_ = a.robot_size_m
            for q, cap in enumerate(scene.captures):
                if q == k:
                    continue
                qx, qy, qyaw = cap.pose
                bc = np.array([qx, qy]) - a.robot_body_behind_m * np.array([math.cos(qyaw), math.sin(qyaw)])
                objects.append((float(np.hypot(*(bc - [x, y]))), "robot", (bc, qyaw, L_, W_, H_)))
        objects.sort(key=lambda o: -o[0])
        markers = []
        for depth, kind, data in objects:
            if kind == "face":
                j, f, quad, nv, fc = data
                uv, z, _ = self.project(quad, C, R)
                if (z < 0.05).any():
                    continue
                n3 = np.array([nv[0], nv[1], 0.0])
                shade = style["ambient"] + (1 - style["ambient"]) * max(0.0, float(n3 @ style["light"]))
                if m:                                      # white, with the box number in the second channel
                    self.draw_polygon(img, uv, [255.0, float(j + 1), 0.0])
                else:
                    tex = self.textures[(j * 3 + f) % len(self.textures)]
                    self.draw_textured_quad(img, tex, uv, shade)
                mc = np.array([[fc[0], fc[1], a.marker_centre_height_m]])
                _, mz, mpc = self.project(mc, C, R)
                markers.append(dict(box=j, face=f, pc=mpc[0], depth=float(mz[0])))
            elif kind == "cylinder":
                oc, r = data
                ang = np.linspace(0, 2 * math.pi, 32, endpoint=False)
                ring = np.stack([oc[0] + r * np.cos(ang), oc[1] + r * np.sin(ang)], 1)
                pts = np.vstack([np.column_stack([ring, np.zeros(32)]),
                                 np.column_stack([ring, np.full(32, a.plain_obstacle_height_m)])])
                uv, z, _ = self.project(pts, C, R)
                if (z < 0.05).any():
                    continue
                self.draw_polygon(img, uv, np.zeros(3) if m else np.array([150.0, 155.0, 160.0]) * style["ambient"] * 1.1)
            elif kind == "solid":
                if m:
                    data = dict(data, tex=np.zeros_like(data["tex"]), top=np.zeros(3))
                self.draw_solid(img, data, C, R, style)
            else:
                bc, qyaw, L_, W_, H_ = data
                fwd = np.array([math.cos(qyaw), math.sin(qyaw)])
                side = np.array([-fwd[1], fwd[0]])
                corners = [bc + sx * fwd * L_ / 2 + sy * side * W_ / 2 for sx in (-1, 1) for sy in (-1, 1)]
                pts = np.array([[*p, zz] for p in corners for zz in (0.02, H_)])
                uv, z, _ = self.project(pts, C, R)
                if (z < 0.05).any():
                    continue
                self.draw_polygon(img, uv, [0.0, 0.0, 0.0] if m else [42.0, 40.0, 40.0])
                top = np.array([[*p, H_] for p in corners])
                tuv, _, _ = self.project(top, C, R)
                self.draw_polygon(img, tuv, [0.0, 0.0, 0.0] if m else [70.0, 68.0, 66.0])
                for sy in ([] if m else (-1, 1)):
                    led = np.array([[*(bc + fwd * L_ / 2 + sy * side * W_ / 4), H_ * 0.6]])
                    luv, lz, _ = self.project(led, C, R)
                    if lz[0] > 0.05 and np.dot(fwd, np.array([x, y]) - bc) > 0:
                        cv2.circle(img, tuple(int(v) for v in luv[0]), max(1, int(6 / max(lz[0], 0.3))),
                                   (255.0, 170.0, 60.0), -1, lineType=cv2.LINE_AA)
        # through the lens
        out = cv2.remap(img, self.cam.map_x, self.cam.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        box_pixels = None
        if m:
            white = out[..., 0] > 127.0
            ids = cv2.remap(img[..., 1], self.cam.map_x, self.cam.map_y, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE)
            box_pixels = np.bincount(np.round(ids[white]).astype(int).clip(0, len(scene.boxes)),
                                     minlength=len(scene.boxes) + 1)[1:].tolist()
            out = np.repeat(np.where(white, 255.0, 0.0).astype(np.float32)[..., None], 3, axis=2)
        elif a.draw_gripper and self.grip_mask is not None:
            out = out * (1 - self.grip_mask) + self.grip * self.grip_mask
        # where the markers fell in the image
        if markers:
            pc = np.array([m["pc"] for m in markers], np.float64)
            uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.cam.K, self.cam.D)
            for m, p in zip(markers, uv.reshape(-1, 2)):
                m["u"], m["v"] = float(p[0]), float(p[1])
                m.pop("pc")
        markers = [m for m in markers if m["depth"] > 0 and 0 <= m["u"] < self.cam.W and 0 <= m["v"] < self.cam.H]
        info = dict(pitch_deg=round(math.degrees(pitch), 2), roll_deg=round(math.degrees(roll), 2),
                    height_m=a.camera_height_m)
        if box_pixels is not None:
            info["box_pixels"] = box_pixels
        return out, markers, info

    def camera_effects(self, img, rng):
        a = self.app
        if a.mask_only:                                    # masks: no camera effects
            return np.clip(img, 0, 255).astype(np.uint8)
        img = img * _u(rng, a.exposure) * (1 + rng.uniform(-a.white_balance, a.white_balance, 3))
        img = img * (1 - a.vignetting * self.cam.radius2[..., None])
        s = _u(rng, a.defocus_sigma_px) * self.cam.scale
        if s > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), s)
        n = int(round(int(rng.integers(a.motion_blur_px[0], a.motion_blur_px[1] + 1)) * self.cam.scale))
        if rng.random() < a.motion_blur_share and n >= 2:
            kern = np.zeros((n, n), np.float32)
            kern[n // 2, :] = 1.0 / n
            M = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(-20, 20), 1.0)
            kern = cv2.warpAffine(kern, M, (n, n))
            img = cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))
        img = img + rng.normal(0, _u(rng, a.noise_sd), img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        q = int(rng.integers(a.jpeg_quality[0], a.jpeg_quality[1] + 1))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
