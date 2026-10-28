"""
mocap_pose.py
Planar pose source (x, y, yaw) from Motive/OptiTrack via natnet 0.3.0.

Built on top of the project code. It adds three things needed to use the
data in closed loop instead of only printing it:

  1. configurable mapping from the 3D Motive frame to the 2D planar frame
     (Motive can be Y-up or Z-up: this changes which pair of axes is the floor)
  2. configurable quaternion order (xyzw or wxyz)
  3. timestamp on the data, so a stale pose is discarded instead of being
     reused forever when tracking is lost

When run on its own it enters diagnostic mode:

    python -m vf.localization.mocap_pose --rb 1

It is used to determine UP_AXIS and QUAT_ORDER by watching the numbers while
you move the robot by hand. Do this BEFORE trying to make the robot move on
its own.
"""

import argparse
import dataclasses
import math
import threading
import time

from natnet import NatNetClient
from vf.control.goto import wrap_pi

MOTIVE_SERVER = "127.0.0.1"     # ip of the PC running Motive
MOTIVE_LOCAL_IP = "127.0.0.1"   # ip of your interface
USE_MULTICAST = False           # True gives WinError 10049 on Windows

UP_AXIS = "y"
QUAT_ORDER = "xyzw"

STALE_AFTER = 0.2               # s - beyond this the pose is considered lost


# ---------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------

def unpack_quat(q, order=None):
    """Reorder the quaternion into (x, y, z, w) whatever the source order."""
    order = order or QUAT_ORDER
    a, b, c, d = q[0], q[1], q[2], q[3]
    if order == "xyzw":
        return a, b, c, d
    if order == "wxyz":
        return b, c, d, a
    raise ValueError(f"invalid QUAT_ORDER: {order}")


def forward_vector(q, order=None):
    """Body +X axis, expressed in the world frame.

    It is the first column of the rotation matrix associated with the
    quaternion.
    """
    x, y, z, w = unpack_quat(q, order)
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    return fx, fy, fz


def to_planar(pos, quat, up_axis=None, order=None):
    """From a 3D Motive pose to a planar pose (x, y, yaw) in radians.

    Z-up:  floor = (X, Y),  yaw about +Z
    Y-up:  floor = (X, -Z), yaw about +Y

    The minus sign on Z in the Y-up case is not arbitrary: it is needed so
    that the resulting planar frame stays right-handed, i.e. so that a
    positive yaw means counter-clockwise seen from above. Without it, all
    rotations come out inverted and the angular control diverges.
    """
    up_axis = (up_axis or UP_AXIS).lower()
    fx, fy, fz = forward_vector(quat, order)

    if up_axis == "z":
        px, py = pos[0], pos[1]
        yaw = math.atan2(fy, fx)
    elif up_axis == "y":
        px, py = pos[0], -pos[2]
        yaw = math.atan2(-fz, fx)
    else:
        raise ValueError(f"invalid UP_AXIS: {up_axis}")

    return px, py, yaw


# ---------------------------------------------------------------------
# SOURCE
# ---------------------------------------------------------------------

class MocapPose:
    """Planar pose of the given rigid body.

    get_pose() -> (x, y, yaw) or None if the data is missing or stale.
    """

    def __init__(self, rb_id,
                 server=MOTIVE_SERVER, local=MOTIVE_LOCAL_IP,
                 use_multicast=USE_MULTICAST,
                 up_axis=None, quat_order=None,
                 stale_after=STALE_AFTER):
        self.rb_id = int(rb_id)
        self.up_axis = up_axis or UP_AXIS
        self.quat_order = quat_order or QUAT_ORDER
        self.stale_after = stale_after

        self._lock = threading.Lock()
        self._pos = None
        self._quat = None
        self._stamp = 0.0
        self._orient_field = None

        self.client = NatNetClient(
            server_ip_address=server,
            local_ip_address=local,
            use_multicast=use_multicast,
        )
        self.client.connect(timeout=5)
        self.client.on_data_frame_received_event.handlers.append(self._on_frame)
        self.client.run_async()

    # ---- callback: runs on the natnet thread, not on the control thread ----

    def _on_frame(self, data_frame):
        for body in data_frame.rigid_bodies:
            if int(body.id_num) != self.rb_id:
                continue
            if body.tracking_valid is False:
                # do not update the stamp: the pose ages and get_pose discards it
                continue
            if self._orient_field is None:
                self._orient_field = self._discover_orientation_field(body)
            with self._lock:
                self._pos = tuple(body.pos)
                self._quat = tuple(getattr(body, self._orient_field))
                self._stamp = time.time()

    @staticmethod
    def _discover_orientation_field(body):
        try:
            names = [f.name for f in dataclasses.fields(body)]
        except TypeError:
            names = [a for a in dir(body) if not a.startswith("_")]
        for n in names:
            if any(k in n.lower() for k in ("rot", "quat", "orient")):
                return n
        raise RuntimeError(f"Orientation field not found. Fields: {names}")

    # ---- reading ----

    def get_raw(self):
        """(pos3d, quat, data_age) without conversions. For diagnostics."""
        with self._lock:
            if self._pos is None:
                return None
            return self._pos, self._quat, time.time() - self._stamp

    def get_pose(self):
        with self._lock:
            if self._pos is None:
                return None
            if time.time() - self._stamp > self.stale_after:
                return None
            pos, quat = self._pos, self._quat
        return to_planar(pos, quat, self.up_axis, self.quat_order)

    def wait_ready(self, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            if self.get_pose() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No data for rigid body id={self.rb_id} within {timeout}s. "
            "Check the Streaming ID in Motive's Assets panel and that "
            "Broadcast Frame Data is enabled."
        )

    def close(self):
        try:
            self.client.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------

def diagnostics(rb_id, server, local, seconds=60):
    src = MocapPose(rb_id, server, local)
    print("waiting for the rigid body...")
    # in diagnostics I also accept the raw data, the conversion may still be
    # wrong: that is exactly what I am trying to find out
    start = time.time()
    while src.get_raw() is None and time.time() - start < 5:
        time.sleep(0.05)
    if src.get_raw() is None:
        src.close()
        raise RuntimeError("no data received")

    print(f"detected orientation field: {src._orient_field}")
    print()
    print(f"{'pos (x,y,z)':>34} | {'yaw Y-up':>18} | {'yaw Z-up':>18}")
    print(f"{'':>34} | {'xyzw':>8} {'wxyz':>9} | {'xyzw':>8} {'wxyz':>9}")
    print("-" * 78)

    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            raw = src.get_raw()
            if raw is None:
                print("  (no data)")
                time.sleep(0.1)
                continue
            pos, quat, age = raw

            cells = []
            for up in ("y", "z"):
                for order in ("xyzw", "wxyz"):
                    _, _, yaw = to_planar(pos, quat, up, order)
                    cells.append(math.degrees(yaw))

            posf = "(" + ", ".join(f"{v:+7.3f}" for v in pos) + ")"
            stale = "  STALE DATA" if age > STALE_AFTER else ""
            print(f"{posf:>34} | {cells[0]:+8.1f} {cells[1]:+9.1f} "
                  f"| {cells[2]:+8.1f} {cells[3]:+9.1f}{stale}")
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=diagnostics.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rb", type=int, required=True, help="Streaming ID of the rigid body")
    ap.add_argument("--server", default=MOTIVE_SERVER)
    ap.add_argument("--local", default=MOTIVE_LOCAL_IP)
    ap.add_argument("--seconds", type=float, default=60)
    args = ap.parse_args()
    diagnostics(args.rb, args.server, args.local, args.seconds)


class MocapHub:
    """A single NatNet connection for all rigid bodies.
        Usage:
            hub = MocapHub([0, 2, 3, 4])
            hub.wait_ready()
            src0 = hub.view(0)      # has get_pose() like MocapPose
    """

    def __init__(self, rb_ids,
                 server=MOTIVE_SERVER, local=MOTIVE_LOCAL_IP,
                 use_multicast=USE_MULTICAST,
                 up_axis=None, quat_order=None,
                 stale_after=STALE_AFTER):
        self.rb_ids = [int(i) for i in rb_ids]
        self.up_axis = up_axis or UP_AXIS
        self.quat_order = quat_order or QUAT_ORDER
        self.stale_after = stale_after

        self._lock = threading.Lock()
        self._data = {i: None for i in self.rb_ids}   # id -> (pos, quat, stamp)
        self._orient_field = None

        self.client = NatNetClient(
            server_ip_address=server,
            local_ip_address=local,
            use_multicast=use_multicast,
        )
        self.client.connect(timeout=5)
        self.client.on_data_frame_received_event.handlers.append(self._on_frame)
        self.client.run_async()

    def _on_frame(self, data_frame):
        now = time.time()
        for body in data_frame.rigid_bodies:
            rb = int(body.id_num)
            if rb not in self._data:
                continue
            if body.tracking_valid is False:
                continue
            if self._orient_field is None:
                self._orient_field = MocapPose._discover_orientation_field(body)
            with self._lock:
                self._data[rb] = (tuple(body.pos),
                                  tuple(getattr(body, self._orient_field)),
                                  now)

    def get_pose(self, rb_id):
        with self._lock:
            entry = self._data.get(int(rb_id))
            if entry is None:
                return None
            pos, quat, stamp = entry
            if time.time() - stamp > self.stale_after:
                return None
        return to_planar(pos, quat, self.up_axis, self.quat_order)

    def wait_ready(self, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            missing = [i for i in self.rb_ids if self.get_pose(i) is None]
            if not missing:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No data for rigid bodies {missing} within {timeout}s. "
            "Check the Streaming IDs in Motive's Assets panel and that "
            "Broadcast Frame Data is enabled."
        )

    def view(self, rb_id):
        """Object with get_pose()/close(), interchangeable with MocapPose."""
        return _HubView(self, rb_id)

    def close(self):
        try:
            self.client.shutdown()
        except Exception:
            pass


class _HubView:
    """View onto a single rigid body of the hub."""

    def __init__(self, hub, rb_id):
        self.hub = hub
        self.rb_id = int(rb_id)

    def get_pose(self):
        return self.hub.get_pose(self.rb_id)

    def wait_ready(self, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            if self.get_pose() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(f"No data for rigid body id={self.rb_id}")

    def close(self):
        pass
