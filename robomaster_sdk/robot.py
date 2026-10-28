"""SDK entry point: the Robot class.

Minimal usage:

    from robomaster_sdk import Robot

    with Robot("T0") as r:
        print(r.arm.angles())
        r.arm.move_to(215, 133)
        r.gripper.close()
"""

import logging

import robomaster
from robomaster import robot as dji_robot

from . import config
from .arm import Arm, Gripper, Watchdog

log = logging.getLogger(__name__)


class Robot:
    """A single RoboMaster EP.

    Hides serial number, connection type and subsystem initialization.
    Exposes: .arm, .gripper, .chassis, .camera, .servo.

    NOTE: the arm is commanded via direct servos, not via robotic_arm,
    which does not respond on these robots. See arm.py.
    """

    def __init__(self, name, sn=None, conn_type=None, ip=None):
        if sn is None:
            if name not in config.ROBOTS:
                raise KeyError(
                    f"Robot '{name}' not in the registry. "
                    f"Known names: {sorted(config.ROBOTS)}. "
                    "Pass sn= explicitly for an unregistered robot."
                )
            sn = config.ROBOTS[name]

        self.name = name
        self.sn = sn
        self.conn_type = conn_type or config.CONN_TYPE
        self.ip = ip
        self._ep = None
        self.arm = None
        self.gripper = None
        self.chassis = None
        self.camera = None
        self.servo = None

        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self, check_safe=True):
        """Open the connection and set up the subsystems."""
        if self._connected:
            log.warning("%s: connect() on an already connected robot", self.name)
            return self

        log.info("Connecting to %s (sn=%s, %s)", self.name, self.sn,
                 self.conn_type)

        self._ep = dji_robot.Robot()

        # --- BEGIN IP TRICK (monkey patching) ---
        if self.ip:
            log.info("Bypassing UDP scan: forcing IP %s", self.ip)
            import robomaster.conn
            # Save the original function
            _original_scan = robomaster.conn.scan_robot_ip
            # Replace the DJI scan function with a fake one that immediately returns your IP
            robomaster.conn.scan_robot_ip = lambda *args, **kwargs: self.ip
            
            try:
                # Now initialize() will "find" the robot right away without timing out
                self._ep.initialize(conn_type=self.conn_type, sn=self.sn)
            finally:
                # Restore the original function to clean up
                robomaster.conn.scan_robot_ip = _original_scan
        else:
            self._ep.initialize(conn_type=self.conn_type, sn=self.sn)
        # --- END IP TRICK ---

        version = self._ep.get_version()
        actual_sn = self._ep.get_sn()
        log.info("%s: firmware %s, reported sn %s", self.name, version,
                 actual_sn)

        if actual_sn and actual_sn != self.sn:
            raise RuntimeError(
                f"{self.name}: the robot answers with sn={actual_sn} but "
                f"{self.sn} was expected. Connected to the wrong robot."
            )

        self.chassis = self._ep.chassis
        self.camera = self._ep.camera
        self.servo = self._ep.servo
        self.arm = Arm(self._ep.servo)
        self.gripper = Gripper(self._ep.gripper)

        self._connected = True

        if check_safe:
            try:
                s1, s2 = self.arm.angles()
                m = self.arm.margin(s1, s2)
                log.info("%s: arm at (%.1f, %.1f), margin %.1f degrees",
                         self.name, s1, s2, m)
                if m < 0:
                    log.warning(
                        "%s: the arm starts OUTSIDE the safe region. "
                        "Call arm.recover() before commanding it, or "
                        "arm.learn_pose() if you have verified that the pose "
                        "is harmless.", self.name)
            except Exception as e:
                log.warning("%s: reading the arm failed: %s",
                            self.name, e)

        return self

    def watchdog(self, **kw):
        """Create a watchdog on the arm. Use as a context manager."""
        return Watchdog(self.arm, **kw)

    def disconnect(self):
        """Close the connection (idempotent)."""
        if self._ep is not None:
            try:
                self._ep.close()
            except Exception:
                log.warning("%s: close() failed", self.name, exc_info=True)
        self._connected = False
        log.info("%s: disconnected", self.name)

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
        return False

    @property
    def is_connected(self):
        return self._connected

    def __repr__(self):
        state = "connected" if self._connected else "not connected"
        return f"<Robot {self.name} sn={self.sn} {state}>"
