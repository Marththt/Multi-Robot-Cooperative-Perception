"""SDK entry point: the Robot class.

Minimal usage:

    from robomaster_sdk import Robot

    with Robot("P0") as r:
        r.chassis.drive_speed(x=0.1, y=0, z=0)
"""

import logging

import robomaster
from robomaster import robot as dji_robot

from . import config

log = logging.getLogger(__name__)


class Robot:
    """A single RoboMaster EP.

    Hides serial number, connection type and subsystem initialization.
    Exposes: .chassis, .camera, .servo.
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
        self.chassis = None
        self.camera = None
        self.servo = None

        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self):
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

        self._connected = True

        return self

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
