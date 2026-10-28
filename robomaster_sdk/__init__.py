"""RoboMaster EP SDK used for the data collection: connection and camera.

Public API:

    from robomaster_sdk import Robot

    with Robot("P0") as r:
        r.chassis.drive_speed(x=0.1, y=0, z=0)
"""

from .camera import Camera, CameraError
from .robot import Robot

__all__ = ["Robot", "Camera", "CameraError"]
__version__ = "0.3.0"
