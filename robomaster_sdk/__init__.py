"""RoboMaster EP SDK for data collection and control.

Public API:

    from robomaster_sdk import Robot

    with Robot("T0") as r:
        with r.watchdog():
            r.arm.move_to(215, 133)
            r.gripper.close()
"""

from .arm import Arm, ArmError, Gripper, UnsafePose, Watchdog
from .arm_safety import SAFE_REGION, SafeRegion
from .camera import Camera, CameraError
from .robot import Robot
from .recorder import Recorder
from .teleop import Teleop

__all__ = [
    "Robot", "Arm", "Gripper", "Watchdog",
    "ArmError", "UnsafePose",
    "SAFE_REGION", "SafeRegion",
    "Teleop", "Recorder", "Camera", "CameraError",
]
__version__ = "0.2.0"
