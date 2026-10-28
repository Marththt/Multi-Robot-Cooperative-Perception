"""SDK configuration constants and parameters.

All public quantities of the SDK are in SI units (meters, m/s, rad).
Conversion to the DJI SDK units (mm, deg) happens only in the modules that
talk directly to the hardware.

NOTE: values marked "TO MEASURE" are taken from the DJI documentation and
must be confirmed on the robot before data collection.
"""

# --------------------------------------------------------------------------
# Robot registry: logical name -> serial number
# --------------------------------------------------------------------------
ROBOTS = {
    "P0": "3JKCJ9800107SK",
    "P1": "3JKCJ6J0010777",
    "T0": "3JKCK7W0030D0K",
    "P4": "3JKCJ9800107SA",
    "P2": "3JKCK7W0030DJE",
    "P3": "3JKCK7W0030CX3",
    "P5": "3JKCK7W0030D0J",

        
}

CONN_TYPE = "sta"

# --------------------------------------------------------------------------
# Arm
# --------------------------------------------------------------------------
# The arm is NOT commanded via robotic_arm (which does not respond on
# these robots) but via direct servos. See arm.py and arm_safety.py.
#
# Model measured over 20 commands:  reading = command + 180
# Repeatability: ~1 degree, with a systematic hysteresis of ~1.2 degrees
# (the same command lands higher if the joint reaches it from below).
# It is NOT 1 mm: that figure was about robotic_arm and does not apply here.
SERVO_OFFSET = 180.0
SERVO_SHOULDER = 1
SERVO_ELBOW = 2
SERVO_REPEATABILITY = 1.0     # degrees
SERVO_HYSTERESIS = 1.2        # degrees

# The limits are NOT a rectangle of angles: the safe region is the
# convex hull of hand-verified poses, in arm_safety.py.
# There are at least two physical constraints: the arm stretched down
# touches the floor, and folded backwards it hits the sensor module.

ARM_MOVE_TIMEOUT = 6.0        # s
ARM_SETTLE = 0.6              # s of settling after a movement

# --------------------------------------------------------------------------
# Gripper
# --------------------------------------------------------------------------
# The gripper has NO grasp feedback: sub_status does not distinguish
# closing on nothing from closing on an object. Success must be verified
# some other way.
GRIPPER_HAS_GRASP_FEEDBACK = False
GRIPPER_POWER_DEFAULT = 50      # 1..100, maps to the PWM duty cycle
GRIPPER_ACTION_TIME = 1.5       # s, wait after open/close

# --------------------------------------------------------------------------
# Chassis: safety limits
# --------------------------------------------------------------------------
CHASSIS_VX_MAX = 0.5      # m/s
CHASSIS_VY_MAX = 0.5      # m/s
CHASSIS_VYAW_MAX = 1.5    # rad/s

# EP footprint: 32 x 24 cm, diagonal swept while rotating ~40 cm.
ROBOT_FOOTPRINT = (0.32, 0.24)
ROBOT_SWEPT_DIAMETER = 0.40

# --------------------------------------------------------------------------
# IR distance sensors
# --------------------------------------------------------------------------
# Actual configuration: 3 sensors, one front and two lateral.
# THE REAR IS BLIND: no coverage behind. See IR_ALLOW_REVERSE.
#
# TO VERIFY at first start: the port -> side map. The SDK reports the
# readings indexed by port, and if the mapping changes between sessions
# the recorded data are inconsistent. Occlude one sensor at a time with
# your hand and note which index reacts.
IR_LAYOUT = {
    "front": 0,   # TO VERIFY
    "left": 1,    # TO VERIFY
    "right": 2,   # TO VERIFY
}
IR_ORDER = ("front", "left", "right")   # canonical order in the dataset

# The arm, extended forward and down, may enter the beam of the front
# sensor and corrupt the reading right during the grasp.
# TO VERIFY: move the arm through the whole envelope and watch whether
# the front reading changes with no obstacles.
IR_FRONT_OCCLUDED_BY_ARM = None   # None = not verified yet

# Reversing allowed but at reduced speed, because there is no sensor
# coverage behind.
IR_ALLOW_REVERSE = True
REVERSE_VX_MAX = 0.15     # m/s

# --------------------------------------------------------------------------
# Sensor subscriptions
# --------------------------------------------------------------------------
# Canonical recording rate. Once chosen it must NOT change: it determines
# the length of the action chunks and the compatibility between data
# collection sessions.
RECORD_HZ = 10

# Subscription rates of the individual streams (>= RECORD_HZ).
SUB_HZ = {
    "arm_position": 20,
    "chassis_position": 20,
    "chassis_attitude": 20,
    "chassis_imu": 20,
    "ir_distance": 20,
    "gripper_status": 10,
}
