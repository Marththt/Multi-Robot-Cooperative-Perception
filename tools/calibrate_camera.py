"""RoboMaster EP camera calibration (standard OpenCV checkerboard).

Replaces the APPROXIMATE camera matrix of aruco_detector.py
(approximate_camera(), from the declared FOV, never verified) with a real
one, measured on YOUR camera: the distance and angle coming out of ArUco
depend on this calibration as much as on the marker size -- without it,
the numbers that end up in the logs (and then in the professor's fault
model) remain an approximation, not a measurement.

How to use
-----------
1. Print a standard OpenCV checkerboard (default 9x6 inner corners,
   squares of known side -- default 25mm below; MEASURE your printout,
   printers often scale slightly).
2. Hold it still/flat, move the camera (or the checkerboard) in front of
   the other in ~15-20 different poses (varied angles, distances, tilts:
   more variation = better calibration), pressing SPACE when you see it
   detected (coloured outlines) to capture that pose. 'q' to finish and
   compute.
3. python -m tools.calibrate_camera --source robot --robot-id P0 --out config/checkerboard_calib_P0.json
4. Pass that file to static_trial.py with
   --camera-calib config/checkerboard_calib_P0.json when using
   --detector aruco.

Using a local webcam instead of the robot camera
--------------------------------------------------------------------------
Useful to try the script/procedure before being in front of the real robot
(the calibration geometry does not depend on which camera it is, only the
final NUMBERS are specific to the camera used):

    python -m tools.calibrate_camera --source webcam --out camera_calib_test.json

Without a printer: checkerboard shown on a screen
--------------------------------------------------------------------------
It works, but detection is objectively harder than with a printout:
non-uniform backlight, moire between the screen pixels and the sensor
pixels, reflections on the glass -- hence the "looks fine but it doesn't
find it" that happens even when the checkerboard is clearly visible to the
eye. detect_checkerboard() below already uses cv2.findChessboardCornersSB
(much more robust than the classic one precisely in these conditions) with
an automatic fallback, but if detection is still inconsistent it helps to
use: screen at MAXIMUM brightness and high contrast, no light creating
reflections on the glass (turn off direct lights behind the camera), a
checkerboard large/close enough to fill a good part of the frame (corners
that are small in pixels are the first to fail).
"""

import argparse
import json

import cv2
import numpy as np


def detect_checkerboard(gray, checkerboard):
    """Checkerboard corners in the greyscale image, or (False, None). Tries
    findChessboardCornersSB FIRST (OpenCV >= 4.0, "sector-based" algorithm
    much more robust than the classic one against uneven lighting, slight
    blur and reflections -- exactly the conditions of a checkerboard
    photographed from a SCREEN instead of printed: non-uniform backlight,
    moire between screen pixels and sensor pixels, possible reflections on
    the glass. It already returns sub-pixel accurate corners, with no need
    for cornerSubPix afterwards. If SB is not available (old OpenCV) or
    finds nothing, it falls back to the classic findChessboardCorners BUT
    with the adaptive flags (CALIB_CB_ADAPTIVE_THRESH +
    CALIB_CB_NORMALIZE_IMAGE, not enabled by default with `None`), which
    still help a bit with the screen case even though they remain weaker
    than SB."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, checkerboard,
            flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if found:
            return True, corners, True  # final True = already sub-pixel
    found, corners = cv2.findChessboardCorners(
        gray, checkerboard,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    return found, corners, False


def capture_poses(read_frame, checkerboard=(9, 6), square_size_m=0.025,
                  min_poses=15):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_pt = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    obj_pt[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    obj_pt *= square_size_m

    object_points, image_points = [], []
    image_size = None
    n_captured = 0

    print(f"SPACE to capture a good pose (coloured outlines "
          f"visible), 'q' to finish and compute "
          f"(recommended minimum: {min_poses} poses)")
    while True:
        img = read_frame()
        if img is None:
            continue
        image_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners, already_subpixel = detect_checkerboard(gray, checkerboard)

        display = img.copy()
        if found:
            if not already_subpixel:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, checkerboard, corners, found)
        cv2.putText(display, f"captured: {n_captured}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("calibration (SPACE=capture, q=finish)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            object_points.append(obj_pt)
            image_points.append(corners)
            n_captured += 1
            print(f"  captured pose {n_captured}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    if n_captured < min_poses:
        print(f"WARNING: only {n_captured} poses (< {min_poses} "
              f"recommended) -- the calibration may not be very "
              f"accurate. You can still proceed, or run again to "
              f"add more.")
    return object_points, image_points, image_size


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["robot", "webcam"], default="robot")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--robot-id", default=None,
                   help="robot name in config.ROBOTS (only with "
                        "--source robot; if omitted you are asked for it)")
    p.add_argument("--checkerboard", default="9x6",
                   help="inner corners horizontal x vertical, e.g. 9x6")
    p.add_argument("--square-size-m", type=float, default=0.025,
                   help="side of one checkerboard square in metres "
                        "-- MEASURE your printout, do not trust the "
                        "default")
    p.add_argument("--out", default="camera_calib.json")
    args = p.parse_args()

    checkerboard = tuple(int(v) for v in args.checkerboard.lower().split("x"))

    if args.source == "webcam":
        cap = cv2.VideoCapture(args.webcam_index)

        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None

        close = cap.release
    else:
        # Imported here, not at the top of the file: this script is also
        # used just to calibrate from a webcam without having robomaster_sdk
        # installed.
        from robomaster_sdk import Robot
        from robomaster_sdk.camera import Camera as SdkCamera

        robot_id = args.robot_id or input("robot name (config.ROBOTS): ").strip()
        robot_cm = Robot(robot_id)
        robot = robot_cm.__enter__()
        camera = SdkCamera(robot.camera).start(resolution="720")

        def read_frame():
            img, _age = camera.read()
            return img

        def close():
            camera.stop()
            robot_cm.__exit__(None, None, None)

    try:
        object_points, image_points, image_size = capture_poses(
            read_frame, checkerboard, args.square_size_m)
    finally:
        close()

    if len(object_points) < 4:
        print("Too few poses captured (< 4, the mathematical minimum "
              "for calibrateCamera): calibration aborted, no file "
              "written.")
        return

    rms_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)

    print(f"\nRMS reprojection error: {rms_error:.4f} px "
          f"(below 0.5 is good, above 1.0 repeat with more poses/more "
          f"angle variation)")

    with open(args.out, "w") as f:
        json.dump({
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "rms_error_px": rms_error,
            "image_size": image_size,
            "n_poses": len(object_points),
        }, f, indent=2)
    print(f"saved to {args.out} -- pass it to static_trial.py with "
          f"--camera-calib {args.out}")


if __name__ == "__main__":
    main()
