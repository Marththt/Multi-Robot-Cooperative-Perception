"""RoboMaster camera, with a background reader.

Why a separate thread
--------------------------
`read_cv2_image` blocks until it has a frame. Calling it inside the
recording loop would tie the sample rate to the video rate: a late frame
becomes a late sample, and the 10 Hz rate, which works well now, would
degrade.

Here instead a thread reads continuously and keeps the LATEST frame in a
cache together with its acquisition time. The loop takes whatever is
there, never waiting, and also records how old it is. That age ends up in
the dataset and is a quality measure: if an episode has images half a
second old, the model would see a scene that no longer matches the
recorded action.

Channel measurements
-----------------
With the stream on, angle readings go from 16.4 to 15.9 per second, i.e.
a 3% drop, with no failures, and the same at 360, 540 and 720p. Video and
the synchronous channel do not interfere: images and angles can be
recorded in the same frame.
"""

import logging
import threading
import time

log = logging.getLogger(__name__)

try:
    from robomaster import camera as camera_mod
except ImportError:      # pragma: no cover - useful for tests without the SDK
    camera_mod = None

RESOLUTIONS = {}
if camera_mod is not None:
    RESOLUTIONS = {
        "360": getattr(camera_mod, "STREAM_360P", None),
        "540": getattr(camera_mod, "STREAM_540P", None),
        "720": getattr(camera_mod, "STREAM_720P", None),
    }

# Default resolution. 540p is the compromise: the channel also handles
# 720, but VLAs resize to a 224-512 side anyway, and at 540 the dataset
# takes about a third of the space.
DEFAULT_RESOLUTION = "540"

# Beyond this age a frame no longer describes the scene at the time of
# the action.
MAX_FRAME_AGE = 0.5


class CameraError(RuntimeError):
    pass


class Camera:
    """Video stream with a cache of the latest frame."""

    def __init__(self, ep_camera):
        self._cam = ep_camera
        self._frame = None
        self._frame_t = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._streaming = False
        self._errors = 0
        self._read_count = 0
        self.resolution = None

    # ------------------------------------------------------------------
    def start(self, resolution=DEFAULT_RESOLUTION, wait=3.0):
        """Start the stream and the background reader."""
        if self._streaming:
            return self

        res = RESOLUTIONS.get(resolution)
        try:
            if res is not None:
                self._cam.start_video_stream(display=False, resolution=res)
            else:
                self._cam.start_video_stream(display=False)
        except Exception as e:
            raise CameraError(f"starting the stream failed: {e}")

        self._streaming = True
        self.resolution = resolution
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        deadline = time.time() + wait
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            self.stop()
            raise CameraError(
                f"no frame within {wait:.0f}s of starting"
            )
        h, w = self._frame.shape[:2]
        log.info("camera started: %dx%d (%sp)", w, h, resolution)
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                img = self._cam.read_cv2_image(strategy="newest", timeout=1)
            except Exception as e:
                self._errors += 1
                if self._errors in (5, 50):
                    log.warning("video read failed %d times",
                                self._errors)
                time.sleep(0.05)
                continue
            if img is None:
                continue
            with self._lock:
                self._frame = img
                self._frame_t = time.time()
                self._read_count += 1

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._streaming:
            try:
                self._cam.stop_video_stream()
            except Exception as e:
                log.warning("stopping the stream failed: %s", e)
            self._streaming = False

    # ------------------------------------------------------------------
    def read(self):
        """Latest frame and its age in seconds.

        Never blocks. Returns (None, None) if there is nothing yet.
        """
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame, time.time() - self._frame_t

    @property
    def is_streaming(self):
        return self._streaming

    @property
    def frame_count(self):
        return self._read_count

    @property
    def error_count(self):
        return self._errors

    def stats(self):
        with self._lock:
            age = (time.time() - self._frame_t) if self._frame_t else None
            shape = self._frame.shape[:2] if self._frame is not None else None
        return {
            "streaming": self._streaming,
            "resolution": self.resolution,
            "frames": self._read_count,
            "errors": self._errors,
            "age": age,
            "shape": shape,
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
