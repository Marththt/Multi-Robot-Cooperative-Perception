"""Client for yolo_server.py: same interface as the local YOLO detector
(yolo_detector.py) but over a socket, to be used inside patrol.py.

Why it exists
-------------
patrol.py imports robomaster (Python <=3.8). The local YOLO detector in
yolo_detector.py imports ultralytics (Python >=3.10). The two cannot live
in the same process/venv. This module never imports ultralytics/torch:
only opencv-python (to encode the frame as JPEG) and the standard library
socket, so it can live happily in the <=3.8 venv of patrol.py.

It exposes the same .detect(img) -> list[(class_name, confidence, bbox)]
that perception.ObjectDetector expects, so patrol.py does not need to
know whether the detector is local or remote.

Wire protocol (unchanged): 4-byte big-endian length + JPEG bytes out;
4-byte big-endian length + UTF-8 JSON list of [class, confidence, bbox]
back. No JSON object keys are involved.
"""

import json
import socket
import struct

import cv2


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("yolo_server closed the connection")
        buf += chunk
    return buf


class RemoteYoloDetector:
    """Connects to yolo_server.py. Reconnects on its own if the connection
    drops (e.g. server restarted); if it stays unreachable, .detect()
    returns an empty list for that round instead of crashing patrol.py --
    better to lose one detection frame than the whole patrol."""

    def __init__(self, host="127.0.0.1", port=8765, timeout=2.0,
                 jpeg_quality=90):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self._sock = None

    def _connect(self):
        self._sock = socket.create_connection(
            (self.host, self.port), timeout=self.timeout)

    def detect(self, img):
        if img is None:
            return []
        ok, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return []
        data = buf.tobytes()

        for attempt in range(2):  # 1 reconnection if the first one fails
            try:
                if self._sock is None:
                    self._connect()
                self._sock.sendall(struct.pack(">I", len(data)) + data)
                header = _recv_exact(self._sock, 4)
                (length,) = struct.unpack(">I", header)
                response = _recv_exact(self._sock, length)
                predictions = json.loads(response.decode("utf-8"))
                return [(c, conf, tuple(bbox)) for c, conf, bbox in predictions]
            except (ConnectionError, OSError, socket.timeout):
                self._sock = None
                if attempt == 1:
                    return []
        return []

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
