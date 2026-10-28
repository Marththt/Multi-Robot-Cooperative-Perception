"""Client of a remote YOLO server: the same .detect(img) -> list[(class_name, confidence, bbox)]
interface as the other detectors, over a socket.

The robot SDK needs Python <= 3.8 while YOLO (ultralytics) needs Python >= 3.10, so YOLO runs in a
separate process and this client depends only on OpenCV (to encode the frame as JPEG) and the
standard library. It is the alternative to ArUco for objects without markers
(acquisition/static_trial.py --detector yolo).

Wire protocol: 4-byte big-endian length + JPEG bytes out; 4-byte big-endian length + UTF-8 JSON list
of [class, confidence, bbox] back.
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
    """Connects to the YOLO server. Reconnects on its own if the connection
    drops (e.g. server restarted); if it stays unreachable, .detect()
    returns an empty list for that round instead of stopping the capture."""

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
