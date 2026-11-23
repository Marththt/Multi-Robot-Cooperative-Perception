"""Infrared (ToF) distance sensors of the RoboMaster EP, with a background subscription.

The SDK pushes the distances of the (up to 4) ToF sensors at a fixed rate: robot.sensor.sub_distance(freq,
callback), the callback gets [d1, d2, d3, d4] in mm, in the order of the sensor ids set in the RoboMaster app.
IRSensors keeps the latest samples of every sensor with their arrival time, so a caller can ask for the median
of n FRESH samples (taken after the request) of one sensor: measure(index, n).

Each sensor measures along its own axis only, to the nearest surface in a narrow cone; which index is the front
one, and where it sits on the robot, is checked with tools/ir_check.py and given to acquisition/ir_ranging.py.

    from robomaster_sdk import Robot
    from robomaster_sdk.ir import IRSensors
    with Robot("P0") as r:
        ir = IRSensors.from_robot(r).start()
        print(ir.measure(0, n=10))      # {'m': 1.234, 'sd_m': 0.004, 'n': 10, 'valid': True, ...}
        ir.stop()

The Robot class (robot.py) is not changed: from_robot() takes the sensor module of the DJI robot object it wraps.
The first message of the SDK is kept as it arrived (first_raw), to check its format on a new firmware.
"""

import collections
import threading
import time

import numpy as np

FREQS = (1, 5, 10, 20, 50)          # rates accepted by sub_distance [Hz]
MIN_M, MAX_M = 0.10, 10.0           # outside: no echo / not valid


class IRSensors:
    def __init__(self, sensor, freq=20, history=200, min_m=MIN_M, max_m=MAX_M):
        if freq not in FREQS:
            raise ValueError(f"freq must be one of {FREQS}")
        self.sensor, self.freq = sensor, freq
        self.min_m, self.max_m = min_m, max_m
        self.samples = collections.deque(maxlen=history)    # (t, [m, m, m, m])
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = False
        self.first_raw = None

    @classmethod
    def from_robot(cls, robot, **kw):
        """From a connected robomaster_sdk.Robot (or a DJI robot object)."""
        ep = getattr(robot, "_ep", robot)
        if ep is None or not hasattr(ep, "sensor"):
            raise RuntimeError("robot not connected, or this SDK has no distance sensor module (robot.sensor)")
        return cls(ep.sensor, **kw)

    def _on_data(self, info):
        t = time.time()
        if self.first_raw is None:
            self.first_raw = repr(info)
        try:
            values = [float(v) / 1000.0 for v in list(info)[:4]]
        except (TypeError, ValueError):                 # unexpected format: kept in first_raw, sample dropped
            return
        with self.cond:
            self.samples.append((t, values))
            self.cond.notify_all()

    def start(self):
        self.sensor.sub_distance(freq=self.freq, callback=self._on_data)
        self.running = True
        return self

    def stop(self):
        if self.running:
            try:
                self.sensor.unsub_distance()
            finally:
                self.running = False

    def latest(self):
        """(time, [m, ...]) of the last sample, or None."""
        with self.lock:
            return self.samples[-1] if self.samples else None

    def valid(self, m):
        return self.min_m <= m <= self.max_m

    def measure(self, index, n=10, timeout_s=3.0):
        """Median of n samples of sensor `index` (0-based) arrived after this call. Returns a dict: m (median of
        the valid samples, None if none), sd_m, n (valid samples), n_total, valid (at least half valid), samples."""
        t0 = time.time()
        got = []
        with self.cond:
            while len(got) < n and time.time() - t0 < timeout_s:
                self.cond.wait(timeout=timeout_s)
                got = [v[index] for t, v in self.samples if t > t0 and index < len(v)]
        ok = [m for m in got if self.valid(m)]
        return dict(m=float(np.median(ok)) if ok else None, sd_m=float(np.std(ok)) if len(ok) > 1 else None,
                    n=len(ok), n_total=len(got), valid=len(ok) >= max(1, len(got) // 2), samples=got)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
