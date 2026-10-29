"""The sensor model of the simulator. Every value is either a property of the setup or fitted on real runs
by sim/calibrate.py, which writes it as JSON (sensor_model.json).

Positions are modelled in the CALIBRATED domain, i.e. after the range / bearing correction of pipeline/calibrate.py,
which is what the fusion methods see. The writer converts them back to raw distance / bearing with the
same correction, so pipeline/evaluate_runs.py --sensor-calib and gnn/train.py --sensor-calib read simulated runs
exactly as real ones.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class SensorModel:
    # --- setup (camera and survey settings)
    hfov_deg: float = 88.0                  # horizontal field of view
    r_min_m: float = 0.15                   # closer boxes are not detected
    r_max_m: float = 3.0                    # farther boxes are not detected

    # --- calibration of the real pipeline (pipeline/calibrate.py), used to write raw values
    range_fit: List[float] = field(default_factory=lambda: [1.0, 0.0])      # d_raw = A * d + B
    bearing_fit: List[float] = field(default_factory=lambda: [1.0, 0.0])    # b_raw = k * b + b0
    camera_offset_m: float = 0.0            # camera ahead of the body pivot

    # --- detection: P(box reported) = sigmoid(w . [1, a, a^2, d, occluded]) for its best face,
    #     a = viewing angle of that face [rad], d = distance [m], occluded = 1 if the line of sight
    #     to its marker crosses another box or an obstacle
    detect_w: List[float] = field(default_factory=lambda: [3.0, 0.0, -3.0, -0.3, -3.5])
    two_face_q: float = 0.02                # second face also reported with prob q * P(second face)
    view_noise_sd_rad: float = 0.1          # noise of the reported viewing angle

    # --- position error of a reported box, relative to its centre, in the calibrated domain;
    #     a is the viewing angle of the reported face, d the distance to the box centre
    along_mean: List[float] = field(default_factory=lambda: [0.035, 0.0, -0.045])  # c0 + c1 d + c2 cos(a)
    along_sd: List[float] = field(default_factory=lambda: [0.0, 0.02])            # s0 + s1 d
    across_sd: List[float] = field(default_factory=lambda: [0.045, 0.004])        # t0 + t1 d
    bias_sd_along_rel: float = 0.02         # per robot and run: range scale error (m per m)
    bias_sd_across_rel: float = 0.015       # per robot and run: bearing offset (m per m = rad)

    # --- lone false detections (markers seen where there is no box)
    false_rate: float = 0.1                 # mean number per capture (Poisson)
    false_near_share: float = 0.2           # share with raw distance in false_near_range_m
    false_near_range_m: List[float] = field(default_factory=lambda: [1.6, 3.0])
    false_far_range_m: List[float] = field(default_factory=lambda: [4.4, 7.4])
    false_edge_share: float = 0.5           # share at the edge of the field of view
    false_edge_band_deg: List[float] = field(default_factory=lambda: [38.0, 44.0])
    false_within_range: bool = False       # drop the false detections beyond r_max (sensing-range scenarios)

    # --- provenance
    source: dict = field(default_factory=dict)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)