"""Simulated twins of the real scenes: the scenes of data/bev_real/<set> rendered by imgsim/render.py with the
same robot poses, pitch and boxes (box orientations are not recorded: drawn at random). The twin set has the
format of bevnet/data.py, so a model can be tested on the real frames and on their twins with the same command:
if it finds the boxes in the twins but not in the real frames, what fails is the look of the images, not the
geometry. --figure puts real frames and twin frames side by side (fig_REAL-2.png).

    python -m bevnet.twin --real data/bev_real/real_k0 --out data/bev_real_twin/ --figure
    python -m bevnet.train --eval-only --out results/bev_obst/ --tests data/bev_real_twin/twin_k0 --results-name twin
"""

import argparse
import json
import os
import pickle
from dataclasses import asdict

import cv2
import numpy as np

from bevnet.data import CALIB
from imgsim.render import Appearance, Camera, Renderer
from sim.scene import Capture, Scene


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default="data/bev_real/real_k0")
    ap.add_argument("--out", default="data/bev_real_twin/")
    ap.add_argument("--scale", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(a.real, "shard_0000.pkl"), "rb") as f:
        real = pickle.load(f)
    app = Appearance()
    renderer = Renderer(CALIB["P0"], app, seed=a.seed, scale=a.scale)
    cam = Camera(CALIB["P0"], scale=a.scale)
    rng = np.random.default_rng(a.seed)
    twins = []
    for s in real:
        boxes = np.asarray(s["boxes"], float)
        scene = Scene(layout_id=s["run"].split("_")[0], run_id=s["run"], boxes=boxes,
                      thetas=rng.uniform(0, 2 * np.pi, len(boxes)), obstacles=[],
                      captures=[Capture(round=0, robot=q, pose=tuple(map(float, p)))
                                for q, p in zip(s["robots"], s["poses"])])
        style = renderer.scene_style(rng)
        jpgs = []
        for k, p in enumerate(s["pitch"]):
            app.pitch_deg = [float(np.degrees(p))] * 2
            app.roll_deg = [0.0, 0.0]
            img, _, _ = renderer.render(scene, k, rng, style)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jpgs.append(buf.tobytes())
        t = {k: v for k, v in s.items() if k not in ("K", "D", "jpgs")}
        t.update(jpgs=jpgs, family="twin")
        twins.append(t)
    name = os.path.basename(os.path.normpath(a.real)).replace("real", "twin")
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "shard_0000.pkl"), "wb") as f:
        pickle.dump(twins, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(set=name, scenario="twin", level="", scenes=len(twins), camera="P0", K=cam.K.tolist(),
                       D=cam.D.tolist(), size=[cam.W, cam.H], scale=a.scale, appearance=asdict(app)), f, indent=2)
    print(f"  {name}: {len(twins)} scenes -> {out}")
    if a.figure:
        rows = []
        for s, t in list(zip(real, twins))[::7][:6]:
            for src in (s, t):
                rows.append(np.hstack([cv2.imdecode(np.frombuffer(j, np.uint8), 1) for j in src["jpgs"]]))
        cv2.imwrite(os.path.join(a.out, "fig_REAL-2.png"), np.vstack(rows))
        print(f"  figure: {os.path.join(a.out, 'fig_REAL-2.png')}")


if __name__ == "__main__":
    main()
