"""MHT with sensor model on the sets of sim/plan.py: modules M2 (setup) and M3 (failure map) of the research plan.

For every scene of a set, the MHT with sensor model (vf/fusion/sensor_aware_mht.py) is run on the reports of all
robots, for every number of kept hypotheses H in --hypotheses and every robot processing order in --orders:
  order 0, 1, ...   the robots in a random order, seeded by the scene and the order index (order 0 is the same
                    order for every H, so H is compared on the same orders)
In simulation the MHT is given exactly the simulator's sensor file: without the two terms added for real data
(viewing-angle outliers, unknown obstacles), with --real-terms to keep them. --clutter-factor multiplies its
false-detection rate ("MHT, more clutter").

Outputs in --out/<set>/:
  scenes.csv          one row per scene, H and order: counts, matches within the cut-off (15 cm), correct,
                      too few / too many boxes, misplaced, run time
  estimates_H<H>_o<k>.jsonl   the boxes of every scene and, for each box, the reports it was built from
                      (indices into the scene's reports as gnn.dataset.build_scenes orders them), for the
                      region-level results of M3-M6
The k-best search of the MHT (branch and bound) is exact by default and exponential only in the worst case
(on the validation set: 0.2 s per scene on average, 1 s at most, with H = 100; 0.5 s and 4 s with H = 1000).
--search-budget caps it if needed; a scene where the cap was reached is marked search_cut (search_cut_pct).
With several sets, --out/summary.csv holds one row per set, H and order, and --choose-h writes table_M2.csv
and fig_M2-1 and prints the chosen H: the smallest H whose correct-scene rate (mean over the orders) is within
0.5 points of the largest H.

    python -m pipeline.plan_mht --sets data/plan/val/ --hypotheses 100 300 1000 --orders 3 --choose-h --out results/plan/M2/
    python -m pipeline.plan_mht --sets data/plan/G0_nominal_circle/ data/plan/E1_kf4_circle/ --hypotheses 300 --out results/plan/M3/
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

from gnn.dataset import build_scenes
from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics
from vf.fusion import sensor_aware_mht
from vf.fusion.localization_methods import run_method
from vf.fusion.sensor_model import SensorModel

ARENA = (-3.0, 3.0, -3.0, 3.0)        # the 6 m x 6 m arena of the plan, world frame
N_PRIOR = 2.0                         # birth prior of the MHT, as in pipeline/evaluate_runs.py


def mht_inputs(scene):
    """reports, poses and meta in the form of vf.fusion.localization_methods.run_method, plus, per robot, the
    scene index of each of its reports (to map the boxes back to the reports)."""
    reports, meta, index = defaultdict(list), defaultdict(list), defaultdict(list)
    poses = {rid: tuple(float(v) for v in p) for rid, p in zip(scene["robots"], scene["poses"])}
    for i, (r, z, v) in enumerate(zip(scene["det_robot"], scene["pos"], scene["view"])):
        rid = scene["robots"][r]
        reports[rid].append(np.asarray(z, float))
        meta[rid].append(dict(view=float(v), cam=poses[rid]))
        index[rid].append(i)
    return dict(reports), poses, dict(meta), dict(index)


def run_scene(scene, H, order_index, cutoff, sensor, max_nodes=None):
    reports, poses, meta, index = mht_inputs(scene)
    robots = sorted(poses)
    rng = np.random.default_rng(zlib.crc32(f"{scene['run']}/{order_index}".encode()))
    order = [robots[k] for k in rng.permutation(len(robots))]
    t0 = time.perf_counter()
    if reports:
        res = run_method_h(order, reports, poses, sensor, meta, H, max_nodes)
        boxes = [np.array(o) for o in res["objects"]]
        members = [[index[rid][k] for rid, k in hist] for hist in res["extra"]["reports_per_object"]]
        nodes, cut = res["extra"]["search_nodes"], res["extra"]["search_cut"]
    else:
        boxes, members, nodes, cut = [], [], 0, 0
    seconds = time.perf_counter() - t0
    m = scene_metrics(boxes, list(scene["truth"]), cutoff, 2.0)
    correct = int(m["dn"] == 0 and m["n_matched"] == m["n_true"])
    row = dict(run=scene["run"], H=H, order=order_index, n_robots=len(robots), n_reports=len(scene["pos"]),
               n_true=m["n_true"], n_est=m["n_est"], n_matched=m["n_matched"], correct=correct,
               too_few=int(m["dn"] < 0), too_many=int(m["dn"] > 0),
               misplaced=int(m["dn"] == 0 and not correct), rmse=m["rmse"], seconds=round(seconds, 4),
               search_nodes=nodes, search_cut=cut)
    est = dict(run=scene["run"], boxes=[[round(float(v), 4) for v in b] for b in boxes], members=members)
    return row, est


def run_method_h(order, reports, poses, sensor, meta, H, max_nodes=None):
    sensor.set_coverage([poses.get(r) for r in order])
    return sensor_aware_mht.mht_sensor(order, reports, poses, sensor, meta=meta, n_hypotheses=H, max_nodes=max_nodes)


_SENSOR = None


def _init(model_path, real_terms, clutter_factor):
    global _SENSOR
    sensor_aware_mht.set_model(model_path, real_terms=real_terms, clutter_factor=clutter_factor)
    _SENSOR = SensorModel(n_prior=N_PRIOR, arena=ARENA)


def _work(task):
    scenes, H, k, cutoff, max_nodes = task
    return [run_scene(s, H, k, cutoff, _SENSOR, max_nodes) for s in scenes]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, set_name):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["H"], r["order"])].append(r)
    for (H, k), rs in sorted(groups.items()):
        n = len(rs)
        out.append(dict(set=set_name, H=H, order=k, scenes=n,
                        correct_pct=round(100 * sum(r["correct"] for r in rs) / n, 2),
                        too_few_pct=round(100 * sum(r["too_few"] for r in rs) / n, 2),
                        too_many_pct=round(100 * sum(r["too_many"] for r in rs) / n, 2),
                        misplaced_pct=round(100 * sum(r["misplaced"] for r in rs) / n, 2),
                        search_cut_pct=round(100 * sum(r["search_cut"] for r in rs) / n, 2),
                        seconds_per_scene=round(sum(r["seconds"] for r in rs) / n, 4)))
    return out


def choose_h(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_h = defaultdict(list)
    for r in summary:
        by_h[r["H"]].append(r)
    hs = sorted(by_h)
    mean = {h: float(np.mean([r["correct_pct"] for r in by_h[h]])) for h in hs}
    secs = {h: float(np.mean([r["seconds_per_scene"] for r in by_h[h]])) for h in hs}
    spread = {h: float(np.ptp([r["correct_pct"] for r in by_h[h]])) for h in hs}
    chosen = next(h for h in hs if abs(mean[h] - mean[hs[-1]]) < 0.5)
    rows = [dict(H=h, correct_pct_mean=round(mean[h], 2), orders_spread_pts=round(spread[h], 2),
                 seconds_per_scene=round(secs[h], 4), chosen=int(h == chosen)) for h in hs]
    write_csv(os.path.join(out, "table_M2.csv"), rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    orders = sorted({r["order"] for r in summary})
    for k in orders:
        ax.plot(hs, [next(r["correct_pct"] for r in by_h[h] if r["order"] == k) for h in hs], "o-",
                label=f"order {k + 1}")
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set(xlabel="kept hypotheses H", ylabel="correct scenes [%]",
           title=f"Fig. M2-1: MHT on the validation set (chosen H = {chosen})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_M2-1_hypotheses.png"), dpi=120)
    plt.close(fig)
    for r in rows:
        print(f"  H {r['H']:5d}: correct {r['correct_pct_mean']:6.2f}% (orders differ by {r['orders_spread_pts']:.2f} "
              f"points), {r['seconds_per_scene']:.3f} s per scene{'  <- chosen' if r['chosen'] else ''}")
    if spread[chosen] > 1.0:
        print("  the orders disagree by more than 1 point: average over the orders in the later modules")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="set folders of sim/plan.py")
    ap.add_argument("--model", default="config/sensor_model.json")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    ap.add_argument("--hypotheses", type=int, nargs="+", default=[300])
    ap.add_argument("--orders", type=int, default=1, help="robot processing orders per scene")
    ap.add_argument("--real-terms", action="store_true",
                    help="keep the viewing-angle outliers and the unknown obstacles of the real-data MHT")
    ap.add_argument("--clutter-factor", type=float, default=1.0, help="false-detection rate x this ('more clutter')")
    ap.add_argument("--max-scenes", type=int, default=None, help="only the first N scenes of each set (quick tests)")
    ap.add_argument("--cutoff", type=float, default=MATCH_CUTOFF_M)
    ap.add_argument("--search-budget", type=int, default=0,
                    help="cap of the k-best search, nodes per kept hypothesis (budget = this x H per hypothesis and "
                         "robot); 0 = exact search (default)")
    ap.add_argument("--choose-h", action="store_true", help="M2: table, figure and the chosen H")
    ap.add_argument("--out", default="results/plan/M2/")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if sys.platform == "win32":
        a.jobs = min(a.jobs, 60)
    os.makedirs(a.out, exist_ok=True)
    summary = []
    with multiprocessing.Pool(a.jobs, initializer=_init,
                              initargs=(a.model, a.real_terms, a.clutter_factor)) as pool:
        for set_dir in a.sets:
            name = os.path.basename(os.path.normpath(set_dir))
            scenes = build_scenes(set_dir, os.path.join(set_dir, "ground_truth.json"), a.sensor_calib)
            if a.max_scenes:
                scenes = scenes[:a.max_scenes]
            for s in scenes:                    # only what the workers need
                for key in ("label", "layout", "round", "distance", "bearing"):
                    s.pop(key, None)
            set_out = os.path.join(a.out, name)
            os.makedirs(set_out, exist_ok=True)
            rows = []
            t0 = time.time()
            for H in a.hypotheses:
                for k in range(a.orders):
                    chunk = max(1, len(scenes) // (4 * a.jobs))
                    budget = a.search_budget * H if a.search_budget else None
                    tasks = [(scenes[i:i + chunk], H, k, a.cutoff, budget) for i in range(0, len(scenes), chunk)]
                    ests = []
                    for part in pool.imap(_work, tasks):
                        for row, est in part:
                            rows.append(row)
                            ests.append(est)
                    with open(os.path.join(set_out, f"estimates_H{H}_o{k}.jsonl"), "w") as f:
                        for e in ests:
                            f.write(json.dumps(e) + "\n")
                    done = [r for r in rows if r["H"] == H and r["order"] == k]
                    print(f"  {name} H {H} order {k}: correct {100 * np.mean([r['correct'] for r in done]):.1f}%  "
                          f"({time.time() - t0:.0f} s)", flush=True)
            write_csv(os.path.join(set_out, "scenes.csv"), rows)
            summary += summarize(rows, name)
    write_csv(os.path.join(a.out, "summary.csv"), summary)
    if a.choose_h:
        choose_h(summary, a.out)


if __name__ == "__main__":
    main()
