"""Module M3 of the research plan: the MHT failure map. Which special cases exist at all?

Reads the scenes.csv that pipeline.plan_mht wrote for the test sets of sim/plan.py (one folder per set,
<scenario>_<level>_<family>, under --results) and writes, in --out:
  table_M3.csv     per scenario, level, family and H: correct, too few, too many, misplaced [%] (a scene with
                   the right number of boxes but one more than 15 cm off is "misplaced"), with the change of the
                   correct share from G0 of the same family
  kept_cases.csv   the special cases (E and S) and their decision: kept if MHT loses at least --drop points
                   from G0 at the highest level, in either family (the plan's rule)
  fig_M3_<ID>.png  one figure per scenario: correct, too few and too many against the level, both families
                   (S1: one line per H)
G and V are expected to stay high and flat; a drop there is printed as a warning, since it goes against the
claim of the plan and must be understood first.
With --data (the set folders of sim/plan.py) the table also gives the share of scenes in which every true box
was reported by at least one robot (no method can find a box nobody reported) and MHT's correct share on those
scenes only, which separates the boxes the sensor never delivered from the association errors.

    python -m pipeline.plan_failure_map --results results/plan/M3/ --out results/plan/M3/ --data data/plan/
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from sim.plan import SCENARIOS

LEVEL_VALUE = {}          # (scenario, level name) -> number on the x axis
for sid, levels in SCENARIOS.items():
    for name, changes in levels:
        v = next(iter(changes.values()), None) if changes else 0
        if sid == "S1" or sid == "V1":
            v = changes["n_boxes"][0]
        elif isinstance(v, list):
            v = v[0]
        LEVEL_VALUE[(sid, name)] = (v * 100 if sid == "G1" else v)
X_LABEL = dict(G0="", G1="box spacing s_min [cm]", G2="robots N", E1="false-detection multiplier k_f",
               E2="two-face multiplier k_2", E3="robots hidden per box h", S1="boxes B (spacing 20 cm)",
               V1="boxes B (disc 1.0 m)", V2="robots N")


def reported_all(data_dir, name, sensor_calib):
    """{run: True if every true box has at least one report} of one set, or None without its data."""
    import json
    from gnn.dataset import build_scenes
    d = os.path.join(data_dir, name)
    if not os.path.isdir(d):
        return None
    with open(os.path.join(d, "ground_truth.json")) as f:
        truth = json.load(f)
    seen = {s["run"]: {int(l) for l in s["label"] if l >= 0}
            for s in build_scenes(d, os.path.join(d, "ground_truth.json"), sensor_calib)}
    return {run: len(seen.get(run, ())) == len(boxes) for run, boxes in truth.items() if run.startswith(name + "_")}


def read(results, data_dir=None, sensor_calib=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(results, "*", "scenes.csv"))):
        name = os.path.basename(os.path.dirname(path))
        parts = name.split("_")
        if len(parts) != 3 or parts[0] not in SCENARIOS:
            continue
        sid, level, family = parts
        full = reported_all(data_dir, name, sensor_calib) if data_dir else None
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(dict(scenario=sid, level=level, family=family, H=int(r["H"]), order=int(r["order"]),
                                 correct=int(r["correct"]), too_few=int(r["too_few"]), too_many=int(r["too_many"]),
                                 misplaced=int(r["misplaced"]),
                                 all_reported=(int(full[r["run"]]) if full is not None else None)))
    return rows


def table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["level"], r["family"], r["H"])].append(r)
    out = []
    for (sid, level, family, H), rs in groups.items():
        row = dict(scenario=sid, level=level, value=LEVEL_VALUE.get((sid, level), ""), family=family, H=H,
                   scenes=len(rs), **{k: round(100 * float(np.mean([r[k] for r in rs])), 2)
                                      for k in ("correct", "too_few", "too_many", "misplaced")})
        if rs[0]["all_reported"] is not None:
            full = [r for r in rs if r["all_reported"]]
            row["all_boxes_reported"] = round(100 * len(full) / len(rs), 2)
            row["correct_when_all_reported"] = round(100 * float(np.mean([r["correct"] for r in full])), 2) if full else ""
        out.append(row)
    g0 = {(r["family"], r["H"]): r["correct"] for r in out if r["scenario"] == "G0"}
    for r in out:
        ref = g0.get((r["family"], r["H"])) or next((v for (f, h), v in g0.items() if f == r["family"]), None)
        r["change_from_G0"] = round(r["correct"] - ref, 2) if ref is not None else ""
    order = {sid: k for k, sid in enumerate(SCENARIOS)}
    out.sort(key=lambda r: (order[r["scenario"]], r["family"], r["H"], float(r["value"] or 0)))
    return out


def figure(sid, rows, out, plt):
    rs = [r for r in rows if r["scenario"] == sid]
    if not rs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, family in zip(ax, ("circle", "random")):
        fr = [r for r in rs if r["family"] == family]
        for H in sorted({r["H"] for r in fr}):
            pts = sorted((r for r in fr if r["H"] == H), key=lambda r: float(r["value"] or 0))
            x = [float(r["value"] or 0) for r in pts]
            tag = f", H = {H}" if len({r['H'] for r in fr}) > 1 else ""
            a.plot(x, [r["correct"] for r in pts], "o-", label=f"MHT correct{tag}")
            if not tag or H == min(r["H"] for r in fr):
                a.plot(x, [r["too_few"] for r in pts], "s--", label="too few boxes")
                a.plot(x, [r["too_many"] for r in pts], "^--", label="too many boxes")
        a.set(title=f"{sid}, {family} family", xlabel=X_LABEL.get(sid, "level"), ylim=(0, 100))
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("scenes [%]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_M3_{sid}.png"), dpi=120)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/plan/M3/", help="folder with one sub-folder per test set")
    ap.add_argument("--out", default="results/plan/M3/")
    ap.add_argument("--drop", type=float, default=5.0, help="points lost from G0 for a special case to be kept")
    ap.add_argument("--data", default=None, help="set folders of sim/plan.py (data/plan/): adds the reported-box columns")
    ap.add_argument("--sensor-calib", default="config/sensor_calib.json")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = table(read(a.results, a.data, a.sensor_calib))
    if not rows:
        raise SystemExit(f"no test-set results under {a.results}")
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "table_M3.csv"), rows)
    for sid in SCENARIOS:
        if sid != "G0":
            figure(sid, rows, a.out, plt)
    kept = []
    for sid in SCENARIOS:
        rs = [r for r in rows if r["scenario"] == sid]
        if not rs or sid == "G0":
            continue
        worst = {}
        for family in ("circle", "random"):
            fr = [r for r in rs if r["family"] == family]
            if fr:
                H = min(r["H"] for r in fr)
                top = max((r for r in fr if r["H"] == H), key=lambda r: float(r["value"] or 0))
                worst[family] = top["change_from_G0"]
        drop = -min(worst.values())
        if sid[0] in "ES":
            kept.append(dict(scenario=sid, change_circle=worst.get("circle", ""), change_random=worst.get("random", ""),
                             kept=int(drop >= a.drop)))
        elif drop >= a.drop:
            print(f"WARNING: MHT loses {drop:.1f} points in {sid} (general / validity): look at these scenes first")
    if kept:
        write_csv(os.path.join(a.out, "kept_cases.csv"), kept)
    print(f"{'scenario':<9}{'level':<9}{'family':<8}{'H':>5}{'correct':>9}{'few':>7}{'many':>7}{'misplaced':>10}"
          f"{'vs G0':>8}{'all rep.':>10}{'corr|all':>10}")
    for r in rows:
        print(f"{r['scenario']:<9}{r['level']:<9}{r['family']:<8}{r['H']:>5}{r['correct']:>9.1f}{r['too_few']:>7.1f}"
              f"{r['too_many']:>7.1f}{r['misplaced']:>10.1f}{r['change_from_G0']:>8}"
              f"{r.get('all_boxes_reported', ''):>10}{r.get('correct_when_all_reported', ''):>10}")
    print("special cases kept (MHT loses >= %.0f points from G0 at the highest level):" % a.drop,
          ", ".join(k["scenario"] for k in kept if k["kept"]) or "none")


if __name__ == "__main__":
    main()
