"""Regions of the research plan (Section 2.1) and region-level correctness of a method.

A region links two reports of any robots when their box-centre estimates are at most `link` m apart and is a
connected component of that graph. A true box belongs to the region holding most of its reports (a box no
robot reported belongs to no region); an estimated box belongs to the region holding most of the reports it
was built from. A region is correct when its number of estimates equals its number of true boxes and each of
its true boxes has an estimate within the cut-off (15 cm).
"""

from collections import Counter, defaultdict

import numpy as np

from pipeline.evaluate_runs import MATCH_CUTOFF_M, scene_metrics

LINK_M = 0.20          # region link distance g_r (plan default 0.35, changed in M1: see sim/plan_check.py)


def regions(pos, link=LINK_M):
    """Connected components of the reports closer than `link`: list of index arrays."""
    n = len(pos)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        for i, j in zip(*np.nonzero(np.triu(d <= link, 1))):
            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [np.array(g) for g in groups.values()]


def region_results(scene, estimate, link=LINK_M, cutoff=MATCH_CUTOFF_M):
    """scene: gnn.dataset.build_scenes entry; estimate: {'boxes': [[x, y]], 'members': [[report index]]}.
    Returns (regions, per region dict(n_true, n_est, correct, true=[box index], est=[estimate index]))."""
    regs = regions(scene["pos"], link)
    region_of = np.empty(len(scene["pos"]), int)
    for k, idx in enumerate(regs):
        region_of[idx] = k
    out = [dict(true=[], est=[]) for _ in regs]
    for j in range(len(scene["truth"])):
        mine = region_of[scene["label"] == j]
        if len(mine):
            out[Counter(mine.tolist()).most_common(1)[0][0]]["true"].append(j)
    for e, members in enumerate(estimate["members"]):
        if members:
            out[Counter(region_of[members].tolist()).most_common(1)[0][0]]["est"].append(e)
    for r in out:
        est = [np.asarray(estimate["boxes"][e], float) for e in r["est"]]
        tru = [scene["truth"][j] for j in r["true"]]
        m = scene_metrics(est, tru, cutoff, 2.0)
        r.update(n_true=len(tru), n_est=len(est), correct=int(m["dn"] == 0 and m["n_matched"] == m["n_true"]))
    return regs, out
