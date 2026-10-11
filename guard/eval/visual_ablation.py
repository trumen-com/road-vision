#!/usr/bin/env python3
"""Clip-level check of the optional visual model (MobileNetV3 + temporal head, train/visual_risk.py) on the test split:
alone, and fused 50/50 with the geometric score (weights fixed before testing). Adds 'visual_ablation' to results.json.

  python eval/visual_ablation.py --geo results/test_scores.csv --visual results/visual_test_scores.csv \
      --solution ../guard_data/nexar/solution.csv --results app/public/results/results.json
"""
import argparse, csv, json
import numpy as np
from sklearn.metrics import average_precision_score as ap

a = argparse.ArgumentParser()
for k in ("--geo", "--visual", "--solution", "--results"):
    a.add_argument(k, required=True)
a = a.parse_args()
sol = {r["id"]: r for r in csv.DictReader(open(a.solution))}
g = {r["id"]: float(r["score"]) for r in csv.DictReader(open(a.geo))}
v = {r["id"]: float(r["score"]) for r in csv.DictReader(open(a.visual))}
ids = [i for i in g if i in v and i in sol]
y = np.array([int(sol[i]["target"]) for i in ids]); grp = np.array([sol[i]["group"] for i in ids])

def m(s):
    s = np.asarray(s)
    return float(np.mean([ap(y[grp == G], s[grp == G]) for G in sorted(set(grp))]))

out = {"n": len(ids), "geometric": m([g[i] for i in ids]), "visual": m([v[i] for i in ids]),
       "fused": m([0.5 * g[i] + 0.5 * v[i] for i in ids]), "shipped": False,
       "note": "MobileNetV3-Large frozen + temporal head, trained on CPU on 600 Nexar training videos; fusion lowered mAP, so the app ships without it"}
out["shipped"] = out["fused"] > out["geometric"]
R = json.load(open(a.results)); R["visual_ablation"] = out; json.dump(R, open(a.results, "w"), indent=1)
print(out)
