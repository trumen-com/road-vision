#!/usr/bin/env python3
"""Train the learned risk model (gradient-boosted trees over 2 s of engine features) and export it for the browser.

  npx tsx eval/score.ts --dets ../guard_data/dets_train --out /tmp/x.csv --features results/train_agg.jsonl --agg 1 --no-model 1
  python eval/train_model.py --features results/train_agg.jsonl --data ../guard_data/nexar --split train \
      --out app/public/models/risk_model.json

The model is fitted on the Nexar TRAINING split only; the test split is used once, for the reported numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from train_calibrator import labels_for

FEATS = ["maxLooming", "minTtcPath", "minTtcSoon", "ttcOfBiggest", "maxProxPath", "maxWidthPath", "maxHeightPath",
         "nPath", "vulnPath", "vulnNear", "minAbsOffset", "egoShift", "nTracks", "heuristic", "tauDotPath", "minTauDot"]
AGGS = ["last", "max1", "min1", "mean2", "delta"]
NAMES = [f"{f}.{a}" for f in FEATS for a in AGGS]


def make():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
                                          min_samples_leaf=40, class_weight="balanced", random_state=0)


def export(m: HistGradientBoostingClassifier) -> dict:
    trees = []
    for it in m._predictors:
        nodes = it[0].nodes
        trees.append({
            "f": [int(n["feature_idx"]) for n in nodes],
            "thr": [float(n["num_threshold"]) for n in nodes],
            "left": [int(n["left"]) if not n["is_leaf"] else -1 for n in nodes],
            "right": [int(n["right"]) if not n["is_leaf"] else -1 for n in nodes],
            "value": [float(n["value"]) for n in nodes],
        })
    return {"kind": "hgb", "baseline": float(m._baseline_prediction.ravel()[0]), "trees": trees, "features": NAMES}


def predict_exported(model: dict, X: np.ndarray) -> np.ndarray:
    out = np.full(len(X), model["baseline"])
    for t in model["trees"]:
        for r in range(len(X)):
            i = 0
            while t["left"][i] >= 0:
                i = t["left"][i] if X[r, t["f"][i]] <= t["thr"][i] else t["right"][i]
            out[r] += t["value"][i]
    return 1 / (1 + np.exp(-out))


def remap(model: dict, p: np.ndarray) -> np.ndarray:
    k = model.get("remap")
    if not k:
        return p
    xs, ys = zip(*k)
    return np.interp(p, xs, ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--labels", default="task", choices=["task", "in-scene"],
                    help="task: negatives only from normal-driving videos (the Nexar task); in-scene: also early frames of crash videos")
    ap.add_argument("--folds-out", default=None, help="write 5 out-of-fold models + clip->fold map (for threshold choice)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.features) if l.strip()]
    y, keep = labels_for(rows, Path(a.data), a.split, in_scene_negatives=(a.labels == "in-scene"))
    X = np.array([rows[i]["x"] for i in keep], np.float64)
    groups = np.array([rows[i]["clip"] for i in keep])
    print(f"frames {len(y)}  positives {int(y.sum())}  clips {len(set(groups))}")
    if a.cv:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            oof[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        print(f"5-fold by clip: frame AUC {roc_auc_score(y, oof):.3f}  AP {average_precision_score(y, oof):.3f}")
    if a.folds_out:
        Path(a.folds_out).mkdir(parents=True, exist_ok=True)
        fold_of = {}
        for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups)):
            mk = export(make().fit(X[tr], y[tr]))
            Path(a.folds_out, f"fold{k}.json").write_text(json.dumps(mk))
            for g in set(groups[te]):
                fold_of[g] = k
        Path(a.folds_out, "folds.json").write_text(json.dumps(fold_of))
        print("wrote out-of-fold models to", a.folds_out)
    m = make().fit(X, y)
    model = export(m)
    model["labels"] = a.labels
    model["trained_on"] = f"Nexar {a.split} split: {len(set(groups))} clips, {len(y)} labelled frames, {int(y.sum())} positive"
    # parity check: our tree walker (same logic as engine/model.ts) must match sklearn
    idx = np.random.default_rng(0).choice(len(X), min(500, len(X)), replace=False)
    diff = np.abs(predict_exported(model, X[idx]) - m.predict_proba(X[idx])[:, 1]).max()
    print(f"export parity: max |diff| = {diff:.2e}")
    assert diff < 1e-6, "exported trees do not reproduce sklearn"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(model))
        print("wrote", a.out, f"({len(model['trees'])} trees)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
