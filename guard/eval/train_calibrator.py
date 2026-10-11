#!/usr/bin/env python3
"""Learn the risk model (logistic regression over the engine's frame features) and export it for the browser.

  # train on the Nexar TRAINING split windows, export for the app (the test split is never used for fitting)
  python eval/train_calibrator.py --features results/train_feats.jsonl --data ../guard_data/nexar --split train \
      --out app/public/models/calibrator.json

  # quick estimate on any feature dump with 5-fold cross-validation by clip (used only during development)
  python eval/train_calibrator.py --features feats.jsonl --data ../guard_data/nexar --split test-public --cv

Labels
  train split : positive frames lie 0.3-2.0 s before the annotated collision / near miss; negatives are all frames of
                negative videos plus frames more than 6 s before the event in positive videos.
  test-public : positive frames are the last 1.0 s of positive clips (they end 0.5-1.5 s before the event).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

FEATURES = ["maxLooming", "minTtcPath", "minTtcSoon", "ttcOfBiggest", "maxProxPath", "maxWidthPath", "maxHeightPath",
            "nPath", "vulnPath", "vulnNear", "minAbsOffset", "egoShift", "nTracks", "heuristic"]


def load(features: str):
    rows = [json.loads(l) for l in open(features) if l.strip()]
    return rows


def labels_for(rows, data: Path, split: str, in_scene_negatives: bool = True):
    y, keep = [], []
    if split == "train":
        ev = {}
        for lab in ("positive", "negative"):
            for r in csv.DictReader(open(data / "train" / lab / "metadata.csv")):
                ev[r["file_name"].replace(".mp4", "")] = float(r["time_of_event"]) if lab == "positive" and r["time_of_event"] else None
        for i, r in enumerate(rows):
            te = ev.get(r["clip"], "missing")
            if te == "missing":
                continue
            if te is None:
                y.append(0); keep.append(i)
            elif te - 2.0 <= r["t"] <= te - 0.3:
                y.append(1); keep.append(i)
            elif in_scene_negatives and r["t"] < te - 6.0:
                y.append(0); keep.append(i)
    else:
        meta = {}
        for lab in ("positive", "negative"):
            for r in csv.DictReader(open(data / split / lab / "metadata.csv")):
                meta[r["file_name"].replace(".mp4", "")] = 1 if lab == "positive" else 0
        t_end = {}
        for r in rows:
            t_end[r["clip"]] = max(t_end.get(r["clip"], 0), r["t"])
        for i, r in enumerate(rows):
            lab = meta.get(r["clip"])
            if lab is None:
                continue
            if lab == 0:
                y.append(0); keep.append(i)
            elif r["t"] >= t_end[r["clip"]] - 1.0:
                y.append(1); keep.append(i)
    return np.array(y), keep


def matrix(rows, idx):
    return np.array([[float(rows[i][f]) for f in FEATURES] for i in idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--C", type=float, default=0.5)
    a = ap.parse_args()
    rows = load(a.features)
    y, keep = labels_for(rows, Path(a.data), a.split)
    X = matrix(rows, keep)
    groups = np.array([rows[i]["clip"] for i in keep])
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mean) / scale
    print(f"frames {len(y)}  positives {int(y.sum())}  clips {len(set(groups))}")
    if a.cv:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(Xs, y, groups):
            m = LogisticRegression(C=a.C, class_weight="balanced", max_iter=2000).fit(Xs[tr], y[tr])
            oof[te] = m.predict_proba(Xs[te])[:, 1]
        print(f"frame-level  AUC {roc_auc_score(y, oof):.3f}  AP {average_precision_score(y, oof):.3f}")
        # clip-level: score at the last frame of each clip, as in the Nexar protocol
        last = {}
        for k, i in enumerate(keep):
            c = rows[i]["clip"]
            if c not in last or rows[i]["t"] > rows[last[c][0]]["t"]:
                last[c] = (i, k)
        cy = np.array([y[k] for _, k in last.values()]); cs = np.array([oof[k] for _, k in last.values()])
        ch = np.array([rows[i]["heuristic"] for i, _ in last.values()])
        print(f"clip-level   AUC {roc_auc_score(cy, cs):.3f}  AP {average_precision_score(cy, cs):.3f}   (hand-set heuristic AP {average_precision_score(cy, ch):.3f})")
    m = LogisticRegression(C=a.C, class_weight="balanced", max_iter=2000).fit(Xs, y)
    for f, c in sorted(zip(FEATURES, m.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {f:<14} {c:+.3f}")
    if a.out:
        # class_weight=balanced shifts the intercept; re-centre so that 0.5 means "as likely as not" at the data's base rate
        cal = {"names": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "coef": m.coef_[0].tolist(),
               "intercept": float(m.intercept_[0]), "trained_on": f"Nexar {a.split} split, {len(set(groups))} clips, {len(y)} frames"}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(cal, indent=1))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
