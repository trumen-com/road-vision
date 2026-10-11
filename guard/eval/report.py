#!/usr/bin/env python3
"""Metrics for Trumen Guard on the Nexar test-public split (our test set: 667 real dashcam clips).

    python eval/report.py --scores results/test_scores.csv --timeline results/test_timeline.json \
        --data ../guard_data/nexar --out app/public/results/results.json [--baseline results/test_baseline.csv]

* Nexar protocol: each positive clip ends 0.5 / 1.0 / 1.5 s before a collision or near miss; negatives are normal
  driving. The score is the risk the engine reports at the last frame (causal). mAP = mean AP over the three horizons
  (the official evaluate_submission.py logic, restricted to the public subset).
* Operating point: a clip "warns" if the alert level is >= 1 at any time in its last 1.5 s. False warnings per hour
  are counted as alert onsets over all negative-clip footage.
* 95% intervals by bootstrap over clips (1000 resamples).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = np.random.default_rng(0)


def load_meta(data: Path) -> dict[str, dict]:
    meta = {}
    for lab in ("positive", "negative"):
        for r in csv.DictReader(open(data / "test-public" / lab / "metadata.csv")):
            meta[r["file_name"].replace(".mp4", "")] = {**r, "target": 1 if lab == "positive" else 0}
    return meta


def nexar_map(ids, y, s, groups) -> float:
    aps = []
    for g in sorted(set(groups)):
        m = np.array([gg == g for gg in groups])
        if y[m].min() == y[m].max():
            continue
        aps.append(average_precision_score(y[m], s[m]))
    return float(np.mean(aps)) if aps else float("nan")


def boot(fn, n, k=1000):
    vals = []
    for _ in range(k):
        idx = RNG.integers(0, n, n)
        try:
            v = fn(idx)
            if not np.isnan(v):
                vals.append(v)
        except ValueError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--label", default="Trumen Guard")
    ap.add_argument("--heuristic", default=None, help="report.py output for the hand-set version (ablation row)")
    ap.add_argument("--shipping", default=None, help="choose_threshold.py output: threshold chosen on the training split")
    ap.add_argument("--speed-ms", type=float, default=None)
    ap.add_argument("--baseline-timeline", default=None, help="per-frame baseline scores, for its warning trade-off curve")
    a = ap.parse_args()

    meta = load_meta(Path(a.data))
    sol = {r["id"]: r for r in csv.DictReader(open(Path(a.data) / "solution.csv")) if r["Usage"] == "Public"}
    scores = {r["id"]: float(r["score"]) for r in csv.DictReader(open(a.scores))}
    tl = json.load(open(a.timeline))
    ids = [i for i in sol if i in scores and i in meta]
    y = np.array([int(sol[i]["target"]) for i in ids])
    s = np.array([scores[i] for i in ids])
    groups = [meta[i]["time_to_accident"] for i in ids]

    res: dict = {"system": a.label, "n_clips": len(ids), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    res["map"] = nexar_map(ids, y, s, groups)
    res["map_ci"] = boot(lambda idx: nexar_map([ids[k] for k in idx], y[idx], s[idx], [groups[k] for k in idx]), len(ids))
    res["auc"] = float(roc_auc_score(y, s))
    res["by_horizon"] = []
    for g in ("0.500", "1.000", "1.500"):
        m = np.array([gg == g for gg in groups])
        if m.sum() == 0 or y[m].min() == y[m].max():
            continue
        res["by_horizon"].append({"horizon_s": float(g), "n": int(m.sum()), "ap": float(average_precision_score(y[m], s[m])),
                                  "ci": boot(lambda idx, m=m: average_precision_score(y[m][idx], s[m][idx]), int(m.sum()))})

    def subset(name, pred):
        m = np.array([pred(meta[i]) for i in ids])
        if m.sum() < 10 or y[m].min() == y[m].max():
            return None
        yy, ss, gg = y[m], s[m], [groups[k] for k in np.where(m)[0]]
        return {"name": name, "n": int(m.sum()), "n_pos": int(yy.sum()),
                "map": nexar_map(None, yy, ss, gg),
                "ci": boot(lambda idx: nexar_map(None, yy[idx], ss[idx], [gg[k] for k in idx]), int(m.sum()))}

    conds = [
        ("Daylight", lambda r: r["light_conditions"] in ("Normal", "Bright")),
        ("Twilight or dark", lambda r: r["light_conditions"] in ("Twilight", "Dark")),
        ("Clear or cloudy", lambda r: r["weather"] in ("Clear", "Cloudy")),
        ("Rain", lambda r: r["weather"] == "Rain"),
        ("Urban", lambda r: r["scene"] == "Urban"),
        ("Highway", lambda r: r["scene"] == "Highway"),
        ("Suburban, rural, other", lambda r: r["scene"] not in ("Urban", "Highway")),
    ]
    res["by_condition"] = [c for c in (subset(n, f) for n, f in conds) if c]

    # operating point from the alert levels
    warned, onsets, neg_seconds = {}, 0, 0.0
    for i in ids:
        frames = tl.get(i, [])
        if not frames:
            continue
        t_end = frames[-1]["t"]
        warned[i] = any(f["l"] >= 1 and f["t"] >= t_end - 1.5 for f in frames)
        if sol[i]["target"] == "0":
            neg_seconds += t_end - frames[0]["t"]
            prev = 0
            for f in frames:
                if f["l"] >= 1 and prev == 0:
                    onsets += 1
                prev = f["l"]
    pos_ids = [i for i in ids if sol[i]["target"] == "1" and i in warned]
    neg_ids = [i for i in ids if sol[i]["target"] == "0" and i in warned]
    res["operating_point"] = {
        "warned_before_event": float(np.mean([warned[i] for i in pos_ids])),
        "warned_before_event_by_horizon": {g: float(np.mean([warned[i] for i in pos_ids if meta[i]["time_to_accident"] == g]))
                                           for g in ("0.500", "1.000", "1.500") if any(meta[i]["time_to_accident"] == g for i in pos_ids)},
        "negative_clips_with_warning": float(np.mean([warned[i] for i in neg_ids])),
        "false_warnings_per_hour": onsets / max(neg_seconds, 1) * 3600,
        "negative_footage_minutes": neg_seconds / 60,
    }
    # trade-off curve: re-run the alert logic (2-frame hold, release below 0.7 x threshold for 0.5 s) at many thresholds
    def simulate(frames, thr):
        lvl, above, below_since, onsets, out = 0, 0, None, 0, []
        for f in frames:
            r = f["r"]
            above = above + 1 if r >= thr else 0
            if lvl == 0 and above >= 2:
                lvl = 1; onsets += 1
            if lvl == 1 and r < 0.7 * thr:
                below_since = f["t"] if below_since is None else below_since
                if f["t"] - below_since > 0.5:
                    lvl = 0
            else:
                below_since = None
            out.append(lvl)
        return out, onsets
    curve = []
    for thr in [round(x, 3) for x in np.linspace(0.2, 0.95, 31)]:
        w, on = [], 0
        for i in pos_ids + neg_ids:
            fr = tl[i]; lv, ons = simulate(fr, thr); t_end = fr[-1]["t"]
            if sol[i]["target"] == "1":
                w.append(any(l >= 1 and f["t"] >= t_end - 1.5 for l, f in zip(lv, fr)))
            else:
                on += ons
        curve.append({"threshold": thr, "warned": float(np.mean(w)), "fa_per_hour": on / max(neg_seconds, 1) * 3600})
    res["tradeoff"] = curve
    if a.heuristic:
        hh = json.load(open(a.heuristic))
        res["heuristic"] = {"map": hh["map"], "auc": hh["auc"]}
    if a.shipping:
        sh = json.load(open(a.shipping))
        thr = sh["alert"]["caution"]
        w, on = [], 0
        for i in pos_ids + neg_ids:
            fr = tl[i]; lv, ons = simulate(fr, thr); t_end = fr[-1]["t"]
            if sol[i]["target"] == "1":
                w.append(any(l >= 1 and f["t"] >= t_end - 1.5 for l, f in zip(lv, fr)))
            else:
                on += ons
        res["shipping"] = {"threshold": thr, "warned": float(np.mean(w)), "fa_per_hour": on / max(neg_seconds, 1) * 3600, "chosen_on": sh["chosen_on"]}
    if a.speed_ms:
        res["speed"] = {"node_cpu_ms": a.speed_ms, "note": "YOLO11n 512x288, onnxruntime-node, 2 CPU threads"}
    if a.baseline_timeline:
        btl = json.load(open(a.baseline_timeline))
        bmax = max((f["r"] for i in ids for f in btl.get(i, [])), default=1) or 1
        bcurve = []
        for q in np.linspace(0.05, 1.0, 40):
            thr = float(q * bmax)
            w, on = [], 0
            for i in pos_ids + neg_ids:
                fr = btl.get(i, [])
                if not fr:
                    continue
                lv, ons = simulate(fr, thr); t_end = fr[-1]["t"]
                if sol[i]["target"] == "1":
                    w.append(any(l >= 1 and f["t"] >= t_end - 1.5 for l, f in zip(lv, fr)))
                else:
                    on += ons
            bcurve.append({"threshold": thr, "warned": float(np.mean(w)), "fa_per_hour": on / max(neg_seconds, 1) * 3600})
        res["baseline_tradeoff"] = bcurve
    if a.baseline:
        b = {r["id"]: float(r["score"]) for r in csv.DictReader(open(a.baseline))}
        sb = np.array([b.get(i, 0.0) for i in ids])
        res["baseline"] = {"name": "Biggest object in front (no tracking, no time-to-collision)", "map": nexar_map(ids, y, sb, groups),
                           "auc": float(roc_auc_score(y, sb))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_condition",)}, indent=1))
    for c in res["by_condition"]:
        print(f"  {c['name']:<24} n={c['n']:>4} mAP={c['map']:.3f}  CI={c['ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
