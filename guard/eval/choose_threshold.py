#!/usr/bin/env python3
"""Choose the shipping alert threshold on the TRAINING split (out-of-fold timelines), never on the test split.

A positive training video counts as warned if an alert is active at some moment in the 2.0 s before the annotated event. False warnings are alert onsets per hour of
normal-driving footage. We pick the threshold with the most warned events subject to a false-warning budget.

  python eval/choose_threshold.py --timeline results/train_timeline_oof.json --data ../guard_data/nexar --out results/shipping.json
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def simulate(frames, thr):
    """Returns (alert onset times, per-frame alert level)."""
    lvl, above, below, onsets, levels = 0, 0, None, [], []
    for f in frames:
        r = f["r"]; above = above + 1 if r >= thr else 0
        if lvl == 0 and above >= 2:
            lvl = 1; onsets.append(f["t"])
        if lvl == 1 and r < 0.7 * thr:
            below = f["t"] if below is None else below
            if f["t"] - below > 0.5:
                lvl = 0
        else:
            below = None
        levels.append(lvl)
    return onsets, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fa-budget", type=float, default=60.0, help="max false warnings per hour of (dense) normal driving")
    a = ap.parse_args()
    tl = json.load(open(a.timeline))
    ev = {}
    for lab in ("positive", "negative"):
        for r in csv.DictReader(open(Path(a.data) / "train" / lab / "metadata.csv")):
            ev[r["file_name"][:-4]] = float(r["time_of_event"]) if lab == "positive" and r["time_of_event"] else None
    rows = []
    for thr in np.round(np.linspace(0.3, 0.95, 27), 3):
        warned, n_pos, onsets, neg_s = 0, 0, 0, 0.0
        for clip, fr in tl.items():
            if clip not in ev or not fr:
                continue
            ons, lv = simulate(fr, thr)
            te = ev[clip]
            if te is None:
                onsets += len(ons); neg_s += fr[-1]["t"] - fr[0]["t"]
            else:
                n_pos += 1
                # warned = an alert is active at some moment in the 2 s before the event (same rule as the test report)
                warned += any(l >= 1 and te - 2.0 <= f["t"] <= te for l, f in zip(lv, fr))
        rows.append({"threshold": float(thr), "warned": warned / max(n_pos, 1), "fa_per_hour": onsets / max(neg_s, 1) * 3600})
    ok = [r for r in rows if r["fa_per_hour"] <= a.fa_budget]
    best = max(ok, key=lambda r: r["warned"]) if ok else min(rows, key=lambda r: r["fa_per_hour"])
    out = {"alert": {"caution": best["threshold"], "brake": round(min(0.97, best["threshold"] + 0.15), 3), "release": round(0.7 * best["threshold"], 3)},
           "train_operating_point": best, "fa_budget": a.fa_budget, "curve": rows,
           "chosen_on": f"Nexar training split, out-of-fold, budget {a.fa_budget:.0f} false warnings per hour"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
