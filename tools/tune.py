#!/usr/bin/env python3
"""Tune per-class segment post-processing against our own dev labels.

Detection + tracking runs once per video (cached in TRAFFIC_CACHE); the rules
run once; then only the cheap post-processing is grid-searched per class to
maximise that class's mean F1 over tIoU {0.3, 0.5, 0.7} with the official
matcher from evaluate.py. Ties prefer the untouched default (no offsets), which
limits over-fitting to four sample videos.

    python tools/tune.py --videos samples --gt my_labels.json            # report only
    python tools/tune.py --videos samples --gt my_labels.json --write    # update configs/params.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluate import TIOU_THRESHOLDS, match_segments, prf  # noqa: E402
from traffic.config import load_params, load_scene_dict  # noqa: E402
from traffic.geometry import clean_intervals  # noqa: E402
from traffic.pipeline import ALL_CLASSES, align_scene, analyze_tracks, track_video  # noqa: E402

OFFSETS = [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]
GAPS = [0.5, 1.0, 2.0, 3.0, 5.0]
MIN_LENS = [0.3, 0.6, 1.0, 2.0]


def class_f1(pred: dict, gt: dict, label: str) -> float:
    tot = {t: [0, 0, 0] for t in TIOU_THRESHOLDS}
    for vid, g in gt.items():
        gs = [(s, e) for s, e, l in g["events"] if l == label]
        ps = pred.get(vid, [])
        for t in TIOU_THRESHOLDS:
            for k, v in enumerate(match_segments(gs, ps, t)):
                tot[t][k] += v
    return sum(prf(*tot[t])["f1"] for t in TIOU_THRESHOLDS) / len(TIOU_THRESHOLDS)


def find_video(folder: str, name: str) -> Path:
    """Label keys are the original file names; accept any file with the same stem (e.g. a 720p proxy .mp4)."""
    stem = Path(name).stem.lower()
    for p in Path(folder).iterdir():
        if p.stem.lower() == stem and p.suffix.lower() == ".mp4":
            return p
    raise FileNotFoundError(f"{name} not found in {folder}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--write", action="store_true", help="write tuned post-processing into configs/params.json")
    ap.add_argument("--stride", type=int, default=None, help="detector stride (1 for the 10 fps proxies)")
    args = ap.parse_args()
    os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / ".cache"))

    params = load_params()
    scene = load_scene_dict()
    gt = json.loads(Path(args.gt).read_text())
    raw: dict[str, list] = {}
    durations = {}
    for name in gt:
        path = find_video(args.videos, name)
        vscene = align_scene(scene, str(path))
        meta, stride, store, sig, _ = track_video(str(path), params, vscene, args.stride)
        an = analyze_tracks(meta, stride, store, sig, params, vscene)
        raw[name] = an.raw
        durations[name] = meta.duration
        print(f"{name}: {len(an.raw)} raw hits")

    labels = sorted({l for g in gt.values() for _, _, l in g["events"]} | {e.label for r in raw.values() for e in r})
    best_cfg = {}
    print(f"\n{'class':<22}{'default':>9}{'tuned':>9}   settings")
    for label in labels:
        if label not in ALL_CLASSES:
            continue
        base = dict(params["post"]["default"]); base.update(params["post"].get(label, {}))

        def score(cfg):
            pred = {}
            for vid, evs in raw.items():
                ivs = [(e.start, e.end) for e in evs if e.label == label]
                segs = [(max(0.0, s + cfg["start_offset"]), min(durations[vid], e + cfg["end_offset"]))
                        for s, e in clean_intervals(ivs, cfg["merge_gap"], cfg["min_len"])]
                pred[vid] = clean_intervals(segs, 0.0, 0.05)
            return class_f1(pred, gt, label)

        default = score(base)
        best, best_s = base, default
        for so, eo, g, m in itertools.product(OFFSETS, OFFSETS, GAPS, MIN_LENS):
            cfg = {"start_offset": so, "end_offset": eo, "merge_gap": g, "min_len": max(m, base["min_len"] if label in ("congestion", "stopped_vehicle") else m)}
            s = score(cfg)
            if s > best_s + 1e-9:
                best, best_s = cfg, s
        best_cfg[label] = best
        print(f"{label:<22}{default:>9.3f}{best_s:>9.3f}   {best}")
    if args.write:
        path = ROOT / "configs" / "params.json"
        p = json.loads(path.read_text())
        for label, cfg in best_cfg.items():
            p["post"][label] = cfg
        path.write_text(json.dumps(p, indent=2) + "\n")
        print(f"\nwrote tuned post-processing to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
