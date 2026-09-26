#!/usr/bin/env python3
"""Ablations on our dev labels: detector size, input resolution, frame stride, tracking on/off.

Writes a table to web/content/ablations.json (shown on the report page) and prints it.

    python tools/ablate.py --videos samples --gt data/dev_labels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

VARIANTS = [
    {"name": "Default (YOLO11m, 960 px, stride 2)", "weights": "weights/yolo11m.pt", "imgsz": 960, "stride": 2},
    {"name": "Smaller detector (YOLO11s)", "weights": "weights/yolo11s.pt", "imgsz": 960, "stride": 2},
    {"name": "Nano detector (YOLO11n)", "weights": "weights/yolo11n.pt", "imgsz": 640, "stride": 2},
    {"name": "Lower resolution (640 px)", "weights": "weights/yolo11m.pt", "imgsz": 640, "stride": 2},
    {"name": "Sparser frames (stride 5)", "weights": "weights/yolo11m.pt", "imgsz": 960, "stride": 5},
    {"name": "No track stitching", "weights": "weights/yolo11m.pt", "imgsz": 960, "stride": 2,
     "params": {"kinematics": {"stitch_gap_seconds": 0.0}}},
    {"name": "No boundary offsets", "weights": "weights/yolo11m.pt", "imgsz": 960, "stride": 2, "no_offsets": True},
]


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
    ap.add_argument("--only", nargs="*", help="run only variants whose name contains one of these words")
    args = ap.parse_args()
    os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / ".cache"))

    import traffic.detector as D
    from evaluate import evaluate
    from traffic.config import load_params, load_scene_dict
    from traffic.pipeline import ALL_CLASSES, align_scene, analyze_tracks, track_video

    gt = json.loads(Path(args.gt).read_text())
    scene = load_scene_dict()
    rows = []
    for v in VARIANTS:
        if args.only and not any(w.lower() in v["name"].lower() for w in args.only):
            continue
        os.environ["TRAFFIC_WEIGHTS"], os.environ["TRAFFIC_IMGSZ"] = v["weights"], str(v["imgsz"])
        D._SHARED = None                                  # reload the detector for this variant
        params = load_params(overrides=v.get("params"))
        if v.get("no_offsets"):
            for k, cfg in params["post"].items():
                if isinstance(cfg, dict):
                    cfg["start_offset"] = cfg["end_offset"] = 0.0
        pred, t0, secs = {}, time.perf_counter(), 0.0
        for name in gt:
            path = find_video(args.videos, name)
            vscene = align_scene(scene, str(path))
            meta, stride, store, sig, timings = track_video(str(path), params, vscene, v["stride"])
            an = analyze_tracks(meta, stride, store, sig, params, vscene, ALL_CLASSES)
            pred[name] = {"events": an.segments, "risk": []}
            secs += meta.duration
        rep = evaluate(gt, {"videos": pred})
        a = rep["part_a"]
        row = {"name": v["name"], "score_a": round(a["score_a"], 4),
               "micro_f1_05": round(a["micro"]["0.5"]["f1"], 4),
               "agnostic_f1_05": round(a["class_agnostic"]["0.5"]["f1"], 4),
               "wall_per_video_sec": round((time.perf_counter() - t0) / max(secs, 1e-3), 3)}
        rows.append(row)
        print(json.dumps(row))
    out = ROOT / "web" / "content" / "ablations.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out} (note: wall time is only meaningful when detections are not cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
