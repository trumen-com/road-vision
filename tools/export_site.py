#!/usr/bin/env python3
"""Build everything the website shows about the sample videos.

For each video: Part A analysis, the causal Part B risk curve (streamed exactly
like the harness does), an annotated H.264 playback, EDA figures and the JSON the
site's interactive pages read.

    python tools/export_site.py --videos samples --out web/public
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.config import CLASS_NAMES, VEHICLES, load_params, load_scene_dict  # noqa: E402
from traffic.pipeline import analyze_video  # noqa: E402
from traffic.render import render_video, stream_risk  # noqa: E402
from traffic.risk import CausalRisk  # noqa: E402


def frame_stats(path: str, fps: float, step_sec: float = 1.0):
    """Brightness / contrast per second and a motion heatmap from frame differencing."""
    cap = cv2.VideoCapture(path)
    step = max(1, int(round(fps * step_sec)))
    idx, prev, heat, rows, first = 0, None, None, [], None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            _, frame = cap.retrieve()
            if first is None:
                first = frame.copy()
            gray = cv2.cvtColor(cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1]))), cv2.COLOR_BGR2GRAY)
            rows.append({"t": round(idx / fps, 2), "brightness": round(float(gray.mean()), 1), "contrast": round(float(gray.std()), 1)})
            if prev is not None:
                d = cv2.absdiff(gray, prev).astype(np.float32)
                heat = d if heat is None else heat + d
            prev = gray
        idx += 1
    cap.release()
    return rows, heat, first


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def heatmap_png(heat: np.ndarray, background: np.ndarray, out: Path) -> None:
    h = cv2.GaussianBlur(heat, (0, 0), 3)
    h = (255 * np.clip(h / (np.percentile(h, 99.5) + 1e-6), 0, 1)).astype(np.uint8)
    col = cv2.applyColorMap(h, cv2.COLORMAP_INFERNO)
    bg = cv2.resize(background, (col.shape[1], col.shape[0]))
    save_png(out, cv2.addWeighted(cv2.cvtColor(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), 0.45, col, 0.75, 0))


def trajectories_png(an, background: np.ndarray, out: Path) -> None:
    """Every vehicle path coloured by heading (hue wheel), pedestrians in white."""
    W = 960
    s = W / an.meta.width
    img = cv2.resize(background, (W, int(an.meta.height * s)))
    img = (cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
    for tr in an.store.trajectories:
        pts = (tr.foot * s).astype(np.int32)
        if len(pts) < 3:
            continue
        if tr.is_person:
            cv2.polylines(img, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        d = pts[-1] - pts[0]
        hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
        color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    save_png(out, img)


def flow_png(an, background: np.ndarray, out: Path) -> None:
    ff = an.scene.flow
    W = 960
    s = W / an.meta.width
    img = cv2.resize(background, (W, int(an.meta.height * s)))
    img = (cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
    coh = ff.coherence()
    cw, ch = W / ff.gx, img.shape[0] / ff.gy
    for gy in range(ff.gy):
        for gx in range(ff.gx):
            if ff.count[gy, gx] < ff.min_support or coh[gy, gx] < ff.min_coherence:
                continue
            v = ff.sum[gy, gx] / (np.linalg.norm(ff.sum[gy, gx]) + 1e-9)
            c = np.array([(gx + 0.5) * cw, (gy + 0.5) * ch])
            g = ff.group((gx + 0.5) * an.meta.width / ff.gx, (gy + 0.5) * an.meta.height / ff.gy)
            color = (0, 200, 255) if g == "flow0" else (255, 120, 0)
            cv2.arrowedLine(img, tuple(c.astype(int)), tuple((c + v * cw * 0.8).astype(int)), color, 1, cv2.LINE_AA, tipLength=0.4)
    save_png(out, img)


def per_second_counts(an) -> list[dict]:
    n = int(np.ceil(an.meta.duration))
    names = {c: CLASS_NAMES.get(c, str(c)) for c in {tr.cls for tr in an.store.trajectories}}
    counts = [{"t": i, **{nm: 0 for nm in names.values()}} for i in range(n)]
    moving = [0] * n
    speeds: list[list[float]] = [[] for _ in range(n)]
    for tr in an.store.trajectories:
        secs = np.floor(tr.t).astype(int)
        for sec in np.unique(secs):
            if 0 <= sec < n:
                counts[sec][names[tr.cls]] += 1
                if tr.cls in VEHICLES:
                    sp = tr.speed[secs == sec]
                    speeds[sec].append(float(np.median(sp)))
                    moving[sec] += int(np.median(sp) > 0.5)
    for i in range(n):
        counts[i]["moving_vehicles"] = moving[i]
        counts[i]["median_speed"] = round(float(np.median(speeds[i])), 3) if speeds[i] else None
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default=str(ROOT / "web" / "public"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--stride", type=int, default=None, help="detector frame stride (default: chosen from the time budget)")
    args = ap.parse_args()

    params, scene = load_params(), load_scene_dict()
    out = Path(args.out)
    (out / "data" / "videos").mkdir(parents=True, exist_ok=True)
    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    index = []
    for path in videos:
        stem = path.stem
        print(f"[{path.name}] analysing")
        an = analyze_video(str(path), params, scene, stride=args.stride)
        print(f"[{path.name}] risk curve")
        risk = stream_risk(str(path), CausalRisk(params), an.meta)
        stats, heat, first = frame_stats(str(path), an.meta.fps)
        figs = out / "media" / "eda"
        save_png(figs / f"{stem}_frame.png", cv2.resize(first, (960, int(960 * first.shape[0] / first.shape[1]))))
        if heat is not None:
            heatmap_png(heat, first, figs / f"{stem}_motion.png")
        trajectories_png(an, first, figs / f"{stem}_trajectories.png")
        flow_png(an, first, figs / f"{stem}_flow.png")
        if not args.no_render:
            print(f"[{path.name}] rendering")
            render_video(str(path), an, str(out / "media" / f"{stem}_annotated.mp4"), risk, max_width=960)
        speeds = [float(s) for tr in an.store.trajectories if tr.is_vehicle for s in tr.speed]
        doc = {**an.to_dict(),
               "risk": risk[:: max(1, int(round(an.meta.fps / 5)))],
               "frame_stats": stats,
               "counts": per_second_counts(an),
               "speed_hist": np.histogram(np.clip(speeds, 0, 6), bins=24, range=(0, 6))[0].tolist() if speeds else [],
               "n_tracks": {CLASS_NAMES.get(c, str(c)): sum(tr.cls == c for tr in an.store.trajectories)
                            for c in sorted({tr.cls for tr in an.store.trajectories})},
               "media": {"annotated": f"media/{stem}_annotated.mp4", "frame": f"media/eda/{stem}_frame.png",
                         "motion": f"media/eda/{stem}_motion.png", "trajectories": f"media/eda/{stem}_trajectories.png",
                         "flow": f"media/eda/{stem}_flow.png"}}
        (out / "data" / "videos" / f"{stem}.json").write_text(json.dumps(doc))
        index.append({"id": stem, "file": path.name, "duration": round(an.meta.duration, 2), "fps": an.meta.fps,
                      "width": an.meta.width, "height": an.meta.height, "events": len(an.segments),
                      "classes": sorted({s[2] for s in an.segments})})
        print(f"[{path.name}] {len(an.segments)} events, timings {an.timings}")
    (out / "data" / "index.json").write_text(json.dumps({"videos": index}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
