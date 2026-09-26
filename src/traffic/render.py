"""Annotated playback: boxes, IDs, trails, scene overlay, live event banner, timeline and risk strip.

Writes browser-ready H.264 (yuv420p, faststart) through an ffmpeg pipe; used by
tools/export_site.py for the website and by the live demo.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import CLASS_NAMES, PERSON, TWO_WHEELERS

EVENT_COLORS = {  # BGR
    "accident": (40, 40, 230), "near_miss": (0, 140, 255), "red_light": (60, 60, 200), "wrong_way": (200, 0, 200),
    "illegal_u_turn": (180, 60, 150), "stopped_vehicle": (0, 200, 255), "jaywalking": (255, 180, 0),
    "failure_to_yield": (255, 120, 60), "illegal_turn": (150, 80, 220), "solid_line_crossing": (0, 230, 230),
    "stop_line": (80, 80, 255), "congestion": (60, 180, 60), "road_obstacle": (120, 120, 120), "fire_smoke": (0, 60, 255),
}


def _obj_color(cls: int) -> tuple[int, int, int]:
    if cls == PERSON:
        return (255, 200, 0)
    if cls in TWO_WHEELERS:
        return (0, 220, 120)
    return (255, 255, 255)


def _ffmpeg_writer(path: Path, w: int, h: int, fps: float) -> subprocess.Popen:
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as err:
            raise RuntimeError("ffmpeg not found (install ffmpeg or imageio-ffmpeg)") from err
    cmd = [exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_video(video_path: str, analysis, out_path: str, risk: list | None = None, max_width: int = 1280,
                 progress: Callable[[float], None] | None = None) -> str:
    meta = analysis.meta
    scale = min(1.0, max_width / meta.width)
    W, H = int(meta.width * scale) // 2 * 2, int(meta.height * scale) // 2 * 2
    strip_h = 70
    writer = _ffmpeg_writer(Path(out_path), W, H + strip_h, meta.fps)
    dur = max(meta.duration, 1e-3)

    # per-sample lookup of tracked boxes
    times = analysis.store.sample_times
    by_sample: list[list] = [[] for _ in range(len(times))]
    for tr in analysis.store.trajectories:
        idx = np.searchsorted(times, tr.t)
        for k, si in enumerate(idx):
            if si < len(times):
                by_sample[si].append((tr, k))
    risk_arr = np.asarray(risk, dtype=np.float32) if risk else None

    # static timeline strip
    strip = np.full((strip_h, W, 3), 24, np.uint8)
    labels = sorted({s[2] for s in analysis.segments})
    row_h = max(6, (strip_h - 24) // max(1, len(labels)))
    for r, lab in enumerate(labels):
        for s, e, l in analysis.segments:
            if l == lab:
                x0, x1 = int(s / dur * W), max(int(e / dur * W), int(s / dur * W) + 2)
                cv2.rectangle(strip, (x0, 4 + r * row_h), (x1, 4 + (r + 1) * row_h - 2), EVENT_COLORS.get(lab, (200, 200, 200)), -1)
    if risk_arr is not None and len(risk_arr):
        pts = [(int(t / dur * W), strip_h - 4 - int(v * 18)) for t, v in risk_arr[:: max(1, len(risk_arr) // W)]]
        cv2.polylines(strip, [np.array(pts, np.int32)], False, (80, 80, 255), 1)
        cv2.line(strip, (0, strip_h - 4 - 9), (W, strip_h - 4 - 9), (70, 70, 70), 1)

    base_overlay = None
    cap = cv2.VideoCapture(str(video_path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / meta.fps
        if scale != 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        if base_overlay is None:
            blank = np.zeros((meta.height, meta.width, 3), np.uint8)
            base_overlay = cv2.resize(analysis.scene.draw(blank), (W, H), interpolation=cv2.INTER_NEAREST)
        frame = cv2.add(frame, base_overlay)
        si = int(np.clip(np.searchsorted(times, t + 1e-6) - 1, 0, max(len(times) - 1, 0))) if len(times) else None
        if si is not None:
            for tr, k in by_sample[si]:
                x1, y1, x2, y2 = (tr.box[k] * scale).astype(int)
                col = _obj_color(tr.cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{CLASS_NAMES.get(tr.cls, tr.cls)} #{tr.id}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                trail = tr.foot[max(0, k - 25): k + 1] * scale
                if len(trail) > 1:
                    cv2.polylines(frame, [trail.astype(np.int32)], False, (0, 255, 255), 1)
        # highlight objects involved in active raw events and show a banner
        active = [e for e in analysis.raw if e.start <= t <= e.end]
        involved = {tid for e in active for tid in e.tracks}
        if si is not None and involved:
            for tr, k in by_sample[si]:
                if tr.id in involved or set(tr.members) & involved:
                    x1, y1, x2, y2 = (tr.box[k] * scale).astype(int)
                    lab = next(e.label for e in active if tr.id in e.tracks or set(tr.members) & set(e.tracks))
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), EVENT_COLORS.get(lab, (0, 0, 255)), 3)
        seg_now = sorted({l for s, e, l in analysis.segments if s <= t <= e})
        y = 26
        for lab in seg_now:
            (tw, th), _ = cv2.getTextSize(lab.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (8, y - th - 8), (8 + tw + 12, y + 6), EVENT_COLORS.get(lab, (0, 0, 255)), -1)
            cv2.putText(frame, lab.upper(), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += th + 18
        if risk_arr is not None and len(risk_arr):
            r = float(risk_arr[min(i, len(risk_arr) - 1), 1])
            bw = 160
            cv2.rectangle(frame, (W - bw - 12, 12), (W - 12, 34), (40, 40, 40), -1)
            cv2.rectangle(frame, (W - bw - 12, 12), (W - bw - 12 + int(bw * r), 34), (0, 0, 255) if r >= 0.5 else (0, 180, 255), -1)
            cv2.putText(frame, f"risk {r:.2f}", (W - bw - 6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{t:6.1f}s", (W - 90, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        bar = strip.copy()
        cv2.line(bar, (int(t / dur * W), 0), (int(t / dur * W), strip_h), (255, 255, 255), 1)
        writer.stdin.write(np.vstack([frame, bar]).tobytes())
        i += 1
        if progress and i % 100 == 0 and meta.n_frames:
            progress(min(1.0, i / meta.n_frames))
    cap.release()
    writer.stdin.close()
    writer.wait()
    return out_path


def stream_risk(video_path: str, estimator, meta) -> list[list[float]]:
    """Feed frames to a RiskEstimator exactly as the harness does; returns [[t, score], ...]."""
    cap = cv2.VideoCapture(str(video_path))
    estimator.reset({"video_id": meta.name, "fps": meta.fps, "width": meta.width, "height": meta.height, "n_frames": meta.n_frames})
    curve, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / meta.fps
        curve.append([round(t, 3), round(float(np.clip(estimator.step(frame, t), 0, 1)), 4)])
        i += 1
    cap.release()
    return curve
