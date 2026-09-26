"""
solution.py — Trumen entry point for the WIUT Hackathon 2026 CV track.

The harness imports this module, then calls detect_events() (Part A) and
streams frames through RiskEstimator (Part B). All logic lives in src/traffic;
this file only wires the official interface to it.

Importing this module loads the detector and warms it up, so model loading is
paid once, before the harness starts timing the first video.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from traffic.config import load_params  # noqa: E402
from traffic.detector import get_detector  # noqa: E402
from traffic.pipeline import analyze_video  # noqa: E402
from traffic.risk import CausalRisk  # noqa: E402

# Official class ids. fire_smoke is removed: we have no detector we trust for it,
# and predicting a class that is absent from the test set adds a zero to the macro F1.
CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
]

RISK_HORIZON_SEC = 5.0

PARAMS = load_params()
get_detector(PARAMS)          # load + warm up now, outside the per-video time budget


def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...] for one .mp4."""
    return analyze_video(video_path, PARAMS, classes=CLASSES).segments


class RiskEstimator:
    """Part B: causal P(accident starts within 5 s). Sees only the frames passed to step()."""

    def __init__(self):
        self._impl = CausalRisk(PARAMS)

    def reset(self, meta: dict) -> None:
        self._impl.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._impl.step(frame, t_sec)
