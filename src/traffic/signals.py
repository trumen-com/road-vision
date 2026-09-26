"""Traffic-light state reader.

Each signal head is a fixed ROI (drawn in the scene map, or discovered from YOLO
'traffic light' detections when the map has none). The lit lamp is the brightest,
most saturated blob in the ROI; its hue decides red / amber / green. States are
smoothed with a majority filter so single-frame glare does not flip them.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import TRAFFIC_LIGHT
from .geometry import iou_matrix

RED, AMBER, GREEN, UNKNOWN = "red", "amber", "green", "unknown"


def lamp_features(frame: np.ndarray, roi: np.ndarray) -> tuple[float, float]:
    """(redness, brightness) of the lit lamp: mean R-G and V over the brightest, most saturated pixels.

    A red lamp gives a large positive R-G, a green one a negative value. Daylight
    washes lamps out, so absolute thresholds fail; the timeline therefore separates
    states by clustering this value over the whole video (see SignalTimeline.finalize).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0, 0.0
    crop = frame[y1:y2, x1:x2].astype(np.float32)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).astype(np.float32)
    score = hsv[..., 1] * hsv[..., 2]
    sel = score >= np.percentile(score, 92)
    red = float((crop[..., 2][sel] - crop[..., 1][sel]).mean())
    return red, float(hsv[..., 2][sel].mean())


def classify_roi(frame: np.ndarray, roi: np.ndarray, min_frac: float = 0.015) -> str:
    """Absolute colour decision for clearly lit lamps (night, overcast); UNKNOWN when unsure."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return UNKNOWN
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (val >= 170) & (sat >= 90)
    n = lit.size
    red = int((lit & ((hue <= 10) | (hue >= 165))).sum())
    amber = int((lit & (hue > 10) & (hue <= 32)).sum())
    green = int((lit & (hue >= 40) & (hue <= 100)).sum())
    best = max((red, RED), (amber, AMBER), (green, GREEN))
    return best[1] if best[0] >= max(3, min_frac * n) else UNKNOWN


class SignalTimeline:
    """Per-signal state over sampled times, with majority smoothing and lookups."""

    def __init__(self, rois: dict[str, np.ndarray]):
        self.rois = rois
        self.t: list[float] = []
        self.raw: dict[str, list[str]] = {k: [] for k in rois}
        self.feat: dict[str, list[tuple[float, float]]] = {k: [] for k in rois}
        self.states: dict[str, np.ndarray] = {}

    def observe(self, frame: np.ndarray, t: float) -> None:
        self.t.append(t)
        for k, roi in self.rois.items():
            self.raw[k].append(classify_roi(frame, roi))
            self.feat[k].append(lamp_features(frame, roi))

    @staticmethod
    def _cluster(red: np.ndarray) -> np.ndarray | None:
        """Two-state split of the redness series (1-D 2-means); None when the lamp never clearly changes."""
        if len(red) < 20:
            return None
        lo, hi = np.percentile(red, 10), np.percentile(red, 90)
        c0, c1 = lo, hi
        for _ in range(20):
            lab = np.abs(red - c1) < np.abs(red - c0)
            if lab.all() or not lab.any():
                return None
            c0, c1 = red[~lab].mean(), red[lab].mean()
        spread = np.sqrt((red[~lab].var() * (~lab).sum() + red[lab].var() * lab.sum()) / len(red))
        if c1 - c0 < max(12.0, 3.0 * spread) or min(lab.mean(), 1 - lab.mean()) < 0.05:
            return None
        return np.where(lab, RED, GREEN)

    def finalize(self, window: int = 7) -> None:
        self.t_arr = np.asarray(self.t)
        for k, seq in self.raw.items():
            seq = np.asarray(seq)
            feats = np.asarray(self.feat.get(k) or [], dtype=np.float32).reshape(-1, 2)
            clustered = self._cluster(feats[:, 0]) if len(feats) == len(seq) else None
            if clustered is not None:
                seq = clustered
            out = seq.copy()
            half = window // 2
            for i in range(len(seq)):
                win = seq[max(0, i - half): i + half + 1]
                win = win[win != UNKNOWN]
                if len(win):
                    vals, cnt = np.unique(win, return_counts=True)
                    out[i] = vals[np.argmax(cnt)]
            # carry the last known state through short unknown gaps (lamp flicker / occlusion)
            last = UNKNOWN
            for i in range(len(out)):
                if out[i] == UNKNOWN:
                    out[i] = last
                else:
                    last = out[i]
            self.states[k] = out

    def state(self, signal_id: str | None, t: float) -> str:
        if not self.states:
            return UNKNOWN
        ids = [signal_id] if signal_id in self.states else list(self.states)
        i = int(np.clip(np.searchsorted(self.t_arr, t), 0, len(self.t_arr) - 1))
        vals = [self.states[k][i] for k in ids]
        known = [v for v in vals if v != UNKNOWN]
        if not known:
            return UNKNOWN
        vals, cnt = np.unique(known, return_counts=True)
        return str(vals[np.argmax(cnt)])

    def next_change(self, signal_id: str | None, t: float, to_state: str = GREEN) -> float | None:
        if not self.states:
            return None
        k = signal_id if signal_id in self.states else next(iter(self.states))
        i = int(np.searchsorted(self.t_arr, t))
        idx = np.nonzero(self.states[k][i:] == to_state)[0]
        return float(self.t_arr[i + idx[0]]) if len(idx) else None

    def next_exit(self, signal_id: str | None, t: float, states: tuple) -> float | None:
        """First time after t at which the signal is no longer in any of `states` (e.g. red ends)."""
        if not self.states:
            return None
        k = signal_id if signal_id in self.states else next(iter(self.states))
        i = int(np.searchsorted(self.t_arr, t))
        idx = np.nonzero(~np.isin(self.states[k][i:], list(states) + [UNKNOWN]))[0]
        return float(self.t_arr[i + idx[0]]) if len(idx) else None

    def to_dict(self) -> dict:
        return {"t": [round(x, 3) for x in self.t], "states": {k: v.tolist() for k, v in self.states.items()},
                "redness": {k: [round(f[0], 1) for f in v] for k, v in self.feat.items()},
                "rois": {k: v.tolist() for k, v in self.rois.items()}}


def discover_lights(detections: list[np.ndarray], min_presence: float = 0.4) -> dict[str, np.ndarray]:
    """Cluster 'traffic light' boxes seen across frames; keep those present in >= min_presence of frames."""
    clusters: list[list[np.ndarray]] = []
    for dets in detections:
        lights = dets[dets[:, 5] == TRAFFIC_LIGHT][:, :4] if len(dets) else np.zeros((0, 4))
        for box in lights:
            if clusters:
                reps = np.stack([np.median(c, axis=0) for c in clusters])
                iou = iou_matrix(box[None], reps)[0]
                j = int(np.argmax(iou))
                if iou[j] > 0.3:
                    clusters[j].append(box)
                    continue
            clusters.append([box])
    n = max(1, len(detections))
    out = {}
    for c in clusters:
        if len(c) / n >= min_presence:
            out[f"auto{len(out)}"] = np.median(np.stack(c), axis=0).astype(np.float32)
    return out
