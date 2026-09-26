"""Small geometry and interval helpers (pure numpy, no state)."""
from __future__ import annotations

import cv2
import numpy as np


def point_in_poly(poly: np.ndarray, x: float, y: float, margin: float = 0.0) -> bool:
    """True if (x, y) lies inside `poly` or within `margin` px of its border."""
    if poly is None or len(poly) < 3:
        return False
    d = cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), margin > 0)
    return d >= -margin if margin > 0 else d >= 0


def in_any(polys: list[np.ndarray], x: float, y: float, margin: float = 0.0) -> bool:
    return any(point_in_poly(p, x, y, margin) for p in polys)


def signed_side(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Signed distance of p from the infinite line a->b (positive = left of a->b in image coords)."""
    d = b - a
    n = np.hypot(*d) + 1e-9
    return float((d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / n)


def proj_param(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Projection parameter of p on segment a->b (0 at a, 1 at b)."""
    d = b - a
    return float(np.dot(p - a, d) / (np.dot(d, d) + 1e-9))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Smallest signed difference a-b in radians, in (-pi, pi]."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average along axis 0 with edge padding (k forced odd)."""
    if k <= 1 or len(x) < 3:
        return x.astype(np.float64)
    k = min(k, len(x))
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return x.astype(np.float64)
    pad = k // 2
    xp = np.pad(x.astype(np.float64), [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    kern = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(xp, kern, mode="valid")
    return np.stack([np.convolve(xp[:, j], kern, mode="valid") for j in range(x.shape[1])], axis=1)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index runs [i, j] (inclusive) where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def time_runs(t: np.ndarray, mask: np.ndarray, fill_gap: float = 0.0, min_len: float = 0.0) -> list[tuple[float, float]]:
    """Time intervals where mask holds; gaps <= fill_gap merged, intervals shorter than min_len dropped."""
    ivs = [(float(t[i]), float(t[j])) for i, j in runs(np.asarray(mask, bool))]
    return clean_intervals(ivs, fill_gap, min_len)


def clean_intervals(ivs: list[tuple[float, float]], merge_gap: float = 0.0, min_len: float = 0.0) -> list[tuple[float, float]]:
    """Sort, union overlapping / near intervals, drop short ones."""
    out: list[list[float]] = []
    for s, e in sorted(ivs):
        if out and s - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= min_len]
