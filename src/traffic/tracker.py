"""ByteTrack-style multi-object tracker (own implementation, deterministic, no framework internals).

Two-stage association as in ByteTrack (Zhang et al., 2022): high-confidence
detections are matched first against all live tracks, then low-confidence
detections rescue the remaining tracked objects. Matching is restricted to the
same class family (people / two-wheelers / four-wheelers) and uses a constant-
velocity Kalman filter on box centre and size. Time-aware, so it works with any
frame stride.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import BICYCLE, MOTORCYCLE, PERSON
from .geometry import iou_matrix


def family(cls: int) -> int:
    if cls == PERSON:
        return 0
    if cls in (BICYCLE, MOTORCYCLE):
        return 1
    return 2


class _KF:
    """ByteTrack's Kalman filter on (cx, cy, aspect, h), with noise scaled by box height.

    Time is measured in reference steps of 1/12.5 s so the filter behaves the same
    for any frame stride.
    """
    W_POS, W_VEL, STEP = 1.0 / 20, 1.0 / 160, 1.0 / 12.5

    def __init__(self, box: np.ndarray):
        w, h = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
        self.x = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w / h, h, 0, 0, 0, 0], dtype=np.float64)
        std = [2 * self.W_POS * h, 2 * self.W_POS * h, 1e-2, 2 * self.W_POS * h,
               10 * self.W_VEL * h, 10 * self.W_VEL * h, 1e-5, 10 * self.W_VEL * h]
        self.P = np.diag(np.square(std))

    def predict(self, dt: float) -> None:
        k = max(dt / self.STEP, 1e-3)
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * k
        h = self.x[3]
        std = [self.W_POS * h, self.W_POS * h, 1e-2, self.W_POS * h, self.W_VEL * h, self.W_VEL * h, 1e-5, self.W_VEL * h]
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag(np.square(std)) * k

    def update(self, box: np.ndarray) -> None:
        w, h = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w / h, h])
        H = np.zeros((4, 8)); H[:, :4] = np.eye(4)
        hh = self.x[3]
        R = np.diag(np.square([self.W_POS * hh, self.W_POS * hh, 1e-1, self.W_POS * hh]))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        h = max(h, 1.0)
        w = max(a * h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Track:
    __slots__ = ("id", "kf", "cls_votes", "hits", "last_t", "conf", "confirmed", "last_box")

    def __init__(self, tid: int, det: np.ndarray, t: float):
        self.id = tid
        self.kf = _KF(det[:4])
        self.cls_votes: dict[int, float] = {int(det[5]): float(det[4])}
        self.hits = 1
        self.last_t = t
        self.conf = float(det[4])
        self.confirmed = False
        self.last_box = det[:4].copy()

    @property
    def cls(self) -> int:
        return max(self.cls_votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def update(self, det: np.ndarray, t: float) -> None:
        self.kf.update(det[:4])
        c = int(det[5])
        if family(c) == family(self.cls):
            self.cls_votes[c] = self.cls_votes.get(c, 0.0) + float(det[4])
        self.hits += 1
        self.last_t = t
        self.conf = float(det[4])
        self.last_box = det[:4].copy()


def _dedupe(dets: np.ndarray, thr: float = 0.75) -> np.ndarray:
    """Drop near-identical boxes of the same object family (keeps the more confident one)."""
    if len(dets) < 2:
        return dets
    order = np.argsort(-dets[:, 4], kind="stable")
    dets = dets[order]
    iou = iou_matrix(dets[:, :4], dets[:, :4])
    fam = np.array([family(int(c)) for c in dets[:, 5]])
    keep = np.ones(len(dets), dtype=bool)
    for i in range(len(dets)):
        if keep[i]:
            dup = (iou[i] > thr) & (fam == fam[i])
            dup[: i + 1] = False
            keep[dup] = False
    return dets[keep]


class ByteTracker:
    def __init__(self, params: dict):
        p = params["tracker"]
        self.high, self.low, self.new_thr = p["high_thresh"], p["low_thresh"], p["new_track_thresh"]
        self.match_iou, self.second_iou = p["match_iou"], p["second_match_iou"]
        self.lost_seconds, self.min_hits = p["lost_seconds"], p["min_hits"]
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_t: float | None = None

    def reset(self) -> None:
        self.tracks, self.next_id, self.last_t = [], 1, None

    def _match(self, tracks: list[Track], dets: np.ndarray, thr: float):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = iou_matrix(np.stack([tr.kf.box for tr in tracks]), dets[:, :4])
        fam_t = np.array([family(tr.cls) for tr in tracks])
        fam_d = np.array([family(int(c)) for c in dets[:, 5]])
        cost = 1.0 - iou
        cost[fam_t[:, None] != fam_d[None, :]] = 1e3
        cost[iou < thr] = 1e3
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c) if cost[i, j] < 1e2]
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def update(self, dets: np.ndarray, t: float) -> list[tuple[int, np.ndarray, int, float]]:
        """dets: (N, 6) x1,y1,x2,y2,conf,cls at time t. Returns [(id, box, cls, conf)] for confirmed tracks seen now."""
        dt = 0.04 if self.last_t is None else max(t - self.last_t, 1e-3)
        self.last_t = t
        for tr in self.tracks:
            tr.kf.predict(dt)
        dets = dets[dets[:, 4] >= self.low] if len(dets) else dets
        dets = _dedupe(dets)
        high = dets[dets[:, 4] >= self.high] if len(dets) else dets
        low = dets[dets[:, 4] < self.high] if len(dets) else dets

        matches, u_tr, u_det = self._match(self.tracks, high, self.match_iou)
        for i, j in matches:
            self.tracks[i].update(high[j], t)
        remaining = [self.tracks[i] for i in u_tr if t - self.tracks[i].last_t <= dt * 1.5]  # tracked (not lost) only
        m2, _, _ = self._match(remaining, low, self.second_iou)
        for i, j in m2:
            remaining[i].update(low[j], t)
        # third pass: leftover confident detections vs leftover tracks by their last observed box
        # (rescues objects whose motion model drifted during a short occlusion)
        left_tr = [self.tracks[i] for i in u_tr if self.tracks[i].last_t < t]
        if left_tr and u_det:
            sub = high[u_det]
            iou = iou_matrix(np.stack([tr.last_box for tr in left_tr]), sub[:, :4])
            fam_ok = np.array([[family(tr.cls) == family(int(c)) for c in sub[:, 5]] for tr in left_tr])
            cost = np.where((iou >= self.second_iou) & fam_ok, 1.0 - iou, 1e3)
            r, c = linear_sum_assignment(cost)
            used = set()
            for i, j in zip(r, c):
                if cost[i, j] < 1e2:
                    left_tr[i].update(sub[j], t)
                    used.add(u_det[j])
            u_det = [j for j in u_det if j not in used]
        for j in u_det:
            if high[j, 4] >= self.new_thr:
                self.tracks.append(Track(self.next_id, high[j], t))
                self.next_id += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.lost_seconds]
        out = []
        for tr in self.tracks:
            if tr.last_t == t:
                if tr.hits >= self.min_hits:
                    tr.confirmed = True
                if tr.confirmed:
                    out.append((tr.id, tr.last_box.copy(), tr.cls, tr.conf))
        return out
