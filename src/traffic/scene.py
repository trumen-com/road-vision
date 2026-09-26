"""Scene map (hand-drawn, rescaled to the video) + a flow field learned from trajectories.

The camera never moves, so the layout is drawn once in tools/scene_editor.html.
Where the map is silent, the FlowField learned from vehicle motion supplies the
drivable area and the legal direction of travel at each image location.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import in_any, point_in_poly


def _poly(pts, sx: float, sy: float, M: np.ndarray | None = None) -> np.ndarray:
    """Reference-frame points -> video pixels: through the alignment matrix M if given, else plain scaling."""
    a = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if M is not None:
        return cv2.perspectiveTransform(a[None], M.astype(np.float64))[0].astype(np.float32)
    return a * np.array([sx, sy], dtype=np.float32)


def _vec(v, sx: float, sy: float, M: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    out = (M[:2, :2] @ v).astype(np.float32) if M is not None else v * np.array([sx, sy], dtype=np.float32)
    return out / (np.linalg.norm(out) + 1e-9)


def estimate_alignment(frame: np.ndarray, ref: np.ndarray, ref_size: tuple[int, int]) -> np.ndarray | None:
    """3x3 matrix mapping scene-map coordinates (reference_size) onto this video's pixels.

    SIFT on contrast-equalised 960-px images and a RANSAC similarity transform: robust to
    the day/night change and to moving traffic. Returns None when the fit is not trustworthy,
    in which case the map is only rescaled.
    """
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        k = 960.0 / g.shape[1]
        return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, (960, int(round(g.shape[0] * k))))), k

    try:
        a, ka = prep(ref)
        b, kb = prep(frame)
        sift = cv2.SIFT_create(4000)
        k1, d1 = sift.detectAndCompute(a, None)
        k2, d2 = sift.detectAndCompute(b, None)
        if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
            return None
        pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            return None
        src = np.float32([k1[g.queryIdx].pt for g in good])
        dst = np.float32([k2[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                             maxIters=5000, confidence=0.999, refineIters=20)
        if A is None or inl is None or int(inl.sum()) < 25:
            return None
        scale = float(np.hypot(A[0, 0], A[1, 0]))
        if not 0.85 < scale < 1.18 or abs(np.degrees(np.arctan2(A[1, 0], A[0, 0]))) > 8:
            return None
    except cv2.error:
        return None
    A3 = np.vstack([A, [0, 0, 1]])
    ref_to_small = np.diag([ka * ref.shape[1] / ref_size[0], ka * ref.shape[0] / ref_size[1], 1.0])
    small_to_frame = np.diag([1.0 / kb, 1.0 / kb, 1.0])
    return small_to_frame @ A3 @ ref_to_small


@dataclass
class StopLine:
    id: str
    a: np.ndarray
    b: np.ndarray
    direction: np.ndarray          # unit vector of legal travel through the line
    signal: str | None = None      # signal id controlling it (None -> any/auto)
    red_states: tuple = ("red",)   # states of that signal that mean "red" for this line


@dataclass
class Lane:
    id: str
    polygon: np.ndarray
    direction: np.ndarray
    group: str


@dataclass
class Scene:
    width: int
    height: int
    carriageway: list[np.ndarray] = field(default_factory=list)
    exclude: list[np.ndarray] = field(default_factory=list)
    crosswalks: list[np.ndarray] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    intersection: np.ndarray | None = None
    stop_lines: list[StopLine] = field(default_factory=list)
    signals: dict[str, np.ndarray] = field(default_factory=dict)   # id -> roi xyxy
    solid_lines: list[np.ndarray] = field(default_factory=list)
    gates: dict[str, np.ndarray] = field(default_factory=dict)
    allowed_movements: set[tuple[str, str]] = field(default_factory=set)
    u_turn_allowed: list[np.ndarray] = field(default_factory=list)
    flow: "FlowField | None" = None

    @classmethod
    def from_dict(cls, d: dict, width: int, height: int) -> "Scene":
        rw, rh = d.get("reference_size") or [width, height]
        sx, sy = width / float(rw), height / float(rh)
        M = np.asarray(d["_align"], dtype=np.float64) if d.get("_align") is not None else None
        P = lambda pts: _poly(pts, sx, sy, M)  # noqa: E731
        s = cls(width=width, height=height)
        s.carriageway = [P(p) for p in d.get("carriageway", []) if len(p) >= 3]
        s.exclude = [P(p) for p in d.get("exclude", []) if len(p) >= 3]
        s.crosswalks = [P(p) for p in d.get("crosswalks", []) if len(p) >= 3]
        for ln in d.get("lanes", []):
            s.lanes.append(Lane(ln.get("id", f"lane{len(s.lanes)}"), P(ln["polygon"]),
                                _vec(ln["direction"], sx, sy, M), ln.get("group", "default")))
        inter = d.get("intersection") or []
        s.intersection = P(inter) if len(inter) >= 3 else None
        for sl in d.get("stop_lines", []):
            a, b = P(sl["line"])
            s.stop_lines.append(StopLine(sl.get("id", f"sl{len(s.stop_lines)}"), a, b, _vec(sl["direction"], sx, sy, M),
                                         sl.get("signal"), tuple(sl.get("red_states", ["red"]))))
        for sg in d.get("signals", []):
            x1, y1, x2, y2 = sg["roi"]
            c = P([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            s.signals[sg["id"]] = np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], dtype=np.float32)
        s.solid_lines = [P(p) for p in d.get("solid_lines", []) if len(p) >= 2]
        s.gates = {g["id"]: P(g["polygon"]) for g in d.get("gates", []) if len(g["polygon"]) >= 3}
        s.allowed_movements = {tuple(m) for m in d.get("allowed_movements", [])}
        s.u_turn_allowed = [P(p) for p in d.get("u_turn_allowed_zones", []) if len(p) >= 3]
        return s

    # ---- queries -------------------------------------------------------------
    def on_road(self, x: float, y: float) -> bool:
        """Carriageway test: hand-drawn polygon if present, else the learned drivable mask."""
        if in_any(self.exclude, x, y):
            return False
        if self.carriageway:
            return in_any(self.carriageway, x, y)
        return self.flow is not None and self.flow.drivable(x, y)

    def in_crosswalk(self, x: float, y: float, margin: float = 0.0) -> bool:
        return in_any(self.crosswalks, x, y, margin)

    def in_intersection(self, x: float, y: float) -> bool:
        return self.intersection is not None and point_in_poly(self.intersection, x, y)

    def lane_at(self, x: float, y: float) -> Lane | None:
        for ln in self.lanes:
            if point_in_poly(ln.polygon, x, y):
                return ln
        return None

    def legal_direction(self, x: float, y: float) -> np.ndarray | None:
        """Unit vector of legal travel at (x, y): lane map first, learned flow second."""
        ln = self.lane_at(x, y)
        if ln is not None:
            return ln.direction
        if self.lanes:          # lanes drawn but point outside all of them -> no opinion
            return None
        return self.flow.direction(x, y) if self.flow is not None else None

    def direction_group(self, x: float, y: float) -> str | None:
        ln = self.lane_at(x, y)
        if ln is not None:
            return ln.group
        if self.flow is not None:
            return self.flow.group(x, y)
        return None

    def gate_at(self, x: float, y: float) -> str | None:
        for gid, poly in self.gates.items():
            if point_in_poly(poly, x, y):
                return gid
        return None

    def draw(self, img: np.ndarray) -> np.ndarray:
        """Overlay the map on a frame (used by the renderer and the website)."""
        ov = img.copy()
        for p in self.carriageway:
            cv2.polylines(ov, [p.astype(np.int32)], True, (90, 90, 90), 2)
        for p in self.crosswalks:
            cv2.fillPoly(ov, [p.astype(np.int32)], (255, 255, 255))
        for ln in self.lanes:
            c = ln.polygon.mean(axis=0)
            cv2.arrowedLine(ov, tuple(c.astype(int)), tuple((c + ln.direction * 40).astype(int)), (0, 200, 255), 2, tipLength=0.4)
        for sl in self.stop_lines:
            cv2.line(ov, tuple(sl.a.astype(int)), tuple(sl.b.astype(int)), (0, 0, 255), 3)
        for p in self.solid_lines:
            cv2.polylines(ov, [p.astype(np.int32)], False, (0, 255, 255), 2)
        if self.intersection is not None:
            cv2.polylines(ov, [self.intersection.astype(np.int32)], True, (255, 120, 0), 2)
        for roi in self.signals.values():
            x1, y1, x2, y2 = roi.astype(int)
            cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 0, 255), 2)
        return cv2.addWeighted(ov, 0.35, img, 0.65, 0)


class FlowField:
    """Dominant direction of vehicle motion per grid cell, learned from moving tracks.

    direction(x, y) is the resultant unit vector of all moving samples in the
    cell, trusted only where enough samples agree (coherence = |mean of unit vectors|).
    Cells are split into two direction groups (e.g. the two carriageways) by
    2-means on the directions.
    """

    def __init__(self, width: int, height: int, grid=(48, 27), min_support=15, min_coherence=0.65):
        self.w, self.h = width, height
        self.gx, self.gy = grid
        self.min_support, self.min_coherence = min_support, min_coherence
        self.sum = np.zeros((self.gy, self.gx, 2), dtype=np.float64)
        self.count = np.zeros((self.gy, self.gx), dtype=np.float64)
        self.occupancy = np.zeros((self.gy, self.gx), dtype=np.float64)
        self._groups: np.ndarray | None = None

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int(np.clip(x / self.w * self.gx, 0, self.gx - 1))
        cy = int(np.clip(y / self.h * self.gy, 0, self.gy - 1))
        return cy, cx

    def add(self, x: float, y: float, vx: float, vy: float, moving: bool) -> None:
        cy, cx = self._cell(x, y)
        self.occupancy[cy, cx] += 1
        if moving:
            n = np.hypot(vx, vy) + 1e-9
            self.sum[cy, cx] += (vx / n, vy / n)
            self.count[cy, cx] += 1
            self._groups = None

    def merge(self, other: "FlowField", weight: float = 1.0) -> None:
        self.sum += other.sum * weight
        self.count += other.count * weight
        self.occupancy += other.occupancy * weight
        self._groups = None

    def coherence(self) -> np.ndarray:
        return np.linalg.norm(self.sum, axis=2) / np.maximum(self.count, 1)

    def direction(self, x: float, y: float) -> np.ndarray | None:
        cy, cx = self._cell(x, y)
        if self.count[cy, cx] < self.min_support:
            return None
        v = self.sum[cy, cx]
        n = np.linalg.norm(v)
        if n / self.count[cy, cx] < self.min_coherence:
            return None
        return (v / n).astype(np.float32)

    def drivable(self, x: float, y: float) -> bool:
        cy, cx = self._cell(x, y)
        y0, y1, x0, x1 = max(cy - 1, 0), cy + 2, max(cx - 1, 0), cx + 2
        return self.count[y0:y1, x0:x1].max() >= max(3, self.min_support // 3)

    def group(self, x: float, y: float) -> str | None:
        if self._groups is None:
            self._fit_groups()
        cy, cx = self._cell(x, y)
        g = self._groups[cy, cx]
        return None if g < 0 else f"flow{g}"

    def _fit_groups(self) -> None:
        coh = self.coherence()
        ok = (self.count >= self.min_support) & (coh >= self.min_coherence)
        groups = -np.ones(self.count.shape, dtype=np.int32)
        if ok.sum() >= 2:
            v = self.sum[ok] / np.linalg.norm(self.sum[ok], axis=1, keepdims=True)
            # deterministic 2-means on unit vectors, seeded with the first cell and its most opposite cell
            c0 = v[0]
            c1 = v[np.argmin(v @ c0)]
            for _ in range(10):
                lab = (v @ c1 > v @ c0).astype(np.int32)
                if lab.min() == lab.max():
                    break
                c0 = v[lab == 0].mean(0); c0 /= np.linalg.norm(c0) + 1e-9
                c1 = v[lab == 1].mean(0); c1 /= np.linalg.norm(c1) + 1e-9
            groups[ok] = lab
        self._groups = groups

    def save(self, path) -> None:
        np.savez_compressed(path, sum=self.sum, count=self.count, occupancy=self.occupancy,
                            size=np.array([self.w, self.h]), grid=np.array([self.gx, self.gy]))

    @classmethod
    def load(cls, path, width: int, height: int, **kw) -> "FlowField":
        d = np.load(path)
        ff = cls(width, height, grid=tuple(int(g) for g in d["grid"]), **kw)
        ff.sum, ff.count, ff.occupancy = d["sum"], d["count"], d["occupancy"]
        return ff
