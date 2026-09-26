"""Trajectory store ("path diary") with offline kinematics for Part A.

Speeds are expressed in object sizes per second (|v| / sqrt(w*h)): a car and a
bus at different depths get comparable numbers without camera calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PERSON, VEHICLES
from .geometry import moving_average


@dataclass
class Trajectory:
    id: int
    cls: int
    t: np.ndarray            # (N,) seconds
    box: np.ndarray          # (N, 4) xyxy (raw)
    conf: np.ndarray         # (N,)
    # derived (filled by compute_kinematics)
    foot: np.ndarray = field(default=None)      # (N, 2) smoothed bottom-centre
    center: np.ndarray = field(default=None)    # (N, 2) smoothed centre
    scale: np.ndarray = field(default=None)     # (N,) sqrt(w*h), smoothed
    vel: np.ndarray = field(default=None)       # (N, 2) px/s of foot point
    speed: np.ndarray = field(default=None)     # (N,) sizes/s
    heading: np.ndarray = field(default=None)   # (N,) radians
    members: tuple = ()                         # original ids merged into this one

    @property
    def is_vehicle(self) -> bool:
        return self.cls in VEHICLES

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def index_at(self, t: float) -> int | None:
        """Index of the sample nearest to t if t is within the track's lifetime."""
        if t < self.t[0] - 1e-6 or t > self.t[-1] + 1e-6:
            return None
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


class TrackStore:
    """Collects per-frame tracker output, then builds smoothed trajectories."""

    def __init__(self):
        self._rows: dict[int, list] = {}
        self._cls: dict[int, dict[int, float]] = {}
        self.trajectories: list[Trajectory] = []
        self.sample_times: np.ndarray = np.zeros(0)
        self._times: list[float] = []

    def add(self, t: float, outputs) -> None:
        self._times.append(t)
        for tid, box, cls, conf in outputs:
            self._rows.setdefault(tid, []).append((t, *box, conf))
            votes = self._cls.setdefault(tid, {})
            votes[cls] = votes.get(cls, 0.0) + conf

    def finalize(self, params: dict, fps_eff: float, frame_size: tuple[int, int] | None = None) -> list[Trajectory]:
        """Build trajectories. Samples whose box is cut by the frame border are dropped: their
        bottom-centre sticks to the edge and would read as a vehicle braking to a halt."""
        self.sample_times = np.asarray(self._times)
        trajs = []
        for tid, rows in self._rows.items():
            a = np.asarray(rows, dtype=np.float64)
            if frame_size is not None:
                W, H = frame_size
                m = 3.0
                a = a[(a[:, 1] > m) & (a[:, 3] < W - m) & (a[:, 4] < H - m)]
            if len(a) < 2:
                continue
            cls = max(self._cls[tid].items(), key=lambda kv: (kv[1], -kv[0]))[0]
            trajs.append(Trajectory(tid, cls, a[:, 0], a[:, 1:5], a[:, 5], members=(tid,)))
        trajs = stitch(trajs, params["kinematics"])
        k = max(1, int(round(params["kinematics"]["smooth_seconds"] * fps_eff)))
        for tr in trajs:
            compute_kinematics(tr, k)
        self.trajectories = sorted(trajs, key=lambda tr: (tr.t[0], tr.id))
        return self.trajectories

    def active_at(self, t: float) -> list[tuple[Trajectory, int]]:
        out = []
        for tr in self.trajectories:
            i = tr.index_at(t)
            if i is not None:
                out.append((tr, i))
        return out

    # serialisation (cache for tuning and for the website) -------------------
    def to_dict(self) -> dict:
        return {"sample_times": self.sample_times.tolist(),
                "tracks": [{"id": tr.id, "cls": tr.cls, "t": tr.t.round(3).tolist(),
                            "box": tr.box.round(1).tolist(), "conf": tr.conf.round(3).tolist(),
                            "members": list(tr.members)} for tr in self.trajectories]}

    @classmethod
    def from_dict(cls, d: dict, params: dict, fps_eff: float) -> "TrackStore":  # used by tests and tools
        st = cls()
        st.sample_times = np.asarray(d["sample_times"])
        k = max(1, int(round(params["kinematics"]["smooth_seconds"] * fps_eff)))
        for r in d["tracks"]:
            tr = Trajectory(r["id"], r["cls"], np.asarray(r["t"]), np.asarray(r["box"]), np.asarray(r["conf"]),
                            members=tuple(r.get("members", [r["id"]])))
            compute_kinematics(tr, k)
            st.trajectories.append(tr)
        return st


def compute_kinematics(tr: Trajectory, k: int) -> None:
    b = tr.box
    w = np.clip(b[:, 2] - b[:, 0], 1, None)
    h = np.clip(b[:, 3] - b[:, 1], 1, None)
    foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
    center = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    tr.foot = moving_average(foot, k)
    tr.center = moving_average(center, k)
    tr.scale = np.clip(moving_average(np.sqrt(w * h), k), 4, None)
    if len(tr.t) >= 2:
        tr.vel = np.stack([np.gradient(tr.foot[:, j], tr.t) for j in range(2)], axis=1)
        tr.vel = moving_average(tr.vel, max(1, k // 2))
    else:
        tr.vel = np.zeros_like(tr.foot)
    tr.speed = np.hypot(tr.vel[:, 0], tr.vel[:, 1]) / tr.scale
    tr.heading = np.arctan2(tr.vel[:, 1], tr.vel[:, 0])


def stitch(trajs: list[Trajectory], p: dict) -> list[Trajectory]:
    """Greedy re-linking of fragmented tracks (occlusion / missed detections).

    Track B continues track A if B starts within `stitch_gap_seconds` after A ends,
    same object family, and B's first box is within `stitch_max_dist` sizes of A's
    constant-velocity extrapolation.
    """
    from .tracker import family

    trajs = sorted(trajs, key=lambda tr: tr.t[0])
    used = set()
    out = []
    for i, a in enumerate(trajs):
        if i in used:
            continue
        cur = a
        while True:
            end_t = cur.t[-1]
            ca = (cur.box[-1, :2] + cur.box[-1, 2:]) / 2
            n = min(5, len(cur.t))
            dtv = cur.t[-1] - cur.t[-n]
            v = ((cur.box[-1, :2] + cur.box[-1, 2:]) / 2 - (cur.box[-n, :2] + cur.box[-n, 2:]) / 2) / dtv if dtv > 0 else np.zeros(2)
            size = np.sqrt(max((cur.box[-1, 2] - cur.box[-1, 0]) * (cur.box[-1, 3] - cur.box[-1, 1]), 1.0))
            best, best_d = None, None
            for j in range(i + 1, len(trajs)):
                if j in used:
                    continue
                b = trajs[j]
                gap = b.t[0] - end_t
                if gap <= 0:
                    continue
                if gap > p["stitch_gap_seconds"]:
                    break
                if family(b.cls) != family(cur.cls):
                    continue
                cb = (b.box[0, :2] + b.box[0, 2:]) / 2
                d = np.linalg.norm(cb - (ca + v * gap)) / size
                if d <= p["stitch_max_dist"] and (best_d is None or d < best_d):
                    best, best_d = j, d
            if best is None:
                break
            b = trajs[best]
            used.add(best)
            cur = Trajectory(cur.id, cur.cls if len(cur.t) >= len(b.t) else b.cls,
                             np.concatenate([cur.t, b.t]), np.concatenate([cur.box, b.box]),
                             np.concatenate([cur.conf, b.conf]), members=cur.members + b.members)
        out.append(cur)
    return out
