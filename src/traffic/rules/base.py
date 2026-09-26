"""Shared types for the rule checker."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene import Scene
from ..signals import SignalTimeline
from ..tracks import TrackStore, Trajectory


@dataclass
class RawEvent:
    label: str
    start: float
    end: float
    tracks: tuple = ()
    score: float = 1.0
    note: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "start": round(self.start, 3), "end": round(self.end, 3),
                "tracks": [int(x) for x in self.tracks], "score": round(float(self.score), 3), "note": self.note}


@dataclass
class Context:
    store: TrackStore
    scene: Scene
    signals: SignalTimeline
    params: dict
    duration: float
    fps_eff: float
    _by_time: dict = field(default_factory=dict)

    @property
    def trajectories(self) -> list[Trajectory]:
        return self.store.trajectories

    @property
    def px(self) -> float:
        """Pixel thresholds in params are for 1080p; this rescales them to the video's resolution."""
        return self.scene.height / 1080.0

    def rule(self, name: str) -> dict:
        return self.params["rules"][name]

    def active(self, t: float) -> list[tuple[Trajectory, int]]:
        """(trajectory, sample index) for all tracks alive at time t (memoised)."""
        key = round(t, 3)
        if key not in self._by_time:
            self._by_time[key] = self.store.active_at(t)
        return self._by_time[key]


def speed_window(tr: Trajectory, t0: float, t1: float, reducer=np.max) -> float | None:
    m = (tr.t >= t0) & (tr.t <= t1)
    return float(reducer(tr.speed[m])) if m.any() else None
