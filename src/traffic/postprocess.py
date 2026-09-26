"""Raw rule hits -> final segments.

Per class: union overlapping hits (the task annotates simultaneous same-class
events as one segment), merge small gaps, drop blips, shift boundaries by the
per-class offsets tuned against our dev labels, clip to the video.
"""
from __future__ import annotations

from .geometry import clean_intervals
from .rules import RawEvent


def class_post(params: dict, label: str) -> dict:
    cfg = dict(params["post"]["default"])
    cfg.update(params["post"].get(label, {}))
    return cfg


def to_segments(raw: list[RawEvent], params: dict, duration: float, classes: list[str]) -> list[list]:
    out = []
    for label in classes:
        ivs = [(e.start, e.end) for e in raw if e.label == label]
        if not ivs:
            continue
        cfg = class_post(params, label)
        shifted = [(max(0.0, s + cfg["start_offset"]), min(duration, e + cfg["end_offset"]))
                   for s, e in clean_intervals(ivs, cfg["merge_gap"], cfg["min_len"])]
        for s, e in clean_intervals(shifted, 0.0, 0.05):     # offsets must not create same-class overlaps
            out.append([round(s, 3), round(e, 3), label])
    out.sort(key=lambda x: (x[0], x[2]))
    return out
