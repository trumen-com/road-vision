"""Time-to-collision between two road users from image-space boxes and velocities.

Positions are the boxes' bottom-centre (ground contact) points. The relative
position is normalised by the pair's combined half-extent - horizontally by
width, vertically by a compressed height, because image depth is foreshortened -
so "contact" is simply the unit circle. TTC is the first time the linearly
extrapolated relative position enters that circle.
"""
from __future__ import annotations

import math

import numpy as np


def normalised_offset(box_a, box_b) -> tuple[np.ndarray, np.ndarray]:
    wa, ha = box_a[2] - box_a[0], box_a[3] - box_a[1]
    wb, hb = box_b[2] - box_b[0], box_b[3] - box_b[1]
    sx = max(0.5 * (wa + wb), 1.0)
    sy = max(0.25 * (ha + hb), 1.0)
    pa = np.array([(box_a[0] + box_a[2]) / 2, box_a[3]])
    pb = np.array([(box_b[0] + box_b[2]) / 2, box_b[3]])
    return (pb - pa) / np.array([sx, sy]), np.array([sx, sy])


def ellipse_distance(box_a, box_b) -> float:
    """< 1 means the ground footprints touch (approximately)."""
    p, _ = normalised_offset(box_a, box_b)
    return float(np.hypot(*p))


def time_to_collision(box_a, vel_a, box_b, vel_b) -> float:
    """Seconds until contact under constant velocity; 0 if already touching, inf if never."""
    p, s = normalised_offset(box_a, box_b)
    v = (np.asarray(vel_b, dtype=float) - np.asarray(vel_a, dtype=float)) / s
    c = float(p @ p) - 1.0
    if c <= 0:
        return 0.0
    a = float(v @ v)
    b = 2.0 * float(p @ v)
    if a < 1e-9 or b >= 0:
        return math.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return math.inf
    t = (-b - math.sqrt(disc)) / (2 * a)
    return t if t >= 0 else math.inf
