"""Rules that need the traffic-light state: red_light, stop_line."""
from __future__ import annotations

import numpy as np

from ..geometry import proj_param, runs, signed_side
from ..signals import AMBER
from .base import Context, RawEvent


def _front_points(tr, direction: np.ndarray, front_frac: float) -> np.ndarray:
    """Approximate ground point of the vehicle's front, given its travel direction in the image."""
    w = tr.box[:, 2] - tr.box[:, 0]
    h = tr.box[:, 3] - tr.box[:, 1]
    fx = tr.foot[:, 0] + 0.5 * w * direction[0]
    fy = tr.foot[:, 1] - front_frac * h * max(0.0, -float(direction[1]))
    return np.stack([fx, fy], axis=1)


def _progress(sl, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance past the stop line (positive = beyond it in the legal direction) and position along it."""
    mid = (sl.a + sl.b) / 2
    orient = np.sign(signed_side(sl.a, sl.b, mid + sl.direction * 10)) or 1.0
    dist = np.array([signed_side(sl.a, sl.b, p) * orient for p in pts])
    along = np.array([proj_param(sl.a, sl.b, p) for p in pts])
    return dist, along


def _is_red(sl, state: str, amber_is_red: bool) -> bool:
    return state in sl.red_states or (amber_is_red and state == AMBER and "red" in sl.red_states)


def red_light(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("red_light")
    if not ctx.scene.stop_lines or not ctx.signals.states:
        return []
    events = []
    for sl in ctx.scene.stop_lines:
        for tr in ctx.trajectories:
            if not tr.is_vehicle or len(tr.t) < 3:
                continue
            front = _front_points(tr, sl.direction, p["front_frac"])
            dist, along = _progress(sl, front)
            moving_along = (tr.vel @ sl.direction) > 0
            for k in range(1, len(dist)):
                if dist[k - 1] < 0 <= dist[k] and -0.1 <= along[k] <= 1.1 and moving_along[k]:
                    # a real crossing: moving, and clearly beyond the line within 2 s (not box jitter at the line)
                    ahead = (tr.t > tr.t[k]) & (tr.t <= tr.t[k] + 2.0)
                    if tr.speed[k] < p["min_speed"] or not ahead.any() or dist[ahead].max() < p["min_advance_sizes"] * tr.scale[k]:
                        continue
                    t_cross = float(tr.t[k])
                    if not _is_red(sl, ctx.signals.state(sl.signal, t_cross), p["amber_is_red"]):
                        break
                    end = float(tr.t[-1])
                    was_in = False
                    for j in range(k, len(tr.t)):
                        inside = ctx.scene.in_intersection(*tr.foot[j])
                        was_in |= inside
                        if (was_in and not inside) or (ctx.scene.intersection is None and dist[j] > p["exit_sizes"] * tr.scale[j]):
                            end = float(tr.t[j])
                            break
                    events.append(RawEvent("red_light", t_cross, min(end, t_cross + p["max_seconds"]), (tr.id,), 1.0, sl.id))
                    break
    return events


def stop_line(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("stop_line")
    if not ctx.scene.stop_lines or not ctx.signals.states:
        return []
    events = []
    for sl in ctx.scene.stop_lines:
        for tr in ctx.trajectories:
            if not tr.is_vehicle:
                continue
            front = _front_points(tr, sl.direction, ctx.rule("red_light")["front_frac"])
            dist, along = _progress(sl, front)
            past = (dist > p["past_sizes"] * tr.scale) & (dist < p["max_past_sizes"] * tr.scale) & (along > -0.1) & (along < 1.1)
            stopped = (tr.speed < p["stop_speed"]) & past
            for i, j in runs(stopped):
                if tr.t[j] - tr.t[i] < p["min_stop_seconds"] or ctx.scene.in_intersection(*tr.foot[i]):
                    continue
                s = float(tr.t[i])
                if ctx.signals.state(sl.signal, s) not in sl.red_states:
                    continue
                green = ctx.signals.next_exit(sl.signal, s, sl.red_states)
                e = green if green is not None else float(tr.t[j])
                events.append(RawEvent("stop_line", s, max(e, s + 0.5), (tr.id,), 1.0, sl.id))
    return events
