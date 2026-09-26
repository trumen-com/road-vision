"""Rules about things that do not move: stopped_vehicle, congestion, road_obstacle."""
from __future__ import annotations

import numpy as np

from ..geometry import box_iou, clean_intervals, proj_param, runs, signed_side, time_runs
from ..signals import GREEN, UNKNOWN
from .base import Context, RawEvent


def _approach_direction(ctx: Context, tr, i: int) -> np.ndarray | None:
    """Direction the vehicle was travelling before stopping, else the legal direction at its spot."""
    j0 = max(0, i - int(3 * ctx.fps_eff))
    mv = tr.speed[j0:i + 1] > 0.5
    if mv.any():
        v = tr.vel[j0:i + 1][mv].mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            return v / n
    return ctx.scene.legal_direction(*tr.foot[i])


def _queued_at(ctx: Context, tr, i: int, direction: np.ndarray | None, p: dict) -> bool:
    """A stationary vehicle is "queued" if another stationary vehicle is right ahead of or behind it,
    or it waits at a stop line while its signal is not green."""
    t = tr.t[i]
    foot, scale = tr.foot[i], tr.scale[i]
    if direction is not None:
        for other, k in ctx.active(t):
            if other is tr or not other.is_vehicle or other.speed[k] > p["stop_speed"] * 2:
                continue
            r = other.foot[k] - foot
            along = float(r @ direction)
            lateral = abs(float(r[0] * direction[1] - r[1] * direction[0]))
            if 0 < abs(along) < p["queue_ahead_sizes"] * scale and lateral < 0.9 * scale:   # a car right ahead or right behind: a queue
                return True
    for sl in ctx.scene.stop_lines:
        # waiting just upstream of a stop line is queueing, whatever the (partly unobservable) signal says
        u = proj_param(sl.a, sl.b, foot)
        d = signed_side(sl.a, sl.b, foot) * (np.sign(signed_side(sl.a, sl.b, (sl.a + sl.b) / 2 + sl.direction * 10)) or 1.0)
        if -0.15 <= u <= 1.15 and -p["stopline_sizes"] * scale <= d <= 0.5 * scale:
            return True
    return False


def stopped_vehicle(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("stopped_vehicle")
    # 1. stationary runs per track
    spots = []
    for tr in ctx.trajectories:
        if not tr.is_vehicle:
            continue
        for i, j in runs(tr.speed < p["stop_speed"]):
            if tr.t[j] - tr.t[i] < 2.0:
                continue
            spots.append({"s": tr.t[i], "e": tr.t[j], "box": np.median(tr.box[i:j + 1], axis=0),
                          "parts": [(tr, i, j)]})
    # 2. merge runs at the same spot (the tracker may re-identify a parked car several times)
    spots.sort(key=lambda d: d["s"])
    merged: list[dict] = []
    for sp in spots:
        for m in merged:
            if box_iou(m["box"], sp["box"]) >= p["spot_iou"] and sp["s"] - m["e"] <= p["spot_gap"] and sp["e"] >= m["s"] - p["spot_gap"]:
                m["s"], m["e"] = min(m["s"], sp["s"]), max(m["e"], sp["e"])
                m["parts"] += sp["parts"]
                break
        else:
            merged.append(dict(sp))
    # 3. keep >= 10 s on the carriageway, not queued at a signal / behind another car
    events = []
    for m in merged:
        if m["e"] - m["s"] < p["min_seconds"]:
            continue
        x = (m["box"][0] + m["box"][2]) / 2
        y = m["box"][3]
        if not ctx.scene.on_road(x, y) or np.sqrt((m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1])) < p["min_scale_px"] * ctx.px:
            continue
        checks = queued = 0
        for tr, i, j in m["parts"]:
            d = _approach_direction(ctx, tr, i)
            step = max(1, int(ctx.fps_eff))            # check about once per second
            for k in range(i, j + 1, step):
                checks += 1
                queued += _queued_at(ctx, tr, k, d, p)
        if checks and queued / checks >= p["queue_fraction"]:
            continue
        ids = tuple(tr.id for tr, _, _ in m["parts"])
        events.append(RawEvent("stopped_vehicle", m["s"], m["e"], ids, 1.0 - queued / max(checks, 1)))
    return events


def congestion(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("congestion")
    dt = p["bin_seconds"]
    bins = np.arange(0.0, ctx.duration, dt)
    per_group: dict[str, np.ndarray] = {}
    st = ctx.store.sample_times
    if len(st) == 0:
        return []
    for b, tb in enumerate(bins):
        groups: dict[str, list[float]] = {}
        t_near = float(st[min(np.searchsorted(st, tb), len(st) - 1)])
        for tr, i in ctx.active(t_near):
            if not tr.is_vehicle or not ctx.scene.on_road(*tr.foot[i]):
                continue
            g = ctx.scene.direction_group(*tr.foot[i])
            if g is None:
                continue
            groups.setdefault(g, []).append(float(tr.speed[i]))
        for g, speeds in groups.items():
            arr = per_group.setdefault(g, np.zeros(len(bins), dtype=bool))
            s = np.asarray(speeds)
            arr[b] = len(s) >= p["min_vehicles"] and (s < p["crawl_speed"]).mean() >= p["slow_fraction"] \
                and np.median(s) < p["crawl_speed"]
    events = []
    enter, exit_ = int(round(p["enter_seconds"] / dt)), int(round(p["exit_seconds"] / dt))
    for g, raw in per_group.items():
        state, start, on_count, off_count = False, 0, 0, 0
        ivs = []
        for b, c in enumerate(raw):
            on_count = on_count + 1 if c else 0
            off_count = off_count + 1 if not c else 0
            if not state and on_count >= enter:
                state, start = True, b - enter + 1
            elif state and off_count >= exit_:
                state = False
                ivs.append((bins[start], bins[b - exit_ + 1]))
        if state:
            ivs.append((bins[start], ctx.duration))
        for s, e in clean_intervals(ivs, 0.0, p["min_seconds"]):
            if ctx.signals.states:     # a signal queue releases on green: a jam must persist through green time
                green_secs = sum(ctx.signals.state(None, x) == GREEN for x in np.arange(s, e, 1.0))
                if green_secs < p["min_green_seconds"]:
                    continue
            events.append(RawEvent("congestion", s, e, (), 1.0, g))
    return events


def road_obstacle(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("road_obstacle")
    events = []
    for tr in ctx.trajectories:
        if tr.cls not in p["animal_classes"]:
            continue
        on = np.array([ctx.scene.on_road(*f) for f in tr.foot])
        for s, e in time_runs(tr.t, on, 1.0, p["min_seconds"]):
            events.append(RawEvent("road_obstacle", s, e, (tr.id,), 0.8, "animal"))
    return events
