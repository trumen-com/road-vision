"""Rules about people: jaywalking, failure_to_yield."""
from __future__ import annotations

import numpy as np

from ..config import TWO_WHEELERS
from ..geometry import box_iou, in_any, point_in_poly, runs, time_runs
from .base import Context, RawEvent


def _is_rider_or_passenger(ctx: Context, tr, i: int) -> bool:
    """A 'person' riding a bike / motorbike, or seen inside a vehicle, is not a pedestrian."""
    pb = tr.box[i]
    fx, fy = tr.foot[i]
    for other, k in ctx.active(tr.t[i]):
        if other is tr or other.is_person:
            continue
        ob = other.box[k]
        if other.cls in TWO_WHEELERS:
            grown = np.array([ob[0] - 0.2 * (ob[2] - ob[0]), ob[1] - (ob[3] - ob[1]), ob[2] + 0.2 * (ob[2] - ob[0]), ob[3] + 5])
            if box_iou(pb, ob) > 0.1 or (grown[0] <= fx <= grown[2] and grown[1] <= fy <= grown[3]):
                return True
        elif other.is_vehicle and ob[0] <= fx <= ob[2] and ob[1] <= fy <= ob[3] - 0.05 * (ob[3] - ob[1]):
            return True
    return False


def _pedestrian_mask(ctx: Context, tr, min_scale: float) -> np.ndarray:
    return np.array([tr.scale[i] >= min_scale and not _is_rider_or_passenger(ctx, tr, i) for i in range(len(tr.t))])


def jaywalking(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("jaywalking")
    if p["require_crosswalk_map"] and not ctx.scene.crosswalks:
        return []
    events = []
    for tr in ctx.trajectories:
        if not tr.is_person:
            continue
        ped = _pedestrian_mask(ctx, tr, p["min_scale_px"] * ctx.px)
        flag = np.array([ped[i] and ctx.scene.on_road(*tr.foot[i])
                         and not ctx.scene.in_crosswalk(*tr.foot[i], p["crosswalk_margin"] * tr.scale[i])
                         for i in range(len(tr.t))])
        for s, e in time_runs(tr.t, flag, p["fill_gap"], p["min_seconds"]):
            events.append(RawEvent("jaywalking", s, e, (tr.id,), 1.0))
    return events


def failure_to_yield(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("failure_to_yield")
    if not ctx.scene.crosswalks:
        return []
    events = []
    peds = [tr for tr in ctx.trajectories if tr.is_person]
    for cw in ctx.scene.crosswalks:
        # time intervals with a pedestrian on (or stepping onto) this crossing
        occupied = []
        for tr in peds:
            ped = _pedestrian_mask(ctx, tr, 8 * ctx.px)
            on = np.array([ped[i] and point_in_poly(cw, *tr.foot[i], p["ped_margin"] * tr.scale[i])
                           and not in_any(ctx.scene.exclude, *tr.foot[i]) for i in range(len(tr.t))])
            occupied += [(float(tr.t[i]), float(tr.t[j]), tr) for i, j in runs(on)]
        if not occupied:
            continue
        for tr in ctx.trajectories:
            if not tr.is_vehicle:
                continue
            inside = np.array([point_in_poly(cw, *f) for f in tr.foot])
            for i, j in runs(inside):
                if tr.speed[i:j + 1].max() < p["vehicle_min_speed"]:
                    continue
                s, e = float(tr.t[i]), float(tr.t[j])
                width = float(np.median(tr.box[i:j + 1, 2] - tr.box[i:j + 1, 0]))
                who = []
                for ps, pe, ped in occupied:          # overlapping in time AND close to the vehicle's path
                    if ps > e or pe < s:
                        continue
                    for k in range(i, j + 1):
                        m = ped.index_at(float(tr.t[k]))
                        if m is not None and np.linalg.norm(ped.foot[m] - tr.foot[k]) < p["near_widths"] * width:
                            who.append(ped.id)
                            break
                if who:
                    events.append(RawEvent("failure_to_yield", s, max(e, s + 0.2), (tr.id, *who), 1.0))
    return events
