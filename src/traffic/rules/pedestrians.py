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


def _nearby_flow(ctx: Context, xy, rings: int = 4):
    """Learned traffic direction at xy, or at the closest grid cell around it that has a coherent one
    (the kerb lane itself often has too few passing cars)."""
    ff = ctx.scene.flow
    if ff is None:
        return None
    cw, ch = ctx.scene.width / ff.gx, ctx.scene.height / ff.gy
    for r in range(rings + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                d = ff.direction(xy[0] + dx * cw, xy[1] + dy * ch)
                if d is not None:
                    return d
    return None


def jaywalking(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("jaywalking")
    if p["require_crosswalk_map"] and not ctx.scene.crosswalks:
        return []
    events = []
    for tr in ctx.trajectories:
        if not tr.is_person:
            continue
        ped = _pedestrian_mask(ctx, tr, p["min_scale_px"] * ctx.px)
        # out on the carriageway (not at a kerb, island or verge edge) and away from any zebra crossing
        flag = np.array([ped[i] and ctx.scene.deep_on_road(*tr.foot[i], p["road_depth"] * tr.scale[i])
                         and not ctx.scene.in_crosswalk(*tr.foot[i], p["crosswalk_margin"] * tr.scale[i])
                         for i in range(len(tr.t))])
        for s, e in time_runs(tr.t, flag, p["fill_gap"], p["min_seconds"]):
            m = (tr.t >= s) & (tr.t <= e)
            path = tr.foot[m]
            if len(path) < 2:
                continue
            # actually crossing: movement ACROSS the traffic direction (walking along the kerb is not jaywalking)
            d = path[-1] - path[0]
            mid = path[len(path) // 2]
            lane = ctx.scene.legal_direction(*mid)
            if lane is None:          # no lane drawn here: use the learned traffic direction nearby
                lane = _nearby_flow(ctx, mid)
            across = abs(float(d[0] * lane[1] - d[1] * lane[0])) if lane is not None else float(np.linalg.norm(d))
            if across < p["min_disp_sizes"] * float(np.median(tr.scale[m])):
                continue
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
            # on the road part of the crossing, not waiting at its kerb end
            on = np.array([ped[i] and point_in_poly(cw, *tr.foot[i], p["ped_margin"] * tr.scale[i])
                           and not in_any(ctx.scene.exclude, *tr.foot[i])
                           and ctx.scene.deep_on_road(*tr.foot[i], p["road_depth"] * tr.scale[i]) for i in range(len(tr.t))])
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
                who, ks = [], []
                for ps, pe, ped in occupied:          # overlapping in time AND close to the vehicle's path while it moves
                    if ps > e or pe < s:
                        continue
                    for k in range(i, j + 1):
                        if tr.speed[k] < p["vehicle_min_speed"] or not (ps <= tr.t[k] <= pe):
                            continue
                        m = ped.index_at(float(tr.t[k]))
                        if m is not None and np.linalg.norm(ped.foot[m] - tr.foot[k]) < p["near_widths"] * width:
                            ks.append(k)
                            if ped.id not in who:
                                who.append(ped.id)
                if who:
                    # the pass over the crossing around the moment it came close (not the whole time it stood on it)
                    s2 = max(s, float(tr.t[min(ks)]) - 1.0)
                    e2 = min(e, float(tr.t[max(ks)]) + 1.0)
                    events.append(RawEvent("failure_to_yield", s2, max(e2, s2 + 1.0), (tr.id, *who), 1.0))
    return events
