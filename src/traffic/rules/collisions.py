"""accident and near_miss from trajectory dynamics.

An accident is a contact between two road users followed by an impact
signature: at least one of them loses most of its speed abruptly (much faster
than normal braking) or a stationary one is suddenly shoved / spun, and the
involved objects then come to rest. A single-vehicle impact is an abrupt stop
from high speed with nobody queued ahead. A near miss is a hard brake or a
swerve while another road user is on a collision course (low TTC) but no
contact follows.
"""
from __future__ import annotations

import numpy as np

from ..config import ROAD_USERS
from ..conflict import ellipse_distance, time_to_collision
from ..geometry import angle_diff, box_iou, moving_average
from .base import Context, RawEvent


def _fast_speed(tr, fps_eff: float) -> np.ndarray:
    """Lightly smoothed speed (sizes/s) that keeps abrupt changes sharp."""
    foot = moving_average(np.stack([(tr.box[:, 0] + tr.box[:, 2]) / 2, tr.box[:, 3]], 1), 3)
    if len(tr.t) < 2:
        return np.zeros(len(tr.t))
    v = np.stack([np.gradient(foot[:, j], tr.t) for j in range(2)], 1)
    return np.hypot(v[:, 0], v[:, 1]) / tr.scale


def _window(tr, arr, t0, t1, fn, coverage: float = 0.0):
    """fn(arr) over samples in [t0, t1]; None if the track covers less than `coverage` of the window."""
    m = (tr.t >= t0) & (tr.t <= t1)
    if not m.any():
        return None
    if coverage and (min(t1, tr.t[-1]) - max(t0, tr.t[0])) < coverage * (t1 - t0):
        return None
    return float(fn(arr[m]))


def _shape_stable(tr, t0: float, t1: float, t2: float, tol: float = 0.35) -> bool:
    """Box area before [t0, t1] and after [t1, t2] agree: a speed drop is not a box being cut by occlusion."""
    area = (tr.box[:, 2] - tr.box[:, 0]) * (tr.box[:, 3] - tr.box[:, 1])
    a0 = _window(tr, area, t0, t1, np.median)
    a1 = _window(tr, area, t1, t2, np.median)
    return a0 is not None and a1 is not None and abs(np.log(a1 / max(a0, 1.0))) < tol


def _continuous(tr, t0: float, t1: float, max_gap: float = 0.4) -> bool:
    """The track was observed without a gap longer than max_gap in [t0, t1] (not lost behind another object).
    The limit widens to 3 sampling steps when frames are sampled sparsely (CPU stride)."""
    if len(tr.t) > 2:
        max_gap = max(max_gap, 3.0 * float(np.median(np.diff(tr.t))))
    ts = tr.t[(tr.t >= t0) & (tr.t <= t1)]
    if len(ts) < 2:
        return False
    edges = np.concatenate([[max(t0, tr.t[0])], ts, [min(t1, tr.t[-1])]])
    return float(np.diff(edges).max()) <= max_gap


def _near_border(ctx: Context, tr) -> bool:
    x, y = tr.foot[-1]
    m = 0.04
    return x < m * ctx.scene.width or x > (1 - m) * ctx.scene.width or y > (1 - m) * ctx.scene.height or y < m * ctx.scene.height


def _impact(tr, fs, tc: float, p: dict) -> dict:
    """Impact signature of one participant around contact time tc.

    decel: speed collapses within ~0.8 s (normal braking takes seconds);
    shoved: a stationary participant is displaced by a real distance;
    rest: it then stays (almost) still.
    """
    before = _window(tr, fs, tc - 1.0, tc, np.max)
    after = _window(tr, fs, tc + 0.2, tc + 0.8, np.median)
    rest = _window(tr, tr.speed, tc + 1.0, tc + 1.0 + p["rest_seconds"], np.min, coverage=0.8)
    res = {"moving": before is not None and before >= p["pre_moving_speed"], "decel": False, "shoved": False,
           "rest": rest is not None and rest < p["stop_speed"], "gone": False}
    if res["moving"] and after is not None:
        res["decel"] = (after <= (1 - p["decel_ratio"]) * before and _shape_stable(tr, tc - 1.0, tc, tc + 1.0)
                        and _continuous(tr, tc - 1.0, tc + 0.8))   # a speed "collapse" measured across an occlusion gap is not evidence
    # a stationary participant that is suddenly displaced by a real distance
    still = _window(tr, tr.speed, tc - 2.0, tc - 0.5, np.max)
    i0, i1 = tr.index_at(tc), tr.index_at(min(tc + 1.0, tr.t[-1]))
    if still is not None and still < p["stop_speed"] * 2 and i0 is not None and i1 is not None:
        res["shoved"] = np.linalg.norm(tr.foot[i1] - tr.foot[i0]) > 0.4 * tr.scale[i0]
    return res


def _in_jam(ctx: Context, involved, tc: float, p: dict) -> bool:
    """True when the neighbourhood is a stop-and-go queue: most nearby vehicles are slow too.

    Crashed vehicles stay put while traffic around them keeps moving; in a jam
    everybody brakes and touches in the image, which is not a crash.
    """
    ref = involved[0]
    k = ref.index_at(tc)
    if k is None:
        return False
    here, scale = ref.foot[k], ref.scale[k]
    speeds = []
    for o, j in ctx.active(tc + 1.5):
        if o in involved or not o.is_vehicle:
            continue
        if np.linalg.norm(o.foot[j] - here) < p["jam_radius_sizes"] * scale:
            speeds.append(o.speed[j])
    return len(speeds) >= 2 and float(np.median(speeds)) < p["jam_speed"]


def _rest_time(tr, tc: float, stop_speed: float, max_t: float) -> float:
    """First time after tc from which the object stays below stop_speed for 1 s (or leaves the frame)."""
    idx = np.nonzero(tr.t > tc)[0]
    for k in idx:
        m = (tr.t >= tr.t[k]) & (tr.t <= tr.t[k] + 1.0)
        if (tr.speed[m] < stop_speed).all():
            return float(tr.t[k])
    return float(min(tr.t[-1], max_t))


def _contacts(ctx: Context, p: dict):
    """Yield (a, b, t_contact) for the first sample of every contact episode between road users."""
    users = [tr for tr in ctx.trajectories if tr.cls in ROAD_USERS]
    for ia, a in enumerate(users):
        for b in users[ia + 1:]:
            if a.is_person and b.is_person:
                continue
            t0, t1 = max(a.t[0], b.t[0]) + p["min_history"], min(a.t[-1], b.t[-1])
            if t1 - t0 < 0.3 or min(np.median(a.scale), np.median(b.scale)) < p["min_scale_px"] * ctx.px:
                continue
            ts = a.t[(a.t >= t0) & (a.t <= t1)]
            prev = False
            for t in ts:
                ia_, ib_ = a.index_at(t), b.index_at(t)
                ba, bb = a.box[ia_], b.box[ib_]
                hmin = min(ba[3] - ba[1], bb[3] - bb[1])
                same_depth = abs(ba[3] - bb[3]) < p["same_depth_frac"] * hmin
                iou = box_iou(ba, bb)
                touching = (ellipse_distance(ba, bb) < 1.0 or (iou > 0.05 and same_depth)) and iou < 0.6  # >0.6: duplicate box
                if touching and not prev and ctx.scene.on_road(*((a.foot[ia_] + b.foot[ib_]) / 2)):
                    yield a, b, float(t)       # (bus-stop manoeuvres, parking and sidewalks are excluded areas)
                prev = touching


def accident(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("accident")
    fast = {tr.id: _fast_speed(tr, ctx.fps_eff) for tr in ctx.trajectories}
    events = []
    for a, b, tc in _contacts(ctx, p):
        ia, ib = _impact(a, fast[a.id], tc, p), _impact(b, fast[b.id], tc, p)
        ia["gone"] = a.t[-1] < tc + 1.5 and _near_border(ctx, a)     # left the frame, not lost under occlusion
        ib["gone"] = b.t[-1] < tc + 1.5 and _near_border(ctx, b)
        if not (ia["moving"] or ib["moving"]):
            continue                       # two parked objects overlapping in the image
        if not (ia["decel"] or ib["decel"] or ia["shoved"] or ib["shoved"]):
            continue                       # no impact signature: overlap in the image only
        # closing speed before contact: were they actually approaching each other?
        ka, kb = a.index_at(tc - 0.5), b.index_at(tc - 0.5)
        closing = False
        if ka is not None and kb is not None:
            rel = np.linalg.norm(a.vel[ka] - b.vel[kb]) / max(min(a.scale[ka], b.scale[kb]), 4.0)
            closing = rel >= p["closing_speed"] and time_to_collision(a.box[ka], a.vel[ka], b.box[kb], b.vel[kb]) < 1.5
        if not closing or _in_jam(ctx, (a, b), tc, p):
            continue
        score = (0.5 + 0.25 * ((ia["decel"] or ib["decel"]) and (ia["shoved"] or ib["shoved"]))
                 + 0.25 * ((ia["rest"] or ia["gone"]) and (ib["rest"] or ib["gone"])))
        if score >= p["min_score"]:
            end = max(_rest_time(a, tc, p["stop_speed"], tc + p["max_seconds"]),
                      _rest_time(b, tc, p["stop_speed"], tc + p["max_seconds"]))
            events.append(RawEvent("accident", tc, min(max(end, tc + 1.0), tc + p["max_seconds"]), (a.id, b.id), score))

    if p["single_vehicle"]:
        for tr in ctx.trajectories:
            if not tr.is_vehicle or len(tr.t) < 4 or np.median(tr.scale) < p["min_scale_px"] * ctx.px:
                continue
            fs = fast[tr.id]
            for k in range(1, len(tr.t)):
                tc = float(tr.t[k])
                before = _window(tr, tr.speed, tc - 1.0, tc, np.median, coverage=0.9)
                after = _window(tr, fs, tc + 0.2, tc + 0.8, np.max)
                rest = _window(tr, tr.speed, tc + 1.0, tc + 1.0 + p["rest_seconds"], np.min, coverage=0.8)
                if before is None or after is None or rest is None or before < p["single_min_speed"]:
                    continue
                if after < 0.15 * before and rest < p["stop_speed"] and _shape_stable(tr, tc - 1.0, tc, tc + 1.0) \
                        and not _in_jam(ctx, (tr,), tc, p):
                    # ignore normal queueing: someone already stationary right in front
                    blocked = any(o is not tr and o.is_vehicle and o.speed[j] < p["stop_speed"]
                                  and ellipse_distance(tr.box[k], o.box[j]) < 2.5 for o, j in ctx.active(tc))
                    if not blocked:
                        end = _rest_time(tr, tc, p["stop_speed"], tc + p["max_seconds"])
                        events.append(RawEvent("accident", tc, max(end, tc + 1.0), (tr.id,), 0.6, "single"))
                    break
    return events


def near_miss(ctx: Context, accidents: list[RawEvent]) -> list[RawEvent]:
    p = ctx.rule("near_miss")
    crash_ids = {(tid, round(ev.start)) for ev in accidents for tid in ev.tracks}
    events = []
    for tr in ctx.trajectories:
        if not tr.is_vehicle or len(tr.t) < 5 or np.median(tr.scale) < p["min_scale_px"] * ctx.px:
            continue
        yaw = np.gradient(np.unwrap(tr.heading), tr.t)
        k = 0
        while k < len(tr.t):
            t = float(tr.t[k])
            if t - tr.t[0] < p["min_history"] or tr.scale[k] < p["min_scale_px"] * ctx.px or not ctx.scene.on_road(*tr.foot[k]):
                k += 1
                continue
            base = _window(tr, tr.speed, t - 0.5, t, np.max)
            if base is None or base < p["min_speed"]:
                k += 1
                continue
            low = _window(tr, tr.speed, t + 0.3, t + p["brake_window"], np.max, coverage=0.9)
            braking = low is not None and (base - low) >= max(p["brake_ratio"] * base, p["brake_abs"]) \
                and _shape_stable(tr, t - 0.6, t, t + p["brake_window"] + 0.4) \
                and _continuous(tr, t - 0.6, t + p["brake_window"])
            swerve = p["swerve_deg_s"] > 0 and abs(np.degrees(yaw[k])) > p["swerve_deg_s"] and tr.speed[k] > 2 * p["min_speed"]
            if not (braking or swerve) or _in_jam(ctx, (tr,), t + 0.5, ctx.rule("accident")):
                k += 1
                continue
            # somebody ahead on a collision course (same lane, in the direction of travel)
            heading = tr.vel[k] / (np.linalg.norm(tr.vel[k]) + 1e-9)
            partner, ttc_min = None, np.inf
            for o, j in ctx.active(t):
                if o is tr or o.cls not in ROAD_USERS or t - o.t[0] < 0.5:
                    continue
                r = o.foot[j] - tr.foot[k]
                along = float(r @ heading)
                lateral = abs(float(r[0] * heading[1] - r[1] * heading[0]))
                if along <= 0 or lateral > p["ahead_lateral"] * 0.5 * ((tr.box[k, 2] - tr.box[k, 0]) + (o.box[j, 2] - o.box[j, 0])):
                    continue
                if ellipse_distance(tr.box[k], o.box[j]) > p["max_gap"]:
                    continue
                # closing fast on it (braking behind a car in slow traffic is normal driving)
                closing = float((tr.vel[k] - o.vel[j]) @ heading) / max(tr.scale[k], 4.0)
                if closing < p["closing_min"]:
                    continue
                ttc = time_to_collision(tr.box[k], tr.vel[k], o.box[j], o.vel[j])
                if ttc < ttc_min:
                    partner, ttc_min = o, ttc
            if partner is None or ttc_min > p["ttc_max"] or ttc_min == 0.0 \
                    or any((tr.id, round(t + d)) in crash_ids for d in (-2, -1, 0, 1, 2)):
                k += 1
                continue
            end = t + 1.0
            for m in range(k + 1, len(tr.t)):
                j = partner.index_at(tr.t[m])
                if j is None or time_to_collision(tr.box[m], tr.vel[m], partner.box[j], partner.vel[j]) > 3.0:
                    end = float(tr.t[m]) + p["clear_seconds"]
                    break
            events.append(RawEvent("near_miss", t, end, (tr.id, partner.id), 0.7, "brake" if braking else "swerve"))
            k = int(np.searchsorted(tr.t, end)) + 1
    return events
