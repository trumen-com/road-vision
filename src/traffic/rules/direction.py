"""Rules about where vehicles go: wrong_way, illegal_u_turn, illegal_turn, solid_line_crossing."""
from __future__ import annotations

import numpy as np

from ..geometry import angle_diff, in_any, moving_average, proj_param, signed_side, time_runs
from .base import Context, RawEvent


def wrong_way(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("wrong_way")
    events = []
    for tr in ctx.trajectories:
        if not tr.is_vehicle or len(tr.t) < 3:
            continue
        flag = np.zeros(len(tr.t), dtype=bool)
        for i in range(len(tr.t)):
            if tr.speed[i] < p["min_speed"] or ctx.scene.in_intersection(*tr.foot[i]):
                continue                     # inside the junction every direction is legal for someone
            d = ctx.scene.legal_direction(*tr.foot[i])
            if d is None:
                continue
            v = tr.vel[i] / (np.linalg.norm(tr.vel[i]) + 1e-9)
            flag[i] = float(v @ d) < p["cos_thresh"]
        for s, e in time_runs(tr.t, flag, p["fill_gap"], p["min_seconds"]):
            i0, i1 = int(np.searchsorted(tr.t, s)), int(np.searchsorted(tr.t, e))
            disp = tr.foot[i1] - tr.foot[i0]
            d = ctx.scene.legal_direction(*tr.foot[(i0 + i1) // 2])
            scale = float(np.median(tr.scale[i0:i1 + 1]))
            # the net displacement, not just noisy per-sample velocity, must go against traffic
            if d is None or scale < p["min_scale_px"] * ctx.px or np.linalg.norm(disp) < p["min_disp_sizes"] * scale \
                    or float(disp @ d) / (np.linalg.norm(disp) + 1e-9) > p["cos_thresh"]:
                continue
            if tr.t[-1] - e < 1.0:           # still wrong-way when it leaves the frame
                e = float(tr.t[-1])
            events.append(RawEvent("wrong_way", s, e, (tr.id,), 1.0))
    return events


def _direction_change(tr, min_speed: float, window: float, min_disp: float):
    """Travel direction over the first and last `window` seconds of motion, and the turn interval between.

    Uses net displacements rather than per-sample headings, so box jitter on slow
    or far vehicles cannot add up to a phantom U-turn.
    """
    mv = np.nonzero(tr.speed > min_speed)[0]
    if len(mv) < 4:
        return None
    t0, t1 = tr.t[mv[0]], tr.t[mv[-1]]
    if t1 - t0 < 2 * window:
        return None
    a0, a1 = mv[0], int(np.searchsorted(tr.t, t0 + window))
    b0, b1 = int(np.searchsorted(tr.t, t1 - window)), mv[-1]
    d0, d1 = tr.foot[a1] - tr.foot[a0], tr.foot[b1] - tr.foot[b0]
    scale = float(np.median(tr.scale))
    if np.linalg.norm(d0) < min_disp * scale or np.linalg.norm(d1) < min_disp * scale:
        return None
    ang = np.degrees(np.arccos(np.clip(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1)), -1, 1)))
    h0, h1 = np.arctan2(d0[1], d0[0]), np.arctan2(d1[1], d1[0])
    return ang, h0, h1, a1, b0


def turns(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("turns")
    settle = np.radians(p["settle_deg"])
    events = []
    for tr in ctx.trajectories:
        if not tr.is_vehicle or tr.duration < 3.0 or np.median(tr.scale) < p["min_scale_px"] * ctx.px:
            continue
        res = _direction_change(tr, p["min_speed"], p["window_seconds"], p["min_disp_sizes"])
        if res is None:
            continue
        total, h0, h1, i_a, i_b = res
        # turn interval: heading leaves the entry direction ... settles on the exit direction
        hd = tr.heading
        mv = tr.speed > p["min_speed"]
        idx = [k for k in range(i_a, i_b + 1) if mv[k]]
        away = [k for k in idx if abs(angle_diff(hd[k], h0)) > settle]
        if away:
            back = [k for k in idx if k > away[0] and abs(angle_diff(hd[k], h1)) < settle]
            ts, te = float(tr.t[away[0]]), float(tr.t[back[0]] if back else tr.t[i_b])
        else:
            ts, te = float(tr.t[i_a]), float(tr.t[i_b])
        mid = tr.foot[(i_a + i_b) // 2]

        if total >= p["u_turn_min_deg"]:
            # a real U-turn takes time and ends in another lane (not a box that flipped under occlusion)
            h0v = np.array([np.cos(h0), np.sin(h0)])
            offset = abs(float((tr.foot[i_b] - tr.foot[i_a]) @ np.array([-h0v[1], h0v[0]])))
            width = float(np.median(tr.box[:, 2] - tr.box[:, 0]))
            if te - ts < p["u_turn_min_seconds"] or offset < p["u_turn_min_offset"] * width:
                continue
            allowed = in_any(ctx.scene.u_turn_allowed, *mid)
            if not allowed and (p["u_turn_prohibited_default"] or ctx.scene.u_turn_allowed):
                events.append(RawEvent("illegal_u_turn", ts, te, (tr.id,), min(1.0, total / 180)))
            continue

        if ctx.scene.gates and ctx.scene.allowed_movements:
            n = max(1, len(tr.t) // 8)
            g_in = next((g for g in (ctx.scene.gate_at(*f) for f in tr.foot[:n]) if g), None)
            g_out = next((g for g in (ctx.scene.gate_at(*f) for f in tr.foot[::-1][:n]) if g), None)
            if g_in and g_out and g_in != g_out and (g_in, g_out) not in ctx.scene.allowed_movements:
                if total < p["turn_min_deg"]:     # prohibited straight movement: report the whole traverse
                    ts, te = float(tr.t[0]), float(tr.t[-1])
                events.append(RawEvent("illegal_turn", ts, te, (tr.id,), 1.0, f"{g_in}->{g_out}"))
    return events


def _side_series(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Side (+1/-1) of each point relative to the polyline; 0 where the point projects outside it."""
    out = np.zeros(len(pts))
    for k, pt in enumerate(pts):
        best = None
        for a, b in zip(poly[:-1], poly[1:]):
            u = proj_param(a, b, pt)
            if -0.02 <= u <= 1.02:
                d = signed_side(a, b, pt)
                if best is None or abs(d) < abs(best):
                    best = d
        out[k] = 0 if best is None else np.sign(best)
    return out


def solid_line_crossing(ctx: Context) -> list[RawEvent]:
    p = ctx.rule("solid_line_crossing")
    events = []
    min_n = max(2, int(p["min_side_seconds"] * ctx.fps_eff))
    for tr in ctx.trajectories:
        if not tr.is_vehicle or len(tr.t) < 2 * min_n:
            continue
        half_w = moving_average((tr.box[:, 2] - tr.box[:, 0]) / 2, 5)
        left = np.stack([tr.foot[:, 0] - half_w * 0.8, tr.foot[:, 1]], 1)
        right = np.stack([tr.foot[:, 0] + half_w * 0.8, tr.foot[:, 1]], 1)
        for line in ctx.scene.solid_lines:
            sc = _side_series(line, tr.foot)
            sl, sr = _side_series(line, left), _side_series(line, right)
            k = 0
            while k < len(sc) - 1:
                if sc[k] != 0 and sc[k + 1] == -sc[k]:
                    before = sc[max(0, k - min_n + 1): k + 1]
                    after = sc[k + 1: k + 1 + min_n]
                    if len(after) == min_n and (before == sc[k]).all() and (after == -sc[k]).all():
                        orig = sc[k]
                        j0 = k
                        while j0 > 0 and (sl[j0 - 1] != orig or sr[j0 - 1] != orig) and tr.t[k] - tr.t[j0 - 1] < 3.0:
                            j0 -= 1
                        if sl[j0] == orig and sr[j0] == orig and j0 < k:
                            j0 += 1
                        j1 = k + 1
                        while j1 < len(sc) - 1 and (sl[j1] != -orig or sr[j1] != -orig) and tr.t[j1] - tr.t[k] < 3.0:
                            j1 += 1
                        events.append(RawEvent("solid_line_crossing", float(tr.t[j0]), float(tr.t[j1]), (tr.id,), 1.0))
                        k = j1
                k += 1
    return events
