"""Rule checker on synthetic trajectories (no video, no detector needed).

Each scenario builds trajectories by hand, runs the real rules and post-processing,
and checks the class and the boundaries. Run: python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic.config import load_params  # noqa: E402
from traffic.pipeline import VideoMeta, build_flow  # noqa: E402
from traffic.postprocess import to_segments  # noqa: E402
from traffic.rules import Context, run_rules  # noqa: E402
from traffic.scene import Scene  # noqa: E402
from traffic.signals import SignalTimeline  # noqa: E402
from traffic.tracks import TrackStore  # noqa: E402

FPS_EFF = 12.5
W, H = 1920, 1080
ROAD = [[0, 300], [1920, 300], [1920, 1080], [0, 1080]]


def track(tid, cls, t, x, y, w=80, h=60):
    x, y = np.broadcast_to(x, t.shape).astype(float), np.broadcast_to(y, t.shape).astype(float)
    box = np.stack([x - w / 2, y - h, x + w / 2, y], axis=1)
    return {"id": tid, "cls": cls, "t": t.tolist(), "box": box.tolist(), "conf": [0.9] * len(t)}


def background_traffic(duration=40.0, n=30, x=500, down=True, start_id=1000):
    """Cars flowing through a lane at a steady 300 px/s so the flow field learns its direction."""
    out = []
    for k in range(n):
        t0 = k * duration / n
        t = np.arange(t0, min(t0 + 2.6, duration), 1 / FPS_EFF)
        y = 320 + 300 * (t - t0) if down else 1060 - 300 * (t - t0)
        out.append(track(start_id + k, 2, t, x, y))
    return out


def run(tracks, duration=40.0, scene=None, params=None):
    params = params or load_params()
    scene_dict = scene or {"reference_size": [W, H], "carriageway": [ROAD]}
    store = TrackStore.from_dict({"sample_times": np.arange(0, duration, 1 / FPS_EFF).tolist(), "tracks": tracks}, params, FPS_EFF)
    meta = VideoMeta("synthetic.mp4", 25.0, int(duration * 25), W, H)
    sc = Scene.from_dict(scene_dict, W, H)
    sc.flow = build_flow(store, meta, params)
    sig = SignalTimeline({})
    sig.finalize()
    ctx = Context(store, sc, sig, params, duration, FPS_EFF)
    raw = run_rules(ctx)
    return raw, to_segments(raw, params, duration, sorted({e.label for e in raw}))


def labels(segs):
    return {s[2] for s in segs}


def test_stopped_vehicle_boundaries():
    t = np.arange(0, 40, 1 / FPS_EFF)
    y = np.where(t < 10, 320 + 30 * t, 620.0)         # drives, stops at t=10 in the lane
    y = np.where(t > 25, 620 + 30 * (t - 25), y)       # moves again at t=25
    segs = run([track(1, 2, t, 1200, y)] + background_traffic(x=500))[1]
    sv = [s for s in segs if s[2] == "stopped_vehicle"]
    assert len(sv) == 1
    assert abs(sv[0][0] - 10) < 1.5 and abs(sv[0][1] - 25) < 1.5


def test_queue_is_not_stopped_vehicle():
    t = np.arange(0, 30, 1 / FPS_EFF)
    cars = [track(i, 2, t, 1200, 700 - 90 * i) for i in range(1, 5)]   # four cars queued, one behind another
    for c in cars:
        c["t"] = t.tolist()
    # queue direction is learned from traffic that flows through the same lane earlier/later
    segs = run(cars + background_traffic(duration=30, x=1200, n=10, start_id=500), duration=30)[1]
    assert sum(s[2] == "stopped_vehicle" for s in segs) <= 1      # only the head of the queue may qualify


def test_wrong_way():
    t = np.arange(12, 16, 1 / FPS_EFF)
    wrong = track(1, 2, t, 500, 1060 - 180 * (t - 12))              # drives up a lane that flows down
    segs = run([wrong] + background_traffic(x=500))[1]
    ww = [s for s in segs if s[2] == "wrong_way"]
    assert len(ww) == 1 and ww[0][0] < 13.5 and ww[0][1] > 14.5


def test_jaywalking_and_crosswalk():
    t = np.arange(5, 12, 1 / FPS_EFF)
    walker = track(1, 0, t, 300 + 120 * (t - 5), 700, w=30, h=90)
    scene = {"reference_size": [W, H], "carriageway": [ROAD]}
    segs = run([walker], scene=scene)[1]
    assert "jaywalking" in labels(segs)
    scene["crosswalks"] = [[[200, 600], [1300, 600], [1300, 800], [200, 800]]]
    segs = run([walker], scene=scene)[1]
    assert "jaywalking" not in labels(segs)


def test_u_turn():
    t = np.arange(5, 15, 1 / FPS_EFF)
    ang = np.clip((t - 8) / 3, 0, 1) * np.pi               # heading rotates 180 degrees between t=8 and t=11
    vx, vy = 150 * np.sin(ang), 150 * np.cos(ang)
    x = 800 + np.cumsum(vx) / FPS_EFF
    y = 400 + np.cumsum(vy) / FPS_EFF
    segs = run([track(1, 2, t, x, y)], duration=20)[1]
    ut = [s for s in segs if s[2] == "illegal_u_turn"]
    assert len(ut) == 1 and 7 <= ut[0][0] <= 9.5 and 10 <= ut[0][1] <= 12.5


def test_crash_is_accident_not_near_miss():
    t = np.arange(0, 20, 1 / FPS_EFF)
    xa = np.where(t < 10, -200 + 120 * t, 1000.0)          # A drives right at 120 px/s ... stops dead at t=10
    xb = np.full_like(t, 1075.0)                           # B waits; A hits it at t=10
    xb = np.where((t > 10) & (t < 10.6), 1075 + 120 * (t - 10), xb)
    xb = np.where(t >= 10.6, 1147.0, xb)                   # B is shoved forward by the impact
    segs = run([track(1, 2, t, xa, 700), track(2, 2, t, xb, 700)], duration=20)[1]
    acc = [s for s in segs if s[2] == "accident"]
    assert len(acc) == 1 and 9 <= acc[0][0] <= 10.5


def test_calm_traffic_has_no_events():
    segs = run(background_traffic(x=500) + background_traffic(x=1400, down=False, start_id=2000))[1]
    assert segs == []


def test_segments_are_valid():
    t = np.arange(0, 40, 1 / FPS_EFF)
    y = np.where(t < 10, 320 + 30 * t, 620.0)
    segs = run([track(1, 2, t, 1200, y)] + background_traffic(x=500))[1]
    for s, e, lab in segs:
        assert 0 <= s < e <= 40
    by = {}
    for s, e, lab in sorted(segs):
        assert s >= by.get(lab, -1)
        by[lab] = e


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
