"""Part B: causal accident-risk estimator ("danger forecaster").

Runs its own detector pass and its own online tracker on the frames it is
given, in order - it never opens the video and never touches Part A output.

Score design follows the metric:
  * AP only cares about ranking, so the score is graded everywhere: a small
    traffic-density term and soft conflict terms keep ordinary frames ordered.
  * Alarms fire at >= 0.5 and every false alarm hurts, so only a genuine
    collision course (short time-to-collision between converging road users)
    can cross 0.5 on its own. Hard braking or wrong-way driving adds to it.
  * The score decays with a half-life instead of dropping instantly, so one
    danger episode stays one alarm (runs < 2 s apart are merged anyway).
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .config import ROAD_USERS, VEHICLES, load_params
from .conflict import ellipse_distance, time_to_collision
from .detector import get_detector
from .tracker import ByteTracker


class _State:
    __slots__ = ("hist", "cls", "last_t")

    def __init__(self):
        self.hist: deque = deque(maxlen=24)   # (t, foot_x, foot_y, scale, box)
        self.cls = -1
        self.last_t = 0.0


class CausalRisk:
    def __init__(self, params: dict | None = None):
        self.params = params or load_params()
        self.p = self.params["risk"]
        self.det = get_detector(self.params)
        self.imgsz = self.p["imgsz_gpu" if self.det.cuda else "imgsz_cpu"]
        self.tracker = ByteTracker(self.params)

    def reset(self, meta: dict) -> None:
        self.meta = meta
        fps = float(meta.get("fps") or 25.0)
        self.fps = fps
        per_frame = self.det.sec_per_frame
        budget = self.params["budget"]
        # causal stride: every 2nd frame on GPU; on CPU as sparse as needed to stay inside the budget
        self.stride = budget["min_stride"]
        while self.stride < budget["max_stride"] + 2 and per_frame / self.stride > budget["part_b_factor"] / fps * 0.8:
            self.stride += 1
        self.tracker.reset()
        self.states: dict[int, _State] = {}
        self.frame_idx = -1
        self.score = 0.0
        self.prev_r_ttc = 0.0
        self.last_t = 0.0
        self.components: dict = {}
        self.t_start = time.perf_counter()

    # ------------------------------------------------------------------
    def _kf_velocity(self, tid: int, st: _State) -> tuple[np.ndarray, float]:
        """Foot-point velocity (px/s) from the tracker's Kalman state: smoothed, and robust to one-off box jumps."""
        tr = self._by_id.get(tid)
        if tr is None:
            return np.zeros(2), 0.0
        vx, vy, _, vh = tr.kf.x[4:8] / tr.kf.STEP
        v = np.array([vx, vy + vh / 2.0])
        return v, float(np.hypot(*v) / max(st.hist[-1][3], 4.0))

    def _brake(self, st: _State) -> float:
        """Speed ratio lost in the last ~1 s (0 = none, 1 = full stop from speed)."""
        h = list(st.hist)
        if len(h) < 6:
            return 0.0
        def spd(a, b):
            dt = b[0] - a[0]
            return math.hypot(b[1] - a[1], b[2] - a[2]) / max(b[3], 4.0) / dt if dt > 0 else 0.0
        mid = len(h) // 2
        v_old = spd(h[0], h[mid])
        v_new = spd(h[mid], h[-1])
        if v_old < 1.0:
            return 0.0
        return max(0.0, (v_old - v_new) / v_old)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.frame_idx += 1
        dt_since = t_sec - self.last_t
        decay = 0.5 ** (max(dt_since, 0.0) / self.p["half_life"])
        if self.frame_idx % self.stride:
            return float(self.score * decay if dt_since > 0 else self.score)
        self.last_t = t_sec

        dets = self.det([frame], imgsz=self.imgsz)[0]
        outs = self.tracker.update(dets, t_sec)
        self._by_id = {tr.id: tr for tr in self.tracker.tracks}
        seen = set()
        for tid, box, cls, conf in outs:
            if cls not in ROAD_USERS:
                continue
            st = self.states.setdefault(tid, _State())
            st.cls, st.last_t = cls, t_sec
            st.hist.append((t_sec, (box[0] + box[2]) / 2, box[3], math.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1.0)), box))
            seen.add(tid)
        for tid in [k for k, s in self.states.items() if t_sec - s.last_t > 2.0]:
            del self.states[tid]

        live = []
        for tid in seen:
            st = self.states[tid]
            if len(st.hist) < 3 or t_sec - st.hist[0][0] < self.p["min_age_seconds"]:
                continue        # young tracks have unconverged velocities
            b = st.hist[-1][4]
            if b[0] < 3 or b[1] < 3 or b[2] > self.meta["width"] - 3 or b[3] > self.meta["height"] - 3:
                continue        # cut by the frame border: its motion is not trustworthy
            v, spd = self._kf_velocity(tid, st)
            live.append((tid, st, st.hist[-1][4], v, spd))

        # 1) most dangerous conflict: short time-to-collision AND a high closing speed.
        # Followers in a lane sit close with similar speeds (small TTC by geometry, small closing
        # speed); a crash needs one party approaching the other much faster than traffic does.
        r_ttc = r_soft = 0.0
        n_moving = 0
        min_scale = self.p["min_scale_px"] * float(self.meta.get("height") or 1080) / 1080.0
        for i, (ta, sa, ba, va, spa) in enumerate(live):
            n_moving += spa > 0.5
            if sa.hist[-1][3] < min_scale:
                continue
            for tb, sb, bb, vb, spb in live[i + 1:]:
                if sb.hist[-1][3] < min_scale or max(spa, spb) < 0.5 or (sa.cls not in VEHICLES and sb.cls not in VEHICLES):
                    continue
                if ellipse_distance(ba, bb) > self.p["max_gap"]:
                    continue        # far apart now: image-plane convergence at a distance is mostly depth/overpass artefacts
                ttc = time_to_collision(ba, va, bb, vb)
                if ttc == 0.0 or not math.isfinite(ttc):   # already overlapping in the image, or diverging
                    continue
                closing = float(np.linalg.norm(vb - va)) / max(min(sa.hist[-1][3], sb.hist[-1][3]), 4.0)
                if max(spa, spb) < self.p["striker_min_speed"]:
                    continue        # creeping queues close gaps slowly; a crash needs a genuinely moving party
                conf = min(1.0, max(0.0, (closing - self.p["closing_min"]) / self.p["closing_span"]))
                if conf <= 0:
                    continue
                r = conf / (1.0 + math.exp((ttc - self.p["ttc_mid"]) / self.p["ttc_width"]))
                r_ttc = max(r_ttc, r)
                r_soft = max(r_soft, 0.3 * conf * math.exp(-ttc / 2.0))     # soft tail keeps the AP ranking informative

        # 2) hard braking anywhere
        brake = max((self._brake(st) for _, st, _, _, _ in live if st.cls in VEHICLES), default=0.0)
        r_brake = self.p["brake_weight"] * max(0.0, (brake - 0.3) / 0.7)

        # 3) density base term (ranking only)
        r_base = self.p["base_weight"] * min(1.0, n_moving / 12.0)

        # a collision course must persist over consecutive processed frames; one-step spikes are box noise
        r_ttc, self.prev_r_ttc = min(r_ttc, self.prev_r_ttc), r_ttc
        combined = 1.0 - (1 - max(r_ttc, r_soft)) * (1 - r_brake) * (1 - r_base)
        self.components = {"r_ttc": r_ttc, "r_soft": r_soft, "brake": r_brake, "base": r_base}
        self.score = float(np.clip(max(combined, self.score * decay), 0.0, 1.0))
        return self.score
