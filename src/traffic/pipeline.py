"""Part A: video -> detections -> tracks -> rules -> segments.

analyze_video() returns everything (tracks, signal states, raw hits, segments)
so the renderer, the website exporter and the live demo reuse one code path;
detect_events() in solution.py only keeps the segments.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import CONFIG_DIR, load_params, load_scene_dict, resolve
from .detector import get_detector
from .postprocess import to_segments
from .rules import Context, RawEvent, run_rules
from .scene import FlowField, Scene, estimate_alignment
from .signals import SignalTimeline, discover_lights
from .tracker import ByteTracker
from .tracks import TrackStore

ALL_CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
               "jaywalking", "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line",
               "congestion", "road_obstacle", "fire_smoke"]


@dataclass
class VideoMeta:
    name: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


@dataclass
class Analysis:
    meta: VideoMeta
    stride: int
    store: TrackStore
    scene: Scene
    signals: SignalTimeline
    raw: list[RawEvent]
    segments: list[list]
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"video": self.meta.name, "fps": self.meta.fps, "n_frames": self.meta.n_frames,
                "width": self.meta.width, "height": self.meta.height, "duration": round(self.meta.duration, 3),
                "stride": self.stride, "segments": self.segments, "raw": [e.to_dict() for e in self.raw],
                "signals": self.signals.to_dict(), "timings": self.timings}


def read_meta(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta = VideoMeta(Path(path).name, float(fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return meta


def align_scene(scene_dict: dict, path: str) -> dict:
    """Scene map aligned to this video: adds a runtime '_align' matrix (see scene.estimate_alignment).

    The frame used is 1 s in, and the result is deterministic for a given video.
    """
    ref_rel = scene_dict.get("reference_image")
    if not ref_rel or "_align" in scene_dict:
        return scene_dict
    ref = cv2.imread(str(resolve(ref_rel)))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    M = estimate_alignment(frame, ref, tuple(scene_dict.get("reference_size") or ref.shape[1::-1])) if ok and ref is not None else None
    if M is None:
        # not this camera (e.g. a demo upload): a hand-drawn map would be wrong here, so fall back to
        # what can be learned from the video itself (drivable area and directions from the flow field)
        return {"reference_size": scene_dict.get("reference_size"), "_align": None, "_map": "unmatched"}
    out = dict(scene_dict)
    out["_align"] = M.tolist()
    return out


def choose_stride(params: dict, fps: float, sec_per_frame: float, cuda: bool, factor: float) -> int:
    """Smallest stride whose detection cost fits the budget share.

    On GPU the answer is the configured minimum (deterministic across runs); on CPU
    it scales with measured throughput, so the demo never times out.
    """
    b = params["budget"]
    if cuda:
        return b["min_stride"]
    decode = 0.004                                   # s/frame, conservative for 1080p H.264 on CPU
    for s in range(b["min_stride"], b["max_stride"] + 1):
        if decode + sec_per_frame / s <= factor / fps:
            return s
    return b["max_stride"]


def _cache_key(path: str, params: dict, stride: int, det) -> str:
    st = os.stat(path)
    blob = json.dumps([Path(path).name, st.st_size, stride, det.weights, det.imgsz, params["detector"]], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def run_tracker(det_log: list[tuple[float, np.ndarray]], params: dict, fps_eff: float, frame_size: tuple[int, int]) -> TrackStore:
    tracker = ByteTracker(params)
    store = TrackStore()
    for t, dets in det_log:
        store.add(t, tracker.update(dets, t))
    store.finalize(params, fps_eff, frame_size)
    return store


def track_video(path: str, params: dict, scene_dict: dict, stride: int | None = None,
                progress: Callable[[float], None] | None = None) -> tuple[VideoMeta, int, TrackStore, SignalTimeline, dict]:
    """Detection + signal reading in one sequential pass, then tracking.

    With TRAFFIC_CACHE set, raw detections and signal readings are cached per
    video, so tracker / rule changes can be re-evaluated without the detector.
    """
    meta = read_meta(path)
    det = get_detector(params)
    stride = stride or choose_stride(params, meta.fps, det.sec_per_frame, det.cuda, params["budget"]["part_a_factor"])
    fps_eff = meta.fps / stride

    cache_dir = os.environ.get("TRAFFIC_CACHE")
    rois = [sg["roi"] for sg in scene_dict.get("signals", [])]        # readings depend on where, not on the ids
    key = _cache_key(path, params, stride, det) + hashlib.sha1(json.dumps([rois, scene_dict.get("_align")]).encode()).hexdigest()[:6]
    cache_file = Path(cache_dir) / f"{Path(path).stem}_{key}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        d = json.loads(cache_file.read_text())
        det_log = [(t, np.asarray(x, np.float32).reshape(-1, 6)) for t, x in d["detections"]]
        store = run_tracker(det_log, params, fps_eff, (meta.width, meta.height))
        ids = [sg["id"] for sg in scene_dict.get("signals", [])] or list(d["signals"]["rois"])
        rename = dict(zip(d["signals"]["rois"], ids))                  # cached ids -> current ids (same order)
        sig = SignalTimeline({rename[k]: np.asarray(v, np.float32) for k, v in d["signals"]["rois"].items()})
        sig.t = d["signals"]["t"]
        sig.raw = {rename[k]: v for k, v in d["signals"]["raw"].items()}
        sig.feat = {rename[k]: [tuple(x) for x in v] for k, v in d["signals"].get("feat", {}).items()}
        sig.finalize()
        return meta, stride, store, sig, {"cached": True}

    t0 = time.perf_counter()
    scene_rois = Scene.from_dict(scene_dict, meta.width, meta.height).signals
    det_log: list[tuple[float, np.ndarray]] = []
    sig = SignalTimeline(scene_rois) if scene_rois else None
    early: list[tuple[float, np.ndarray, np.ndarray]] = []     # (t, frame, dets) until lights are discovered
    cap = cv2.VideoCapture(str(path))
    batch_frames, batch_times = [], []
    idx, det_time = 0, 0.0

    def flush():
        nonlocal det_time, sig
        if not batch_frames:
            return
        td = time.perf_counter()
        dets_list = det(batch_frames)
        det_time += time.perf_counter() - td
        for frame, t, dets in zip(batch_frames, batch_times, dets_list):
            det_log.append((t, dets))
            if sig is not None:
                sig.observe(frame, t)
            else:
                early.append((t, frame, dets))
        if sig is None and len(early) >= 40:
            rois = discover_lights([d for _, _, d in early])
            sig = SignalTimeline(rois)
            for t_, f_, _ in early:
                sig.observe(f_, t_)
            early.clear()
        batch_frames.clear()
        batch_times.clear()

    while True:
        if idx % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            batch_frames.append(frame)
            batch_times.append(idx / meta.fps)
            if len(batch_frames) >= det.batch:
                flush()
        elif not cap.grab():
            break
        idx += 1
        if progress and idx % 250 == 0 and meta.n_frames:
            progress(min(0.95, idx / meta.n_frames))
    flush()
    cap.release()
    if sig is None:
        sig = SignalTimeline(discover_lights([d for _, _, d in early]))
        for t_, f_, _ in early:
            sig.observe(f_, t_)
    sig.finalize()
    store = run_tracker(det_log, params, fps_eff, (meta.width, meta.height))
    timings = {"track_sec": round(time.perf_counter() - t0, 2), "detect_sec": round(det_time, 2),
               "frames_detected": len(store.sample_times), "device": "cuda" if det.cuda else "cpu",
               "weights": Path(det.weights).name, "imgsz": det.imgsz}
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"detections": [(t, d.round(2).tolist()) for t, d in det_log],
                                          "signals": {"t": sig.t, "raw": sig.raw, "feat": sig.feat, "rois": {k: v.tolist() for k, v in sig.rois.items()}}}))
    return meta, stride, store, sig, timings


def build_flow(store: TrackStore, meta: VideoMeta, params: dict) -> FlowField:
    fp = params["flow"]
    ff = FlowField(meta.width, meta.height, tuple(fp["grid"]), fp["min_support"], fp["min_coherence"])
    for tr in store.trajectories:
        if not tr.is_vehicle:
            continue
        for i in range(len(tr.t)):
            ff.add(tr.foot[i, 0], tr.foot[i, 1], tr.vel[i, 0], tr.vel[i, 1], tr.speed[i] > fp["min_speed"])
    prior = CONFIG_DIR / "flow_prior.npz"
    if prior.exists():
        pr = FlowField.load(prior, meta.width, meta.height, min_support=fp["min_support"], min_coherence=fp["min_coherence"])
        if pr.count.shape == ff.count.shape:
            ff.merge(pr)
    return ff


def analyze_tracks(meta: VideoMeta, stride: int, store: TrackStore, sig: SignalTimeline, params: dict,
                   scene_dict: dict, classes: list[str] | None = None) -> Analysis:
    scene = Scene.from_dict(scene_dict, meta.width, meta.height)
    scene.flow = build_flow(store, meta, params)
    ctx = Context(store, scene, sig, params, meta.duration, meta.fps / stride)
    raw = run_rules(ctx)
    segments = to_segments(raw, params, meta.duration, classes or ALL_CLASSES)
    return Analysis(meta, stride, store, scene, sig, raw, segments)


def analyze_video(path: str, params: dict | None = None, scene_dict: dict | None = None,
                  stride: int | None = None, progress: Callable[[float], None] | None = None,
                  classes: list[str] | None = None) -> Analysis:
    params = params or load_params()
    scene_dict = scene_dict if scene_dict is not None else load_scene_dict()
    t0 = time.perf_counter()
    scene_dict = align_scene(scene_dict, path)
    meta, stride, store, sig, timings = track_video(path, params, scene_dict, stride, progress)
    t1 = time.perf_counter()
    an = analyze_tracks(meta, stride, store, sig, params, scene_dict, classes)
    timings.update(rules_sec=round(time.perf_counter() - t1, 2), total_sec=round(time.perf_counter() - t0, 2))
    an.timings = timings
    if progress:
        progress(1.0)
    return an
