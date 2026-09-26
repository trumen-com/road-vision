"""YOLO detector wrapper: one shared model per process, warmed up at import time.

Loading happens when solution.py is imported, before the harness starts the
per-video clock, so model load and CUDA warm-up never count against the budget.
"""
from __future__ import annotations

import os
import random
import time

os.environ.setdefault("YOLO_OFFLINE", "1")          # never phone home during the offline run
os.environ.setdefault("YOLO_VERBOSE", "False")

import numpy as np
import torch

from .config import resolve

_SHARED: "Detector | None" = None


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Detector:
    """Returns, per frame, an (N, 6) float32 array: x1, y1, x2, y2, conf, cls."""

    def __init__(self, params: dict):
        from ultralytics import YOLO

        cfg = params["detector"]
        self.cuda = torch.cuda.is_available() and os.environ.get("TRAFFIC_FORCE_CPU") != "1"
        self.device = 0 if self.cuda else "cpu"
        weights = os.environ.get("TRAFFIC_WEIGHTS") or cfg["weights_gpu" if self.cuda else "weights_cpu"]
        self.weights = str(resolve(weights))
        self.imgsz = int(os.environ.get("TRAFFIC_IMGSZ", cfg["imgsz_gpu" if self.cuda else "imgsz_cpu"]))
        self.conf, self.iou = cfg["conf"], cfg["iou"]
        self.batch = cfg["batch"] if self.cuda else 1
        self.classes = cfg["classes"]
        self.model = YOLO(self.weights, task="detect")
        self.sec_per_frame = self._warmup()

    def _predict(self, frames: list[np.ndarray], imgsz: int | None = None):
        return self.model.predict(frames, imgsz=imgsz or self.imgsz, conf=self.conf, iou=self.iou,
                                  classes=self.classes, device=self.device, half=self.cuda,
                                  verbose=False, agnostic_nms=True)   # one box per object (no car+truck duplicates)

    def _warmup(self) -> float:
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self._predict([dummy])
        t0 = time.perf_counter()
        n = 3
        for _ in range(n):
            self._predict([dummy] * self.batch)
        return (time.perf_counter() - t0) / (n * self.batch)

    def __call__(self, frames: list[np.ndarray], imgsz: int | None = None) -> list[np.ndarray]:
        out = []
        for i in range(0, len(frames), self.batch):
            for r in self._predict(frames[i:i + self.batch], imgsz):
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append(np.zeros((0, 6), dtype=np.float32))
                    continue
                out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                           b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
        return out


def get_detector(params: dict) -> Detector:
    global _SHARED
    if _SHARED is None:
        seed_everything(params.get("seed", 0))
        _SHARED = Detector(params)
    return _SHARED
