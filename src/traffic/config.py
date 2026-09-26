"""Configuration loading: params.json (thresholds) and scene.json (hand-drawn map)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

# COCO ids used throughout
PERSON = 0
BICYCLE, MOTORCYCLE = 1, 3
CAR, BUS, TRUCK = 2, 5, 7
TRAFFIC_LIGHT = 9
VEHICLES = (CAR, BUS, TRUCK, MOTORCYCLE)
TWO_WHEELERS = (BICYCLE, MOTORCYCLE)
ROAD_USERS = (PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK)
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
               9: "traffic light", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_params(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    path = Path(path or os.environ.get("TRAFFIC_PARAMS", CONFIG_DIR / "params.json"))
    params = json.loads(Path(path).read_text())
    if overrides:
        params = _deep_update(copy.deepcopy(params), overrides)
    return params


def load_scene_dict(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("TRAFFIC_SCENE", CONFIG_DIR / "scene.json"))
    return json.loads(Path(path).read_text()) if path.exists() else {}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
