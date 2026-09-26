#!/usr/bin/env python3
"""Assemble a Hugging Face Docker Space for the live demo -> hf_space/ (push that folder to the Space).

    python tools/make_hf_space.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_space"

README = """---
title: Trumen Road Vision Demo
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

Live-demo backend for the Trumen traffic-event detector (WIUT Hackathon 2026, CV track).
POST an .mp4 (≤ 2 min, ≤ 100 MB) to `/api/jobs`, poll `/api/jobs/{id}`; the website renders the result.
Source: https://github.com/trumen-com/road-vision
"""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for sub in ("src", "configs", "demo"):
        dst = OUT / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ROOT / sub, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (OUT / "weights").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "weights" / "yolo11n.pt", OUT / "weights" / "yolo11n.pt")
    shutil.copy2(ROOT / "demo" / "Dockerfile", OUT / "Dockerfile")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
