#!/usr/bin/env python3
"""Visual-risk timeline for every clip (4 fps, causal: each value uses only the last 8 frames), with the exported
ONNX models, so the engine replay can fuse it exactly as the browser does.

  python eval/visual_timelines.py --models app/public/models --list ../guard_data/test_public.txt --out results/visual_test_tl.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

H, W, FPS, T = 160, 288, 4, 8


def decode(path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS},scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    n = len(out) // (H * W * 3)
    return np.frombuffer(out[: n * H * W * 3], np.uint8).reshape(n, H, W, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    bb = ort.InferenceSession(str(Path(a.models) / "visual_backbone.onnx"), so)
    hd = ort.InferenceSession(str(Path(a.models) / "visual_head.onnx"), so)
    out = {}
    paths = [l.strip().split(",")[0] for l in open(a.list) if l.strip()]
    for k, p in enumerate(paths):
        fr = decode(p)
        if len(fr) == 0:
            continue
        x = fr.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        E = np.concatenate([bb.run(None, {"image": x[i: i + 1]})[0] for i in range(len(x))])
        tl = []
        for j in range(len(E)):
            win = E[max(0, j - T + 1): j + 1]
            if len(win) < T:
                win = np.concatenate([np.repeat(win[:1], T - len(win), 0), win])
            r = float(hd.run(None, {"embeddings": win[None].astype(np.float32)})[0].ravel()[0])
            tl.append([round(j / FPS, 3), round(r, 5)])
        out[Path(p).stem] = tl
        if k % 50 == 0:
            print(f"{k}/{len(paths)}", flush=True)
    Path(a.out).write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
