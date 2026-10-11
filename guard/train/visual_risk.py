#!/usr/bin/env python3
"""Train Trumen Guard's visual crash-risk model on the Nexar training split and export it to ONNX for the browser.

Runs top to bottom on a Google Colab T4 (about 1.5 h, mostly video decoding) or on CPU with --tiny for a dry run.

  python train/visual_risk.py --work /content/work            # full run (Colab)
  python train/visual_risk.py --work /tmp/vr --tiny 12        # CPU dry run on 12 videos

Design (built for phones):
  * Backbone: MobileNetV3-Large (ImageNet weights, frozen) on 160x288 frames (the same 16:9 crop the detector sees),
    sampled at 4 frames per second. Output per frame: global average pool + centre-region pool = 1920 numbers.
  * Head (trained): over the last T=8 frame embeddings (2 s), per-step projection + first differences (motion),
    1-D temporal convolutions, max/mean pooling -> P(collision or near miss within ~2 s).
  * Labels: a window ending at time t is positive if the annotated event happens 0.3-2.0 s after t (Nexar test clips
    end 0.5-1.5 s before the event); negatives are windows from normal-driving videos and windows ending more than
    4 s before the event in positive videos.
  * Validation: 20% of training videos held out by video. Test: the Nexar public test split, scored at the last frame
    of each clip exactly like the official protocol; scores are written to visual_test_scores.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "akhil0790/nexar_collision_prediction"      # open mirror of nexar-ai/nexar_collision_prediction
H, W, FPS, T = 160, 288, 4, 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)


def download(work: Path, patterns, tiny: int | None):
    from huggingface_hub import hf_hub_download, snapshot_download
    data = work / "nexar"
    if tiny:
        # dry run: metadata + a few videos of each class
        for split in ("train", "test-public"):
            for lab in ("positive", "negative"):
                hf_hub_download(REPO, f"{split}/{lab}/metadata.csv", repo_type="dataset", local_dir=data)
        hf_hub_download(REPO, "solution.csv", repo_type="dataset", local_dir=data)
        for split, n in (("train", tiny), ("test-public", max(4, tiny // 2))):
            for lab in ("positive", "negative"):
                rows = list(csv.DictReader(open(data / split / lab / "metadata.csv")))[: n // 2]
                for r in rows:
                    hf_hub_download(REPO, f"{split}/{lab}/{r['file_name']}", repo_type="dataset", local_dir=data)
    else:
        snapshot_download(REPO, repo_type="dataset", local_dir=data, allow_patterns=patterns, max_workers=16)
    return data


def decode(path: Path, start: float, dur: float) -> np.ndarray:
    """Frames at FPS, resized to W x H (RGB uint8), using ffmpeg (fast seek)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(path),
           "-vf", f"fps={FPS},scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    n = len(out) // (H * W * 3)
    return np.frombuffer(out[: n * H * W * 3], np.uint8).reshape(n, H, W, 3)


class Backbone(nn.Module):
    """Frozen MobileNetV3-Large features: global pool + centre-region pool (1920-d)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features = m.features
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):                      # x: [N, 3, H, W] float in 0..1
        f = self.features((x - self.mean) / self.std)           # [N, 960, 5, 9]
        g = f.mean(dim=(2, 3))
        h, w = f.shape[2], f.shape[3]
        c = f[:, :, h // 4: h - h // 4 if h > 2 else h, w // 3: w - w // 3].mean(dim=(2, 3))   # road ahead
        return torch.cat([g, c], 1)


class Head(nn.Module):
    def __init__(self, d=1920, k=128):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, k), nn.GELU(), nn.Dropout(0.2))
        self.conv = nn.Sequential(nn.Conv1d(2 * k, k, 3, padding=1), nn.GELU(), nn.Conv1d(k, k, 3, padding=1), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * k, 1))

    def forward(self, e):                      # e: [B, T, d]
        z = self.proj(e)
        dz = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], 1)
        y = self.conv(torch.cat([z, dz], 2).transpose(1, 2))
        y = torch.cat([y.max(2).values, y.mean(2)], 1)
        return torch.sigmoid(self.out(y)).squeeze(1)


@torch.no_grad()
def embed(bb, frames: np.ndarray, dev, bs=256) -> np.ndarray:
    outs = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i: i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255
        outs.append(bb(x).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1920), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--tiny", type=int, default=None, help="dry run on N videos")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data", default=None, help="existing local Nexar folder (metadata + test-public videos)")
    ap.add_argument("--stream-train", type=int, default=None, help="download N train videos one by one, embed, delete")
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data = Path(a.data) if a.data else download(work, ["train/*", "test-public/*", "solution.csv"], a.tiny)
    print(f"data ready in {time.time() - t0:.0f}s on {dev}")

    bb = Backbone().to(dev).eval()
    if dev == "cuda":
        bb = bb.half()
        _embed = lambda fr: embed(lambda x: bb(x.half()), fr, dev)
    else:
        _embed = lambda fr: embed(bb, fr, dev)

    # ---------------- training windows: embeddings at 4 fps over the relevant range of each video
    items = []
    for lab in ("positive", "negative"):
        rows_ = list(csv.DictReader(open(data / "train" / lab / "metadata.csv")))
        if a.stream_train:
            random.Random(1).shuffle(rows_)
            rows_ = rows_[: a.stream_train // 2]
        for r in rows_:
            p = data / "train" / lab / r["file_name"]
            if p.exists() or a.stream_train:
                te = float(r["time_of_event"]) if lab == "positive" and r["time_of_event"] else None
                items.append((p, te, r))
    print(f"train videos: {len(items)}")

    def job(it):
        p, te, r = it
        fetched = False
        if a.stream_train and not p.exists():
            from huggingface_hub import hf_hub_download
            rel = str(p.relative_to(data))
            hf_hub_download(REPO, rel, repo_type="dataset", local_dir=data)
            fetched = True
        if te is not None:
            start, dur = max(0.0, te - 9.0), 9.0                 # covers early negatives and the pre-event positives
        else:
            start, dur = random.uniform(2, 22), 12.0
        fr = decode(p, start, dur)
        if fetched:
            p.unlink(missing_ok=True)
        return p, te, start, fr, r

    embs = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        for k, (p, te, start, fr, r) in enumerate(ex.map(job, items)):
            if len(fr) >= T:
                embs[p.stem] = (start, te, _embed(fr), r)
            if k % 100 == 0:
                print(f"  embedded {k}/{len(items)}  {time.time() - t0:.0f}s")

    def windows(keys):
        X, y = [], []
        for key in keys:
            start, te, E, _ = embs[key]
            for j in range(T - 1, len(E)):
                tend = start + j / FPS
                if te is None:
                    lab = 0
                elif 0.3 <= te - tend <= 2.0:
                    lab = 1
                elif te - tend > 4.0:
                    lab = 0
                else:
                    continue
                X.append(E[j - T + 1: j + 1]); y.append(lab)
        return np.array(X, np.float32), np.array(y, np.float32)

    keys = sorted(embs)
    random.shuffle(keys)
    nval = max(2, len(keys) // 5)
    Xtr, ytr = windows(keys[nval:]); Xva, yva = windows(keys[:nval])
    print(f"windows train {len(ytr)} (pos {int(ytr.sum())})  val {len(yva)} (pos {int(yva.sum())})")

    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), device=dev)
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xv = torch.from_numpy(Xva).to(dev)
    from sklearn.metrics import average_precision_score, roc_auc_score
    best, best_state = -1, None
    epochs = 3 if a.tiny else a.epochs
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(perm), 256):
            b = perm[i: i + 256]
            xb = Xt[b] + 0.05 * torch.randn_like(Xt[b])
            p = head(xb).clamp(1e-5, 1 - 1e-5)
            w = torch.where(yt[b] > 0, pw, torch.ones_like(yt[b]))
            loss = (F.binary_cross_entropy(p, yt[b], reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xv).cpu().numpy()
        ap_ = average_precision_score(yva, pv) if 0 < yva.sum() < len(yva) else float("nan")
        auc = roc_auc_score(yva, pv) if 0 < yva.sum() < len(yva) else float("nan")
        print(f"epoch {ep + 1:>2}  loss {loss.item():.3f}  val AP {ap_:.3f}  AUC {auc:.3f}")
        if not np.isnan(ap_) and ap_ > best:
            best, best_state = ap_, {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state:
        head.load_state_dict(best_state)
    head.eval()

    # ---------------- test-public: score at the last frame of each clip (official protocol)
    rows_out, test_meta = [], {}
    for lab in ("positive", "negative"):
        for r in csv.DictReader(open(data / "test-public" / lab / "metadata.csv")):
            p = data / "test-public" / lab / r["file_name"]
            if p.exists():
                test_meta[p.stem] = (p, 1 if lab == "positive" else 0, r["time_to_accident"])

    def tjob(key):
        p = test_meta[key][0]
        dur = float(sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{p}'").stdout or 0)
        return key, decode(p, max(0.0, dur - 2.6), 2.6)

    scores = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        for key, fr in ex.map(tjob, sorted(test_meta)):
            if len(fr) < 2:
                continue
            E = _embed(fr)
            if len(E) < T:
                E = np.concatenate([np.repeat(E[:1], T - len(E), 0), E])
            with torch.no_grad():
                scores[key] = float(head(torch.from_numpy(E[-T:][None]).to(dev)).item())
    with open(work / "visual_test_scores.csv", "w") as f:
        f.write("id,score\n" + "".join(f"{k},{v:.6f}\n" for k, v in sorted(scores.items())))
    ys = np.array([test_meta[k][1] for k in scores]); ss = np.array([scores[k] for k in scores])
    gs = np.array([test_meta[k][2] for k in scores])
    maps = [average_precision_score(ys[gs == g], ss[gs == g]) for g in sorted(set(gs)) if 0 < ys[gs == g].sum() < (gs == g).sum()]
    summary = {"val_ap": best, "test_public_map": float(np.mean(maps)) if maps else None, "n_test": len(scores),
               "n_train_videos": len(keys) - nval, "n_val_videos": nval, "minutes": (time.time() - t0) / 60}
    print(json.dumps(summary, indent=1))
    (work / "visual_summary.json").write_text(json.dumps(summary, indent=1))

    # ---------------- ONNX export (float32, fixed shapes, opset 17)
    bb32 = Backbone().eval()
    torch.onnx.export(bb32, torch.zeros(1, 3, H, W), work / "visual_backbone.onnx", input_names=["image"], output_names=["embedding"], opset_version=17)
    head_cpu = Head(); head_cpu.load_state_dict({k: v.cpu() for k, v in head.state_dict().items()}); head_cpu.eval()
    torch.onnx.export(head_cpu, torch.zeros(1, T, 1920), work / "visual_head.onnx", input_names=["embeddings"], output_names=["risk"], opset_version=17)
    for f in ("visual_backbone.onnx", "visual_head.onnx"):
        print(f, round(os.path.getsize(work / f) / 1e6, 2), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
