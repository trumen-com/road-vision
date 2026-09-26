"""Live-demo backend: upload an .mp4, get events, an annotated video and the risk curve.

    uvicorn demo.app:app --host 0.0.0.0 --port 7860

Runs the same pipeline as the submission (src/traffic) on CPU with the small
detector. Jobs run one at a time in a background worker; the website polls
/api/jobs/{id} for progress.
"""
from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("TRAFFIC_FORCE_CPU", "1")

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.config import CLASS_NAMES, load_params, load_scene_dict  # noqa: E402
from traffic.detector import get_detector  # noqa: E402
from traffic.pipeline import analyze_video, read_meta  # noqa: E402
from traffic.render import render_video  # noqa: E402
from traffic.risk import CausalRisk  # noqa: E402

MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 125.0
JOB_TTL = 3600
WORK = Path(os.environ.get("DEMO_WORKDIR", "/tmp/truminds_jobs"))
WORK.mkdir(parents=True, exist_ok=True)

# the demo has no 3x budget: spend more CPU time for denser sampling (better tracks)
PARAMS = load_params(overrides={"budget": {"part_a_factor": 4.0, "part_b_factor": 2.5}})
SCENE = load_scene_dict()
get_detector(PARAMS)            # load once at start-up

app = FastAPI(title="TruMinds traffic demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()


def _public(job: dict) -> dict:
    out = {k: job[k] for k in ("id", "status", "progress", "stage", "error") if k in job}
    if job.get("status") == "queued":
        waiting = [j for j in list(QUEUE.queue)]
        out["stage"] = f"Queued (position {waiting.index(job['id']) + 1 if job['id'] in waiting else 1} in line)"
    if job.get("status") == "done":
        out["result"] = job["result"]
        out["video_url"] = f"/api/jobs/{job['id']}/video"
    return out


def _risk_curve(path: Path, meta, job: dict) -> list[list[float]]:
    est = CausalRisk(PARAMS)
    est.reset({"video_id": meta.name, "fps": meta.fps, "width": meta.width, "height": meta.height, "n_frames": meta.n_frames})
    cap = cv2.VideoCapture(str(path))
    curve, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / meta.fps
        curve.append([round(t, 3), round(float(np.clip(est.step(frame, t), 0, 1)), 4)])
        i += 1
        if i % 50 == 0 and meta.n_frames:
            job["progress"] = 0.55 + 0.2 * i / meta.n_frames
    cap.release()
    return curve


def _run(job_id: str) -> None:
    job = JOBS[job_id]
    path = Path(job["dir"]) / "input.mp4"
    try:
        job.update(status="running", stage="Detecting and tracking road users", progress=0.02)
        an = analyze_video(str(path), PARAMS, SCENE, progress=lambda f: job.update(progress=0.02 + 0.5 * f))
        job.update(stage="Computing the causal accident risk", progress=0.55)
        risk = _risk_curve(path, an.meta, job)
        job.update(stage="Rendering the annotated video", progress=0.76)
        out = Path(job["dir"]) / "annotated.mp4"
        render_video(str(path), an, str(out), risk, max_width=960, progress=lambda f: job.update(progress=0.76 + 0.22 * f))
        doc = an.to_dict()
        doc["risk"] = risk[:: max(1, int(round(an.meta.fps / 5)))]
        doc["n_tracks"] = {CLASS_NAMES.get(c, str(c)): sum(tr.cls == c for tr in an.store.trajectories)
                           for c in sorted({tr.cls for tr in an.store.trajectories})}
        job.update(status="done", stage="Done", progress=1.0, result=doc)
    except Exception as err:  # reported to the page, never crashes the server
        traceback.print_exc()
        job.update(status="error", error=f"{type(err).__name__}: {err}", progress=1.0)


def _worker() -> None:
    while True:
        job_id = QUEUE.get()
        if job_id in JOBS:
            _run(job_id)
        _cleanup()


def _cleanup() -> None:
    now = time.time()
    for jid, job in list(JOBS.items()):
        if now - job["created"] > JOB_TTL and job["status"] in ("done", "error"):
            shutil.rmtree(job["dir"], ignore_errors=True)
            JOBS.pop(jid, None)


threading.Thread(target=_worker, daemon=True).start()


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "jobs": len(JOBS), "queued": QUEUE.qsize()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(400, "Please upload an .mp4 file.")
    job_id = uuid.uuid4().hex[:12]
    d = WORK / job_id
    d.mkdir(parents=True)
    dst = d / "input.mp4"
    size = 0
    with dst.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "File is larger than 100 MB.")
            f.write(chunk)
    try:
        meta = read_meta(str(dst))
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Could not read this video. Is it a valid H.264 .mp4?")
    if meta.duration > MAX_SECONDS or meta.n_frames <= 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Video is {meta.duration:.0f} s long; the demo accepts up to 120 s.")
    JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "stage": "Queued", "dir": str(d), "created": time.time()}
    QUEUE.put(job_id)
    return _public(JOBS[job_id])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job (results are kept for one hour).")
    return _public(JOBS[job_id])


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    job = JOBS.get(job_id)
    path = Path(job["dir"]) / "annotated.mp4" if job else None
    if not path or not path.exists():
        raise HTTPException(404, "No video for this job.")
    return FileResponse(path, media_type="video/mp4")
