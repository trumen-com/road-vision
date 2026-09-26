# TruMinds: traffic event detection and accident anticipation

WIUT Hackathon 2026, Computer Vision track (Toyota). Given a video from a fixed road camera, the system

* **Part A** returns every traffic event as `[start_sec, end_sec, label]` over the 14 official classes, and
* **Part B** returns, frame by frame and from past frames only, the probability that an accident starts within 5 s.

Website: **https://trumen-com.github.io/road-vision/** · Live demo: **https://trumen-com.github.io/road-vision/demo/** · Report: **https://trumen-com.github.io/road-vision/report/**

## Run it

```bash
pip install -r requirements.txt            # Python 3.10–3.13, CUDA GPU optional (falls back to CPU)
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

Or with Docker: `docker build -t truminds . && docker run --gpus all --network none -v /data/test:/data/test truminds`.

**Weights** are committed in `weights/` (YOLO11m 40 MB for the GPU run, YOLO11n 5.6 MB for CPU and the demo, YOLO11s for
development). If they are missing, `bash weights/download.sh` fetches them once and checks their SHA-256. Nothing is
downloaded at run time: `YOLO_OFFLINE=1` is set before Ultralytics is imported, and every dependency is pinned in
`requirements.txt`.

**Runtime.** Importing `solution.py` loads and warms up the detector, before the harness starts timing the first video.
On a T4 each part runs at about real time or faster (YOLO11m, 960 px, FP16, every 2nd frame; Part A batches 8 frames),
well inside the 3× budget. Without CUDA, the stride grows from the measured throughput, so a video never runs over budget.

`run_submission.py` and `evaluate.py` are the organisers' files, unchanged.

**Official sample run on a T4.** `notebooks/run_on_colab_t4.ipynb` clones this repository on a Colab T4 (the
organisers' GPU class), downloads the four sample videos from the organisers' links, runs the harness offline and
checks every video against the 3× time budget. `predictions_samples.json` in this repository is its output.

## The camera and the scene map

The samples are 4K (3840×2160, 29.97 fps) views of a signalised Tashkent intersection. The scene map
(`configs/scene.json`, drawn on `configs/scene_ref.jpg`) holds the carriageway, three zebra crossings, refuge
islands, the median, curb parking and the bus stop (excluded areas), the stop line of the approach facing the
camera and its signal head. The camera shifts slightly between recordings (up to ~60 px, ~1°), so every video
is first registered to the reference frame (SIFT features + RANSAC similarity transform) and the map is moved
with it. Footage that does not match the reference (e.g. a demo upload from another camera) runs without the
hand-drawn map, using only what is learned from the video (drivable area and lane directions).

## Repository layout

```
solution.py              official interface (detect_events, RiskEstimator), wires into src/traffic
run_submission.py        organisers' harness (unchanged)
evaluate.py              organisers' metric (unchanged)
configs/params.json      every threshold, stride, model choice and post-processing offset
configs/scene.json       hand-drawn scene map of the camera (tools/scene_editor.html)
src/traffic/
  detector.py            YOLO wrapper: shared model, warm-up at import, FP16 on GPU, seeds
  tracker.py             ByteTrack-style tracker (own implementation, deterministic)
  tracks.py              trajectory store: stitching, smoothing, speed in sizes/s, headings
  scene.py               scene map + flow field learned from trajectories
  signals.py             traffic-light state from signal-head ROIs (auto-discovered if not drawn)
  conflict.py            time-to-collision between two road users
  rules/                 one rule per event family (occupancy, direction, pedestrians, signals, collisions)
  postprocess.py         union, gap merge, blip filter, tuned boundary offsets
  pipeline.py            Part A end to end (+ detection cache for fast tuning)
  risk.py                Part B causal risk estimator
  render.py              annotated playback with timeline and risk bar
tools/
  labeler.html           browser tool to annotate the sample videos (exports ground-truth JSON)
  scene_editor.html      browser tool to draw the scene map (exports configs/scene.json)
  tune.py                grid-search per-class post-processing against our labels
  export_site.py         EDA figures, annotated videos and JSON for the website
tests/                   synthetic-trajectory tests for every rule (python -m pytest tests)
demo/                    FastAPI live-demo backend + Dockerfile (CPU)
web/                     Next.js website (static export, deployed by .github/workflows/pages.yml)
predictions_samples.json our output on the sample videos (python run_submission.py --videos samples)
```

## Approach

```
camera → YOLO11 (every 2nd frame) ─┬─ Part A: tracker + stitching → path diary ─┐
                                   │    scene map + learned flow + light reader ─┴→ 13 rules → post-processing → events
                                   └─ Part B: own online tracker → time-to-collision, hard braking, density → risk per frame
```

**What is learned and what is rule-based.**

| Component | Learned or rule-based |
|---|---|
| Object detection (road users, traffic lights, animals) | Learned: YOLO11 pre-trained on COCO, not fine-tuned |
| Tracking | Algorithmic (Kalman filter + two-stage IoU association, after ByteTrack) |
| Legal travel direction per image cell | Learned from the video's own trajectories (flow field), unless lanes are drawn |
| Traffic-light state | Unsupervised: red-minus-green colour of the brightest lamp pixels, split into two states by 1-D 2-means over the video (daylight washes lamps out, so fixed thresholds fail) |
| Scene alignment | Geometric: SIFT + RANSAC similarity from each video to the reference frame |
| All 13 event decisions | Rule-based on trajectories and the scene map; thresholds in `configs/params.json` |
| Boundary offsets per class | Tuned by grid search on our own labels of the sample videos (`tools/tune.py`) |
| Accident risk (Part B) | Rule-based physics (time-to-collision) shaped for the metric |

Key design choices:

* **Speed in object sizes per second** (pixel speed ÷ √box area). Comparable from the bottom of the frame to the horizon,
  without calibration, so one threshold works everywhere.
* **Contact on ground footprints**, normalised by the pair's size. Boxes that only overlap because of perspective do not count.
* **Accident = contact + impact signature**: an abrupt speed collapse (within ~0.8 s, far faster than braking), or a
  stationary vehicle being shoved. Both then come to rest while nearby traffic keeps moving; stop-and-go queues are rejected.
* **Guards against tracking artefacts**: boxes cut by the frame border are dropped; speed drops only count when the box
  area is stable (not an occlusion); duplicate boxes are suppressed; objects under 20–25 px are ignored by motion rules.
* **Part B is causal by construction.** `RiskEstimator` runs its own detector calls and its own online tracker on the frames
  it receives. It never opens the video and never reads Part A output. Scores are graded for ranking (AP); only a genuine
  collision course (TTC below ~1.1 s) crosses the 0.5 alarm threshold, and a 1.2 s half-life keeps each danger episode as one alarm.
* **`fire_smoke` is not predicted.** We have no detector we trust for it, and a predicted class that is absent from the
  test set adds a zero to the macro F1.

## Data and licences

| Item | Use | Licence |
|---|---|---|
| Sample videos from the organisers | EDA, our dev labels, threshold / offset tuning | Hackathon terms |
| COCO (via YOLO11 pre-training) | Detector weights | CC BY 4.0 (annotations) |
| Ultralytics YOLO11 + library | Detector | AGPL-3.0 |
| ByteTrack (method only, re-implemented) | Tracking | MIT (original code) |
| Public clips used only for development testing (Roboflow Supervision examples, Intel IoT sample videos) | Smoke tests of the pipeline; not used for training or tuning | See their repositories |

No footage from the competition camera other than the provided samples is used.

## Reproducibility

* Seeds fixed (`seed` in `configs/params.json`: Python, NumPy, PyTorch); cuDNN deterministic, benchmark off.
* On GPU, the frame stride is fixed by config (no timing-dependent choices). The tracker has no randomness.
* Floating-point differences between GPU models can move a box by a fraction of a pixel. This is the only source of
  non-determinism we know of.
* `predictions_samples.json` is produced by `python run_submission.py --videos samples --out predictions_samples.json --team TruMinds`.
* Dev labels of the samples: `data/dev_labels.json`; score them with `python evaluate.py --pred predictions_samples.json --gt data/dev_labels.json --per-video`.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q                                   # rule tests on synthetic trajectories
TRAFFIC_CACHE=.cache python tools/tune.py --videos samples --gt data/dev_labels.json   # add --write to save offsets
python tools/export_site.py --videos samples --out web/public                        # website data + annotated videos
cd web && npm install && npm run build                                              # static site in web/out
uvicorn demo.app:app --port 7860                                                    # live-demo backend
```

Environment overrides for experiments: `TRAFFIC_WEIGHTS`, `TRAFFIC_IMGSZ`, `TRAFFIC_FORCE_CPU=1`, `TRAFFIC_PARAMS`,
`TRAFFIC_SCENE`, `TRAFFIC_CACHE`.

## Team

| Member | Role | Contributions |
|---|---|---|
| Asilbek Shodmonov | Website, demo, tooling | Website and live demo, scene-map and labelling tools, deployment |
| Komron Akmalov | TODO | TODO |
| TODO | TODO | TODO |
