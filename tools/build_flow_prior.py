#!/usr/bin/env python3
"""Learn the camera's legal travel directions from all sample videos -> configs/flow_prior.npz.

At run time the prior is merged with the flow learned from the test video itself,
so one wrong-way vehicle (or a mostly-empty test clip) cannot flip a lane's direction.

    python tools/build_flow_prior.py --videos samples
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.config import CONFIG_DIR, load_params, load_scene_dict  # noqa: E402
from traffic.pipeline import align_scene, build_flow, track_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    args = ap.parse_args()
    os.environ.setdefault("TRAFFIC_CACHE", str(ROOT / ".cache"))
    params, scene = load_params(), load_scene_dict()
    prior_path = CONFIG_DIR / "flow_prior.npz"
    if prior_path.exists():
        prior_path.rename(prior_path.with_suffix(".npz.bak"))   # build from the videos only
    total = None
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4"):
        meta, stride, store, sig, _ = track_video(str(path), params, align_scene(scene, str(path)))
        ff = build_flow(store, meta, params)
        if total is None:
            total = ff
        else:
            total.merge(ff)
        print(f"{path.name}: {int(ff.count.sum())} moving samples")
    if total is None:
        print("no videos found")
        return 1
    total.save(prior_path)
    ok = ((total.count >= total.min_support) & (total.coherence() >= total.min_coherence)).sum()
    print(f"wrote {prior_path}: {ok} cells with a confident direction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
