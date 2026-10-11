#!/bin/bash
# Trumen Scan: phone walk-around -> 3D exterior split into named parts -> glb + parts.json + viewer.
#   put frames and photos in imgs/ (e.g. ffmpeg -i walk.mov -vf "fps=3,scale=1280:-2" imgs/f_%03d.jpg), then:
#   bash run.sh
# Needs: pip install pycolmap ultralytics "transformers==4.46.3" torch trimesh scipy pillow huggingface_hub
set -euo pipefail
python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('mitbersh/car-parts-segmentation', 'parts_segmentation.pt', local_dir='seg')
PY
python3 sfm2.py         # camera poses + sparse points (COLMAP via pycolmap); video frames share one camera; ~20 min on 2 CPU cores
                        # (expects imgs2/video/*.jpg and imgs2/photos/*.jpg; per-view masks and depth below read imgs/)
python3 seg_all.py      # per-view part masks (33 classes, side-aware)
python3 depth_all.py    # per-view monocular depth (Depth Anything V2 Small)
# tuned fusion: per-view depth aligned with a tilt term, neighbour-view depth check, voxel fusion, ground plane and
# symmetry axes, metric scale from the published length; prints the accuracy checks and writes car_parts.npz
SFM=sfm2 python3 tune.py '{"mv_min":2,"mv_tau":0.015,"erode":10,"align":"inv_plane","fill":1,"fill_views":3}' --save | tail -1 > tune_final.json
python3 parts.py        # part instances (left/right), sizes, search links, glb + parts.json + viewer payload
python3 -c "t=open('template.html').read();d=open('viewer_data.json').read();open('index.html','w').write(t.replace('__DATA__',d))"
