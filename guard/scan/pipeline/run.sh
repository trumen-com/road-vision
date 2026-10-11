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
python3 sfm.py          # camera poses + sparse points (COLMAP via pycolmap), ~11 min on 2 CPU cores for 95 views
python3 seg_all.py      # per-view part masks (33 classes, side-aware)
python3 depth_all.py    # per-view monocular depth (Depth Anything V2 Small)
python3 fuse.py         # align depth to the SfM points of each view, back-project car pixels (plate removed)
python3 vox.py          # multi-view fusion (>=3 views per 2 cm voxel), majority part label, car axes, metric scale
python3 parts.py        # part instances (left/right), sizes, search links, glb + parts.json + viewer payload
python3 -c "t=open('template.html').read();d=open('viewer_data.json').read();open('index.html','w').write(t.replace('__DATA__',d))"
