#!/usr/bin/env bash
# Weights are committed to the repository (weights/*.pt, 65 MB total). This script only
# re-fetches them if they are missing, e.g. after a shallow clone without the files.
# Run once, with internet, before the offline evaluation.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.3.0
for m in yolo11m yolo11s yolo11n; do
  if [ ! -s "$m.pt" ]; then
    echo "downloading $m.pt"
    curl -fL --retry 3 -o "$m.pt" "$BASE/$m.pt"
  fi
done
sha256sum -c SHA256SUMS
