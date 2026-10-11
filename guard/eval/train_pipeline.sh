#!/bin/bash
# Download Nexar train videos in batches, extract detections on a window, delete the video.
cd ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/nexar
R=akhil0790/nexar_collision_prediction
N=${1:-150}
python3 - $N > ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/train_windows.txt <<'PY'
import csv, random, sys
random.seed(7); n=int(sys.argv[1])
pos=list(csv.DictReader(open("train/positive/metadata.csv"))); neg=list(csv.DictReader(open("train/negative/metadata.csv")))
random.shuffle(pos); random.shuffle(neg)
for a,b in zip(pos[:n],neg[:n]):
    te=float(a["time_of_event"]); print(f"train/positive/{a['file_name']},{max(0,te-7):.3f},7.3")
    print(f"train/negative/{b['file_name']},10,12")
PY
mkdir -p ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/dets_train
while read line; do
  f=${line%%,*}; key=$(basename $f .mp4)
  [ -s ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/dets_train/$key.json ] && continue
  mkdir -p $(dirname $f)
  curl -s -L -m 300 --retry 3 -o $f "https://huggingface.co/datasets/$R/resolve/main/$f" || continue
  echo "${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/nexar/$line" > /tmp/one_$$.txt
  (cd $(cd "$(dirname "$0")/.." && pwd) && npx tsx eval/extract_dets.ts --list /tmp/one_$$.txt --out ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/dets_train >/dev/null 2>&1)
  rm -f $f
done < ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/train_windows.txt
echo TRAIN_DONE
