#!/bin/bash
# usage: dl.sh split  -> downloads all mp4 listed in metadata
cd ${GUARD_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/guard_data}/nexar
R=akhil0790/nexar_collision_prediction
for lab in positive negative; do
  tail -n +2 $1/$lab/metadata.csv | cut -d, -f1 | while read f; do
    [ -s "$1/$lab/$f" ] && continue
    echo "$1/$lab/$f"
  done
done | xargs -P 6 -I{} sh -c 'curl -s -L -m 300 --retry 3 -o "{}.part" "https://huggingface.co/datasets/'$R'/resolve/main/{}" && mv "{}.part" "{}"'
echo DONE
