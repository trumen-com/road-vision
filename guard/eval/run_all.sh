#!/bin/bash
# Reproduce everything shown on the "Test results" page, from cached detections.
#   GUARD_DATA=/path/to/guard_data bash eval/run_all.sh
# Expects: $GUARD_DATA/nexar (metadata + solution.csv), $GUARD_DATA/dets_test, $GUARD_DATA/dets_train
# (create them with eval/download_nexar.sh, eval/extract_dets.ts and eval/train_pipeline.sh, see README).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${GUARD_DATA:-../guard_data}
mkdir -p results app/public/results app/public/models

# 1. learned risk model, fitted on the TRAINING split only
npx tsx eval/score.ts --dets "$D/dets_train" --out results/train_scores.csv --features results/train_agg.jsonl --agg 1 --no-model 1
(cd eval && python3 train_model.py --features ../results/train_agg.jsonl --data "$(cd .. && cd "$D" && pwd)/nexar" --split train --out ../app/public/models/risk_model.json)

# 2. shipping alert threshold, chosen on the TRAINING split (target false-warning rate)
npx tsx eval/score.ts --dets "$D/dets_train" --model app/public/models/risk_model.json --out results/train_scored.csv --timeline results/train_timeline.json
python3 eval/choose_threshold.py --timeline results/train_timeline.json --data "$D/nexar" --out results/shipping.json

# 3. test split: hand-set version, naive baseline, final system
npx tsx eval/score.ts --dets "$D/dets_test" --no-model 1 --out results/test_heuristic.csv --timeline results/test_heuristic_tl.json
npx tsx eval/score.ts --dets "$D/dets_test" --baseline 1 --out results/test_baseline.csv
python3 eval/report.py --scores results/test_heuristic.csv --timeline results/test_heuristic_tl.json --data "$D/nexar" --out results/heuristic.json --label hand-set
npx tsx eval/score.ts --dets "$D/dets_test" --model app/public/models/risk_model.json --params "$(python3 -c 'import json;print(json.dumps({"alert":json.load(open("results/shipping.json"))["alert"]}))')" \
  --out results/test_scores.csv --timeline results/test_timeline.json
python3 eval/report.py --scores results/test_scores.csv --timeline results/test_timeline.json --data "$D/nexar" \
  --baseline results/test_baseline.csv --heuristic results/heuristic.json --shipping results/shipping.json \
  --out app/public/results/results.json
echo "results written to app/public/results/results.json"
