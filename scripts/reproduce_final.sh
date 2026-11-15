#!/usr/bin/env bash
# Command-line version of notebooks/pipeline_from_weights.ipynb (full run, ~10 h on one 16 GB GPU).
# Uses the released weights in release/weights; compares the result with release/submission_final.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=release/run; mkdir -p "$OUT"; ALPHA=0.75; PER=()
for k in 0 1 2 3 4; do
  W=$(python3 -c "import sys; sys.path.insert(0,'src'); from weights_io import assemble as a; print(a('release/weights','fold${k}_seed1_swa.pth')); print(a('release/weights','fold${k}_seed2.pth'))")
  W1=$(echo "$W" | sed -n 1p); W2=$(echo "$W" | sed -n 2p)
  raw=$OUT/test_raw_f$k.json
  [ -f "$raw" ] || python3 -u src/predict_submit.py --config configs/_seed2_f$k.yaml --weights "$W1" "$W2" \
      --tta --tta-views 8 --tta-merge prob --mask-bin-thr 0.5 --dump-raw "$raw" --out "$OUT/tmp_f$k.csv"
  python3 src/vote_rescore.py --preds "$raw" --alpha $ALPHA --out "$OUT/test_raw_f${k}_a$ALPHA.json"
  PER+=("$OUT/test_raw_f${k}_a$ALPHA.json")
done
python3 -u src/ensemble.py --preds "${PER[@]}" --out "$OUT/test_fused.json" --min-votes 2
python3 src/json_to_csv.py --preds "$OUT/test_fused.json" --postproc release/postproc_final.json --out "$OUT/submission.csv"
if cmp -s "$OUT/submission.csv" release/submission_final.csv; then echo "REPRODUCED: identical to the submitted file"
else echo "MISMATCH: differs from the submitted file"; exit 1; fi
