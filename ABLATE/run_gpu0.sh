#!/usr/bin/env bash
# GPU 0 (RTX 4060, Ada, native bf16 but fp16 used for parity with the 2070):
#   5G half+ch64  (~42h)  ->  5I half+ch16  (~9h)
# Screening on folds 0-6 only. run_kfold skips folds that already wrote
# test_results.json, so re-running this script after an interrupt resumes.
set -u
cd "$(dirname "$0")/.."
LOG=ABLATE/logs; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=0
FOLDS=0,1,2,3,4,5,6

for arm in ABLATE_5G_half_ch64 ABLATE_5I_half_ch16; do
    echo "[$(date '+%F %T')] START $arm"
    python3 utils/run_kfold.py --repo . \
        --config "ABLATE/$arm.json" \
        --data-dir "data/[collapsed]t8192_ctx512_inner512_old" \
        --name "$arm" --folds "$FOLDS" \
        --log-dir "$LOG/folds" >> "$LOG/$arm.log" 2>&1
    echo "[$(date '+%F %T')] DONE $arm rc=$? ($(ls saved/models/${arm}*/test_results.json 2>/dev/null | wc -l)/7 folds)"
done
echo "[$(date '+%F %T')] GPU0_COMPLETE"
