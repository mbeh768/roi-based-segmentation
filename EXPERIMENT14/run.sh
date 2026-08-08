#!/usr/bin/env bash
# EXPERIMENT 14 on gpu-box: parameter-matched decoder rebalance, 14-fold LOCO.
# Same protocol as EXPERIMENT 11, so the result is directly comparable to 0.9243.
set -u
BASE=/mnt/1TBssd/mayo_kfold
REPO=$BASE/att_vent_seg
LOG=$REPO/EXPERIMENT14/logs
mkdir -p "$LOG"
export MKL_THREADING_LAYER=GNU
PY=/mnt/500GBnvme/miniconda3/envs/giacomelli/bin/python
"$PY" -c "import numpy, torch" || { echo "FATAL: $PY cannot import numpy/torch"; exit 1; }

cd "$REPO"
arm=EXPERIMENT_14_rebalanced
echo "[$(date '+%F %T')] START $arm"
"$PY" utils/run_kfold.py \
    --base-dir "$BASE" --repo "$REPO" \
    --config "$REPO/EXPERIMENT14/EXPERIMENT_14.json" \
    --data-dir "data/[collapsed]t8192_ctx512_inner512_old" \
    --name "$arm" \
    --log-dir "$LOG/folds" >> "$LOG/$arm.log" 2>&1
echo "[$(date '+%F %T')] DONE $arm rc=$? ($(ls $BASE/saved/models/${arm}*/test_results.json 2>/dev/null | wc -l)/14 folds)"
echo "[$(date '+%F %T')] EXP14_COMPLETE"
