#!/usr/bin/env bash
# EXPERIMENT 13 on gpu-box: both arms, 14-fold leave-one-case-out, sequential.
#
#   EXPERIMENT_13          neighbours active
#   EXPERIMENT_13_CONTROL  zero_neighbours=True -- identical parameter count and
#                          code path, no context. The difference between the two
#                          is context with capacity held exactly fixed, which is
#                          what makes the result interpretable given the 1x1
#                          projection adds ~2.6M parameters over the champion.
#
# Both arms use the full 14 folds so they are directly comparable to
# EXPERIMENT 11's 0.9243. run_kfold skips folds that already wrote
# test_results.json, so re-running this resumes rather than restarts.
#
# ~16s/epoch at 8 workers => roughly 12h per arm.
set -u
BASE=/mnt/1TBssd/mayo_kfold
REPO=$BASE/att_vent_seg
LOG=$REPO/EXPERIMENT13/logs
mkdir -p "$LOG"
export MKL_THREADING_LAYER=GNU

# The conda env explicitly: gpu-box's `python3` is the system interpreter and has
# no numpy/torch. run_kfold launches train.py via sys.executable, so setting it
# here carries through to the children.
PY=/mnt/500GBnvme/miniconda3/envs/giacomelli/bin/python
"$PY" -c "import numpy, torch" || { echo "FATAL: $PY cannot import numpy/torch"; exit 1; }

cd "$REPO"
for arm in EXPERIMENT_13 EXPERIMENT_13_CONTROL; do
    echo "[$(date '+%F %T')] START $arm"
    "$PY" utils/run_kfold.py \
        --base-dir "$BASE" --repo "$REPO" \
        --config "$REPO/EXPERIMENT13/$arm.json" \
        --data-dir "data/[collapsed]t8192_ctx512_inner512_old" \
        --name "$arm" \
        --log-dir "$LOG/folds" >> "$LOG/$arm.log" 2>&1
    rc=$?
    n=$(ls "$BASE"/saved/models/${arm}*/test_results.json 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] DONE $arm rc=$rc ($n/14 folds)"
done
echo "[$(date '+%F %T')] EXP13_COMPLETE"
