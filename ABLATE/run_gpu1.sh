#!/usr/bin/env bash
# GPU 1 (RTX 2070, Turing -- fp16, since bf16 is emulated and ~4.7x slower):
#   champion control, fold 0 only (~6h)  ->  5E half (~35h)  ->  5H half+ch32 (~17h)
#
# The control runs FIRST and deliberately: the ablations train fp16 on this card
# while the deployed champion was trained bf16 on the 5090, so a difference could
# be architecture or precision. Scoring the unmodified architecture here on fold 0
# isolates that -- compare it against the champion's 0.9225 on the same fold.
set -u
cd "$(dirname "$0")/.."
LOG=ABLATE/logs; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=1

run () {   # arm, folds
    echo "[$(date '+%F %T')] START $1 folds=$2"
    python3 utils/run_kfold.py --repo . \
        --config "ABLATE/$1.json" \
        --data-dir "data/[collapsed]t8192_ctx512_inner512_old" \
        --name "$1" --folds "$2" \
        --log-dir "$LOG/folds" >> "$LOG/$1.log" 2>&1
    echo "[$(date '+%F %T')] DONE $1 rc=$? ($(ls saved/models/${1}*/test_results.json 2>/dev/null | wc -l) folds)"
}

run ABLATE_CONTROL_champion 0
run ABLATE_5E_half          0,1,2,3,4,5,6
run ABLATE_5H_half_ch32     0,1,2,3,4,5,6
echo "[$(date '+%F %T')] GPU1_COMPLETE"
