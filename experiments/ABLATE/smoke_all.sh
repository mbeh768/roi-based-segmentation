#!/usr/bin/env bash
# 2-epoch smoke of every ablation arm on GPU 0, before committing days of compute.
# Verifies each variant builds, trains, and completes its post-training test eval.
set -u
cd "$(dirname "$0")/.."
LOG=ABLATE/logs; mkdir -p "$LOG"

for arm in ABLATE_5E_half ABLATE_5G_half_ch64 ABLATE_5H_half_ch32 ABLATE_5I_half_ch16; do
    echo "=== $arm ==="
    CUDA_VISIBLE_DEVICES=0 python3 utils/run_kfold.py --repo . \
        --config "ABLATE/$arm.json" \
        --data-dir "data/[collapsed]t8192_ctx512_inner512_old" \
        --name "SMOKE_$arm" --folds 0 --epochs 2 --early-stop 2 \
        --log-dir /tmp/smoke_logs > "$LOG/smoke_$arm.log" 2>&1
    rc=$?
    res="saved/models/SMOKE_$arm/test_results.json"
    if [ $rc -eq 0 ] && [ -f "$res" ]; then
        echo "PASS $arm $(python3 -c "import json;print('test_iou=%.4f'%json.load(open('$res'))['test_iou'])")"
    else
        echo "FAIL $arm rc=$rc"
        tail -15 "$LOG/smoke_$arm.log"
    fi
done
echo "SMOKE_COMPLETE"
