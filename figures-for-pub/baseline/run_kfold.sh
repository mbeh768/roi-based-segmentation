#!/usr/bin/env bash
#
# Run leave-one-case-out k-fold training.
#
# Run FROM THE REPO ROOT. train.py resolves the config's data_dir and
# trainer.save_dir relative to the working directory, so running it from
# anywhere else silently looks for the data in the wrong place.
#
# Everything machine-specific is an environment variable:
#
#   PY           interpreter. Conda is usually absent from PATH in non-login
#                and ssh-spawned shells, so pass the env's python by ABSOLUTE
#                path.                                        Default: python
#   DEVICE       CUDA_VISIBLE_DEVICES value.                       Default: 0
#   CONFIG       config file.                             Default: config.json
#   FOLDS        "0-13", or a list like "0,3,7".               Default: 0-13
#   N_FOLDS      folds to split the cases into.                    Default: 14
#   NAME_PREFIX  run name stem; each fold gets -fNN appended.
#                                                    Default: from CONFIG name
#   DATA_DIR     override the config's data_dir.
#   BATCH_SIZE   override the config's batch_size.
#   EPOCHS       override the config's trainer.epochs.
#   EARLY_STOP   override the config's trainer.early_stop.
#   MIN_FREE_MB  VRAM required before launching.               Default: 24000
#   FORCE        1 to launch despite insufficient free VRAM.
#   DETACH       1 to run under `setsid nohup` and return immediately.
#
# Usage:
#   PY=/abs/path/to/env/bin/python bash run_kfold.sh
#   PY=... DETACH=1 bash run_kfold.sh
#   PY=... FOLDS=0,1,2 bash run_kfold.sh
#
# Folds that already wrote test_results.json are skipped, so re-running after
# an interruption resumes rather than restarting. An interrupted fold has no
# test_results.json and is therefore retried, not silently counted as done.
#
# To stop a detached run, kill the WHOLE PROCESS GROUP in one command. Killing
# only the trainer hands control back to this loop, which immediately starts
# the next fold and burns GPU time on work you are discarding:
#
#   PG=$(ps -o pgid= -p "$(pgrep -f run_kfold.sh | head -1)" | tr -d ' ')
#   kill -9 -- -"$PG"
#
set -euo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-0}"
CONFIG="${CONFIG:-config.json}"
FOLDS="${FOLDS:-0-13}"
N_FOLDS="${N_FOLDS:-14}"
NAME_PREFIX="${NAME_PREFIX:-}"
DATA_DIR="${DATA_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-}"
EARLY_STOP="${EARLY_STOP:-}"
MIN_FREE_MB="${MIN_FREE_MB:-24000}"
FORCE="${FORCE:-0}"
DETACH="${DETACH:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- preflight ------------------------------------------------------------
# Cheapest and most certain checks first: nothing here should wait behind a
# GPU that may be busy for hours.

[ -f train.py ] || { echo "ERROR: no train.py in $ROOT -- not a repo root" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: CONFIG=$CONFIG does not exist" >&2; exit 1; }

command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || {
    echo "ERROR: PY=$PY is not executable." >&2
    echo "       Pass the conda environment's python by absolute path; a" >&2
    echo "       relocated environment also leaves console scripts with stale" >&2
    echo "       shebangs, so prefer 'python -m <module>' over bin/ wrappers." >&2
    exit 1
}

# Resolve the effective data_dir and run name from the config, applying the
# same overrides that get passed to train.py, so the checks below inspect what
# the run will actually use rather than what the config says.
eval "$("$PY" - "$CONFIG" "$DATA_DIR" "$NAME_PREFIX" <<'PYEOF'
import json, shlex, sys
cfg = json.load(open(sys.argv[1]))
data_dir = sys.argv[2] or cfg['data_loader']['args']['data_dir']
name = sys.argv[3] or cfg['name']
print(f"CFG_DATA_DIR={shlex.quote(data_dir)}")
print(f"CFG_NAME={shlex.quote(name)}")
print(f"CFG_SAVE_DIR={shlex.quote(cfg['trainer']['save_dir'])}")
print(f"CFG_REQUIRE_ROI={int(bool(cfg['data_loader']['args'].get('require_roi', False)))}")
PYEOF
)"

[ -d "$CFG_DATA_DIR/images" ] || {
    echo "ERROR: no images/ under data_dir '$CFG_DATA_DIR'." >&2
    echo "       Run preprocess.py first, or set DATA_DIR." >&2
    exit 1
}

# ROIs are NOT produced by preprocess.py -- they come from annotate_globals.py
# and are copied in per case. Catch a missing one here rather than after the
# first fold has already claimed the GPU.
if [ "$CFG_REQUIRE_ROI" = "1" ]; then
    missing=0
    for d in "$CFG_DATA_DIR"/images/*/; do
        s="$(basename "$d")"
        [ -f "$d/roi_$s.pt" ] || { echo "  missing ROI: $d/roi_$s.pt" >&2; missing=1; }
    done
    [ "$missing" = "0" ] || {
        echo "ERROR: require_roi is set but ROIs are missing (see above)." >&2
        exit 1
    }
fi

"$PY" - <<'PYEOF' || exit 1
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"ERROR: {sys.executable} cannot import torch: {e}")
if not torch.cuda.is_available():
    sys.exit(f"ERROR: {sys.executable} has torch {torch.__version__} but no CUDA device")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {torch.cuda.get_device_name(0)}")
PYEOF

# The champion config needs ~24GB at batch_size 40. A co-tenant job holding a
# few GB is enough to OOM in backward, so check before committing to 14 folds.
if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_MB="$(nvidia-smi -i "$DEVICE" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU $DEVICE free: ${FREE_MB} MiB (need ${MIN_FREE_MB} MiB)"
    if [ "$FREE_MB" -lt "$MIN_FREE_MB" ] && [ "$FORCE" != "1" ]; then
        echo "ERROR: not enough free VRAM on device $DEVICE." >&2
        echo "       Wait for the card, lower BATCH_SIZE, raise MIN_FREE_MB," >&2
        echo "       or set FORCE=1 to launch anyway." >&2
        exit 1
    fi
fi

# --- expand the fold list -------------------------------------------------
if [[ "$FOLDS" == *-* ]]; then
    FOLD_LIST=($(seq "${FOLDS%%-*}" "${FOLDS##*-}"))
else
    FOLD_LIST=(${FOLDS//,/ })
fi

# MKL_THREADING_LAYER: without it the training processes abort at import with
# "incompatible with libgomp.so.1" on some conda stacks.
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$DEVICE"

mkdir -p kfold_logs
STAMP="$(date +%Y%m%d_%H%M%S)"

run_all() {
    local t0=$SECONDS
    echo "config=$CONFIG  data_dir=$CFG_DATA_DIR  name=$CFG_NAME"
    echo "folds: ${FOLD_LIST[*]}  (n_folds=$N_FOLDS)  device=$DEVICE"
    echo

    for f in "${FOLD_LIST[@]}"; do
        local fname
        fname="$(printf '%s-f%02d' "$CFG_NAME" "$f")"

        # A fold counts as done only once it has written test_results.json, so
        # an interrupted fold is retried rather than skipped.
        local done_dir=""
        for d in "$CFG_SAVE_DIR"models/"$fname" "$CFG_SAVE_DIR"models/"$fname"_[0-9]*; do
            [ -f "$d/test_results.json" ] && { done_dir="$d"; break; }
        done
        if [ -n "$done_dir" ]; then
            echo "[skip] fold $f -- $done_dir/test_results.json exists"
            continue
        fi

        local args=(train.py -c "$CONFIG" --n_folds "$N_FOLDS" --fold "$f" --name "$fname")
        [ -n "$DATA_DIR" ]   && args+=(--data_dir "$DATA_DIR")
        [ -n "$BATCH_SIZE" ] && args+=(--bs "$BATCH_SIZE")
        [ -n "$EPOCHS" ]     && args+=(--epochs "$EPOCHS")
        [ -n "$EARLY_STOP" ] && args+=(--early_stop "$EARLY_STOP")

        local log="kfold_logs/${fname}_${STAMP}.log"
        echo "=== fold $f -> $fname   ($log)"
        if "$PY" "${args[@]}" > "$log" 2>&1; then
            "$PY" - "$CFG_SAVE_DIR" "$fname" <<'PYEOF' || true
import json, sys, glob, os
save, name = sys.argv[1], sys.argv[2]
for d in sorted(glob.glob(os.path.join(save, 'models', name + '*'))):
    p = os.path.join(d, 'test_results.json')
    if os.path.exists(p):
        r = json.load(open(p))
        print(f"    test_iou {r.get('test_iou'):.4f}  best_epoch {r.get('best_epoch')}  "
              f"test={r.get('test_cases')}")
        break
PYEOF
        else
            echo "    *** fold $f FAILED (exit $?) -- see $log; continuing" >&2
            tail -5 "$log" >&2
        fi
    done

    echo
    "$PY" - "$CFG_SAVE_DIR" "$CFG_NAME" <<'PYEOF' || true
import json, glob, os, sys, statistics
save, name = sys.argv[1], sys.argv[2]
rows = []
for d in sorted(glob.glob(os.path.join(save, 'models', name + '-f*'))):
    p = os.path.join(d, 'test_results.json')
    if os.path.exists(p):
        rows.append(json.load(open(p)))
if rows:
    v = [r['test_iou'] for r in rows]
    print(f"{len(rows)} folds complete   test_iou mean {statistics.mean(v):.4f}"
          + (f"  sd {statistics.stdev(v):.4f}" if len(v) > 1 else ""))
else:
    print("no completed folds found")
PYEOF
    echo "elapsed $((SECONDS - t0))s"
}

echo "logs: $ROOT/kfold_logs/"
if [ "$DETACH" = "1" ]; then
    # setsid makes this a process-group leader so the whole tree can be killed
    # with one signal later (see the header).
    LOG="kfold_logs/kfold_${STAMP}.log"
    setsid nohup bash -c "$(declare -f run_all); run_all" > "$LOG" 2>&1 < /dev/null &
    echo "detached, pid $!  --  tail -f $ROOT/$LOG"
else
    run_all
fi
