#!/usr/bin/env bash
#
# Run the ROI ablation across all 14 leave-one-case-out folds.
#
# Run this ON the GPU machine, FROM THE REPO ROOT, against a clone of this repo
# whose layout matches the development machine -- data/ and saved/ inside the
# repo. train.py resolves the config's data_dir and trainer.save_dir relative to
# the working directory, so running it from anywhere else silently looks for the
# data in the wrong place.
#
# Nothing here is host-specific; everything that differs between machines is an
# environment variable:
#
#   BASE_CONFIG REQUIRED. The json this ablation derives from; each ablation
#               gets its own, so there is deliberately no default.
#   PY          interpreter to use. Conda is frequently absent from PATH in
#               non-login and remote-spawned shells, so pass the environment's
#               python by ABSOLUTE path. Default: whatever `python` resolves to.
#   DEVICE      CUDA_VISIBLE_DEVICES value.                      Default: 0
#   ARMS        comma-separated subset of the arms.     Default: noise_guided
#               ("noise" also drops ROI-guided tile selection and costs ~4.8x
#               the steps per fold; defined but not run by default.)
#   FOLDS       "0-13", or a list like "0,3,7".                  Default: 0-13
#   DATA_DIR    override the data directory named in the flagship config.
#   BATCH_SIZE  override the flagship's batch size.
#   MIN_FREE_MB VRAM required before launching.               Default: 24000
#   FORCE       set to 1 to launch despite insufficient free VRAM.
#   DETACH      set to 1 to run under `setsid nohup` and return immediately.
#
# Usage:
#   PY=/abs/path/to/env/bin/python bash roi-ablation/run_all.sh
#   PY=/abs/path/to/env/bin/python DETACH=1 bash roi-ablation/run_all.sh
#   PY=... ARMS=noise_guided FOLDS=0,1,2 bash roi-ablation/run_all.sh
#
# Folds that already wrote test_results.json are skipped, so re-running this
# after an interruption resumes rather than restarting. An interrupted fold has
# no test_results.json and is therefore retried, not silently counted as done.
#
# To stop a detached run, kill the WHOLE PROCESS GROUP in one command. Killing
# only the trainer hands control back to the driver, which immediately launches
# the next fold and burns GPU time on work you are discarding:
#
#   PG=$(ps -o pgid= -p "$(pgrep -f train_ablation.py | head -1)" | tr -d ' ')
#   kill -9 -- -"$PG"
#
set -euo pipefail

PY="${PY:-python}"
DEVICE="${DEVICE:-0}"
ARMS="${ARMS:-noise_guided}"
FOLDS="${FOLDS:-0-13}"
BASE_CONFIG="${BASE_CONFIG:-}"
DATA_DIR="${DATA_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MIN_FREE_MB="${MIN_FREE_MB:-24000}"
FORCE="${FORCE:-0}"
DETACH="${DETACH:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

# --- preflight ------------------------------------------------------------
# Each check below has cost one debugging session at some point; failing loudly
# here is cheaper than failing 40 minutes into fold 0.

[ -f train.py ] || { echo "ERROR: no train.py in $ROOT -- not a repo root" >&2; exit 1; }

# Cheapest and most certain check first: a missing required argument should not
# wait behind a GPU that may be busy for hours.
[ -n "$BASE_CONFIG" ] || { echo "ERROR: set BASE_CONFIG to the json this ablation derives from." >&2; exit 1; }
[ -f "$BASE_CONFIG" ] || { echo "ERROR: BASE_CONFIG=$BASE_CONFIG does not exist." >&2; exit 1; }

command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || {
    echo "ERROR: PY=$PY is not executable." >&2
    echo "       Pass the conda environment's python by absolute path;" >&2
    echo "       relocated environments also leave console scripts with stale" >&2
    echo "       shebangs, so prefer 'python -m <module>' over bin/ wrappers." >&2
    exit 1
}

"$PY" - <<'EOF' || exit 1
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"ERROR: {sys.executable} cannot import torch: {e}")
if not torch.cuda.is_available():
    sys.exit(f"ERROR: {sys.executable} has torch {torch.__version__} but no CUDA device")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"{torch.cuda.get_device_name(0)}")
EOF

# The flagship config needs ~24GB at batch_size 40. A co-tenant job holding a few
# GB is enough to OOM in backward, so check before committing to a 14-fold sweep.
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

# --- assemble the command -------------------------------------------------
ARGS=(roi-ablation/train_ablation.py --base-config "$BASE_CONFIG"
      --arms "$ARMS" --folds "$FOLDS" --device "$DEVICE")
[ -n "$DATA_DIR" ]   && ARGS+=(--data-dir "$DATA_DIR")
[ -n "$BATCH_SIZE" ] && ARGS+=(--batch-size "$BATCH_SIZE")

mkdir -p roi-ablation/logs
LOG="roi-ablation/logs/roi_ablation_$(date +%Y%m%d_%H%M%S).log"

# MKL_THREADING_LAYER: without it the child training processes abort at import
# with "incompatible with libgomp.so.1" on some conda stacks.
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "arms=$ARMS folds=$FOLDS device=$DEVICE"
echo "log: $ROOT/$LOG"

if [ "$DETACH" = "1" ]; then
    # setsid makes the driver a process-group leader so the whole tree can be
    # killed with one signal later (see the header).
    setsid nohup "$PY" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
    echo "detached, pid $!  --  tail -f $ROOT/$LOG"
else
    "$PY" "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
