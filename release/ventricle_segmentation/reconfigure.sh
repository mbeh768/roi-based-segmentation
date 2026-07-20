#!/usr/bin/env bash
# ===================================================================
#  Switch between the GPU and CPU version of PyTorch.
#
#  Reinstalls PyTorch only; the other packages are left alone.
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "No environment found yet. Run ./run_mac_linux.sh first."
    exit 1
fi

echo
echo "Which version of PyTorch should be installed?"
echo "   1) GPU  - needs an NVIDIA graphics card, about 3 GB"
echo "   2) CPU  - works anywhere, about 300 MB"
echo
read -r -p "Enter 1 or 2: " CHOICE

case "$CHOICE" in
    1) .venv/bin/python install_deps.py --gpu --yes ;;
    2) .venv/bin/python install_deps.py --cpu --yes ;;
    *) echo "Not a valid choice. Nothing was changed."; exit 1 ;;
esac
