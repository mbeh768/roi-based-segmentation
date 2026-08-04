#!/usr/bin/env bash
# ===================================================================
#  Epicardium quantification - macOS / Linux launcher
#
#  Run:  ./run_quantify_mac_linux.sh
#
#  The first run creates a private Python environment in .venv next to
#  this script and installs everything needed (shared with
#  run_mac_linux.sh -- if you already ran that once, this reuses the
#  same environment). Nothing is installed system-wide.
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
STAMP="$VENV_DIR/.deps_installed"

if ! command -v python3 >/dev/null 2>&1; then
    cat <<'EOF'

=====================================================
 Python 3 was not found on this computer.

 macOS:  brew install python
 Ubuntu: sudo apt install python3 python3-venv
 Or:     https://www.python.org/downloads/

 Then run this script again.
=====================================================

EOF
    exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating Python environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"

if [ ! -f "$STAMP" ]; then
    echo
    echo "Installing dependencies. This happens only once."
    echo
    "$VPY" install_deps.py --auto
    touch "$STAMP"
fi

"$VPY" -u quantify_epicardium.py "$@"
