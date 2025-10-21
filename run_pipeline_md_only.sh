#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/workspace/venv/bin/python"

if [ -x "$VENV_PY" ]; then
    PYTHON_CMD="$VENV_PY"
else
    PYTHON_CMD="$(command -v python3 || command -v python || true)"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No Python interpreter found. Install python3 or create workspace/venv." >&2
    exit 1
fi

exec "$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py" markdown "$@"
