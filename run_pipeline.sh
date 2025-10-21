#!/bin/bash
set -euo pipefail

echo "======================================" >&2
echo "DEBUG: run_pipeline.sh starting" >&2
echo "DEBUG: Arguments: $@" >&2
echo "======================================" >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "DEBUG: SCRIPT_DIR=$SCRIPT_DIR" >&2

VENV_PY="$SCRIPT_DIR/workspace/venv/bin/python"
echo "DEBUG: Checking for venv at: $VENV_PY" >&2

if [ -x "$VENV_PY" ]; then
    PYTHON_CMD="$VENV_PY"
    echo "DEBUG: Using venv python: $PYTHON_CMD" >&2
else
    PYTHON_CMD="$(command -v python3 || command -v python || true)"
    echo "DEBUG: Using system python: $PYTHON_CMD" >&2
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No Python interpreter found. Install python3 or create workspace/venv." >&2
    exit 1
fi

echo "DEBUG: Python version:" >&2
"$PYTHON_CMD" --version >&2

echo "DEBUG: Executing: $PYTHON_CMD $SCRIPT_DIR/run_pipeline.py full $@" >&2
exec "$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py" full "$@"
