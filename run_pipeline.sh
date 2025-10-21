#!/bin/bash
set -euo pipefail

PROFILE=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            if [[ $# -lt 2 ]]; then
                echo "Error: --profile requires a value" >&2
                exit 1
            fi
            PROFILE="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        --)
            POSITIONAL_ARGS+=("$1")
            shift
            POSITIONAL_ARGS+=("$@")
            break
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

echo "======================================" >&2
echo "DEBUG: run_pipeline.sh starting" >&2
if [ -n "$PROFILE" ]; then
    echo "DEBUG: AWS profile override: $PROFILE" >&2
fi
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

if [ -n "$PROFILE" ]; then
    export AWS_PROFILE="$PROFILE"
fi

echo "DEBUG: Python version:" >&2
"$PYTHON_CMD" --version >&2

CMD=("$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py")
if [ -n "$PROFILE" ]; then
    CMD+=("--aws-profile" "$PROFILE")
fi
CMD+=("full" "$@")

echo "DEBUG: Executing: ${CMD[*]}" >&2
exec "${CMD[@]}"
