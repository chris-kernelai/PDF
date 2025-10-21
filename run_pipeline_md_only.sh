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
            shift
            POSITIONAL_ARGS+=("--")
            POSITIONAL_ARGS+=("$@")
            break
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

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

if [ -n "$PROFILE" ]; then
    export AWS_PROFILE="$PROFILE"
fi

CMD=("$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py")
if [ -n "$PROFILE" ]; then
    CMD+=("--aws-profile" "$PROFILE")
fi
CMD+=("markdown" "$@")

exec "${CMD[@]}"
