#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_pipeline_images_only.sh [options] <min_doc_id> <max_doc_id> [doc_batch_size] [-- extra run_pipeline.py args]

Options:
  --profile <aws_profile>    AWS profile to use
  --cpu                      Force CPU mode (disable GPU)

Examples:
  ./run_pipeline_images_only.sh 27000 27199
  ./run_pipeline_images_only.sh --profile local 27000 27199 50
  ./run_pipeline_images_only.sh --cpu 27000 27199 10
  ./run_pipeline_images_only.sh --profile local --cpu 27000 27199 50 -- --image-batch-size 75
EOF
}

PROFILE=""
CPU_MODE=""
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
        --cpu)
            CPU_MODE="true"
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

if [ $# -lt 2 ]; then
    usage
    exit 1
fi

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

MIN_DOC_ID=$1
MAX_DOC_ID=$2
shift 2

if ! [[ "$MIN_DOC_ID" =~ ^[0-9]+$ ]] || ! [[ "$MAX_DOC_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: min_doc_id and max_doc_id must be integers" >&2
    exit 1
fi

if [ "$MIN_DOC_ID" -gt "$MAX_DOC_ID" ]; then
    echo "Error: min_doc_id must be <= max_doc_id" >&2
    exit 1
fi

DOC_BATCH_SIZE=""
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    DOC_BATCH_SIZE=$1
    shift
fi

EXTRA_ARGS=("$@")

CMD=("$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py")
if [ -n "$PROFILE" ]; then
    CMD+=("--aws-profile" "$PROFILE")
fi
CMD+=("images" "$MIN_DOC_ID" "$MAX_DOC_ID")
if [ -n "$DOC_BATCH_SIZE" ]; then
    CMD+=("--doc-batch-size" "$DOC_BATCH_SIZE")
fi
if [ -n "$CPU_MODE" ]; then
    CMD+=("--cpu")
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${CMD[*]}" >&2
exec "${CMD[@]}"
