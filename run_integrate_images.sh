#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_integrate_images.sh [--profile <aws_profile>] --session-id <session_id> [options]

Integrate existing image descriptions from a Gemini batch session and upload to Supabase.

Required:
  --session-id <id>          Session ID from previous Gemini batch run

Optional:
  --profile <aws_profile>    AWS profile to use
  --min-doc-id <id>          Filter to only process docs >= this ID
  --max-doc-id <id>          Filter to only process docs <= this ID
  --download                 Download results from GCS before integrating
  --skip-upload              Don't upload to Supabase (just integrate)

Examples:
  ./run_integrate_images.sh --session-id abc123de
  ./run_integrate_images.sh --profile local --session-id abc123de --min-doc-id 27290 --max-doc-id 27344
  ./run_integrate_images.sh --session-id abc123de --download
  ./run_integrate_images.sh --session-id abc123de --skip-upload
EOF
}

PROFILE=""
SESSION_ID=""
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
        --session-id)
            if [[ $# -lt 2 ]]; then
                echo "Error: --session-id requires a value" >&2
                exit 1
            fi
            SESSION_ID="$2"
            shift 2
            ;;
        --session-id=*)
            SESSION_ID="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$SESSION_ID" ]; then
    echo "Error: --session-id is required" >&2
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

CMD=("$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py")
if [ -n "$PROFILE" ]; then
    CMD+=("--aws-profile" "$PROFILE")
fi
CMD+=("integrate-images" "--session-id" "$SESSION_ID")

# Add any remaining args
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    CMD+=("${POSITIONAL_ARGS[@]}")
fi

echo "Running: ${CMD[*]}" >&2
exec "${CMD[@]}"
