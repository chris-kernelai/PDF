#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run_pipeline_images_only.sh <min_doc_id> <max_doc_id> [batch_size] [-- extra run_pipeline.py args]

Examples:
  ./run_pipeline_images_only.sh 27000 27199
  ./run_pipeline_images_only.sh 27000 27199 50 -- --image-batch-size 75
EOF
}

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

BATCH_SIZE=100

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    BATCH_SIZE=$1
    shift
fi

EXTRA_ARGS=("$@")

PDF_DIR="$SCRIPT_DIR/workspace/data/to_process"

PDFS=()
for doc_id in $(seq "$MIN_DOC_ID" "$MAX_DOC_ID"); do
    pdf_path="$PDF_DIR/doc_${doc_id}.pdf"
    if [ -f "$pdf_path" ]; then
        PDFS+=("$pdf_path")
    fi
done

if [ ${#PDFS[@]} -eq 0 ]; then
    echo "No PDFs found in range $MIN_DOC_ID-$MAX_DOC_ID under $PDF_DIR" >&2
    exit 0
fi

total=${#PDFS[@]}
index=0
batch=1
while [ $index -lt $total ]; do
    chunk=("${PDFS[@]:index:BATCH_SIZE}")
    echo "Processing ${#chunk[@]} PDFs (batch $batch)"
    cmd=("$PYTHON_CMD" "$SCRIPT_DIR/run_pipeline.py" images)
    cmd+=("${EXTRA_ARGS[@]}")
    for pdf in "${chunk[@]}"; do
        cmd+=("--pdf" "$pdf")
    done
    "${cmd[@]}"
    index=$((index + BATCH_SIZE))
    batch=$((batch + 1))
done

echo "Images-only pipeline completed for $total PDFs."
