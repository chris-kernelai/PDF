#!/bin/bash
################################################################################
# run_pipeline_local.sh
#
# Local pipeline runner: assumes documents already exist under data/,
# runs image description + filter + integration, and does NOT upload.
#
# Usage:
#   ./scripts/run_pipeline_local.sh [--images-only]
#
# Notes:
# - Skips fetching and uploading
# - Generates a session ID and passes it to 3a/3b/3c/3d and 4a/4b/4c/4d
################################################################################

set -e

IMAGES_ONLY=false
WORKERS=2

while [ $# -gt 0 ]; do
  case "$1" in
    --images-only|-i)
      IMAGES_ONLY=true
      shift
      ;;
    --batch-size)
      shift
      WORKERS="${1:-2}"
      shift
      ;;

    *)
      shift
      ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
ok()  { echo -e "${GREEN}✅ $1${NC}"; }
err() { echo -e "${RED}❌ $1${NC}"; }
warn(){ echo -e "${YELLOW}⚠️  $1${NC}"; }

trap 'err "Pipeline failed"; exit 1' ERR

log "Local pipeline start"

# Count markdowns that need image processing
PROCESSED_MDS=$(find data/processed -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
PROCESSED_IMAGES_MDS=$(find data/processed_images -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
NEED_IMAGE_PROCESSING=$(( PROCESSED_MDS - PROCESSED_IMAGES_MDS ))

if [ "$NEED_IMAGE_PROCESSING" -le 0 ]; then
  warn "No markdowns need image processing. Continuing to integration after checks."
fi

# Run PDF->MD conversion (skip only if --images-only is set)
if [ "$IMAGES_ONLY" = false ]; then
  TOTAL_PDFS=$(find data/to_process -name "doc_*.pdf" 2>/dev/null | wc -l | tr -d ' ')
  log "Converting PDFs to Markdown (found ${TOTAL_PDFS:-0} PDF(s))"
  python3 2_batch_convert_pdfs.py data/to_process data/processed --batch-size "$WORKERS" --extract-images || true
  ok "PDF conversion step finished"
fi

# Recompute counts after conversion
PROCESSED_MDS=$(find data/processed -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
PROCESSED_IMAGES_MDS=$(find data/processed_images -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
NEED_IMAGE_PROCESSING=$(( PROCESSED_MDS - PROCESSED_IMAGES_MDS ))

if [ "$NEED_IMAGE_PROCESSING" -le 0 ]; then
  warn "No image description work needed (processed=$PROCESSED_MDS, with_images=$PROCESSED_IMAGES_MDS)."
else
  # Generate session id
  PIPELINE_SESSION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4())[:8])")
  log "Session ID: $PIPELINE_SESSION_ID"

  # 3a: Prepare image batches
  log "3a: Preparing image batches"
  python3 3a_prepare_image_batches.py --session-id "$PIPELINE_SESSION_ID"

  # 3b: Upload batches
  log "3b: Uploading image batches"
  python3 3b_upload_batches.py --session-id "$PIPELINE_SESSION_ID"

  # 3c: Monitor
  log "3c: Monitoring image batches"
  MAX_RETRIES=60
  RETRY=0
  WAIT=60
  while [ $RETRY -lt $MAX_RETRIES ]; do
    OUT=$(python3 3c_monitor_batches.py --session-id "$PIPELINE_SESSION_ID" 2>&1 || true)
    echo "$OUT"
    echo "$OUT" | grep -q "All batches completed" && break
    echo "$OUT" | grep -q "Some batches failed" && { err "Some image batches failed"; exit 1; }
    RETRY=$((RETRY+1))
    [ $RETRY -lt $MAX_RETRIES ] && { log "Waiting $WAIT seconds... ($RETRY/$MAX_RETRIES)"; sleep $WAIT; }
  done
  [ $RETRY -ge $MAX_RETRIES ] && { err "Timeout monitoring image batches"; exit 1; }

  # 3d: Download results
  log "3d: Downloading image batch results"
  python3 3d_download_batch_results.py --session-id "$PIPELINE_SESSION_ID"
  ok "Image description pipeline complete"

  # 4a: Prepare filter batches
  log "4a: Preparing filter batches"
  python3 4a_prepare_filter_batches.py --session-id "$PIPELINE_SESSION_ID"

  # 4b: Upload filter batches
  log "4b: Uploading filter batches"
  python3 4b_upload_filter_batches.py --session-id "$PIPELINE_SESSION_ID"

  # 4c: Monitor filter
  log "4c: Monitoring filter batches"
  MAX_RETRIES=60
  RETRY=0
  WAIT=60
  while [ $RETRY -lt $MAX_RETRIES ]; do
    OUT=$(python3 4c_monitor_filter_batches.py --session-id "$PIPELINE_SESSION_ID" 2>&1 || true)
    echo "$OUT"
    echo "$OUT" | grep -q "All batches completed" && break
    echo "$OUT" | grep -q "Some batches failed" && { err "Some filter batches failed"; exit 1; }
    RETRY=$((RETRY+1))
    [ $RETRY -lt $MAX_RETRIES ] && { log "Waiting $WAIT seconds... ($RETRY/$MAX_RETRIES)"; sleep $WAIT; }
  done
  [ $RETRY -ge $MAX_RETRIES ] && { err "Timeout monitoring filter batches"; exit 1; }

  # 4d: Download filter results
  log "4d: Downloading filter results"
  python3 4d_download_filter_results.py --session-id "$PIPELINE_SESSION_ID"
  ok "Filter pipeline complete"
fi

# 5: Integrate descriptions locally (only if filtered outputs exist)
DESC_DIR=".generated/image_description_batches_outputs_filtered"
if [ -d "$DESC_DIR" ] && [ "$(find "$DESC_DIR" -type f -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]; then
  log "5: Integrating image descriptions"
  python3 5_integrate_descriptions.py
  ok "Integration complete"
else
  warn "Skipping integration: no filtered description outputs found in $DESC_DIR"
fi

log "Local pipeline complete"

