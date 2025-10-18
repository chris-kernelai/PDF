#!/bin/bash

################################################################################
# PDF Processing Pipeline - Batch Runner
#
# This script runs the entire PDF processing pipeline in batches of 500 documents
# to avoid GPU memory overflow. It processes documents within a specified ID range.
#
# Usage: ./run_pipeline.sh <min_doc_id> <max_doc_id> [batch_size] [--background] [--skip-download]
#
# Examples:
#   ./run_pipeline.sh 27000 30000 500             # Run interactively
#   ./run_pipeline.sh 27000 30000 500 --background # Run in background with nohup
#   ./run_pipeline.sh 27000 30000 500 --skip-download # Skip document fetching step
################################################################################

# Parse flags
BACKGROUND_MODE=false
SKIP_DOWNLOAD=false
ARGS=()

for arg in "$@"; do
    case $arg in
        --background|-b)
            BACKGROUND_MODE=true
            ;;
        --skip-download|-s)
            SKIP_DOWNLOAD=true
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Reset positional parameters to non-flag arguments
set -- "${ARGS[@]}"

# If background mode and this is the parent process, relaunch with nohup
if [ "$BACKGROUND_MODE" = true ] && [ -z "${PIPELINE_CHILD:-}" ]; then
    LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
    echo "🚀 Starting pipeline in background mode..."
    echo "📝 Log file: $LOG_FILE"
    echo ""

    # Relaunch with nohup, passing through flags
    EXTRA_FLAGS=""
    [ "$SKIP_DOWNLOAD" = true ] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-download"
    PIPELINE_CHILD=1 nohup "$0" "$@" $EXTRA_FLAGS > "$LOG_FILE" 2>&1 &
    PID=$!

    echo "✅ Pipeline started with PID: $PID"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check if running:"
    echo "  ps aux | grep $PID"
    echo ""
    echo "Kill if needed:"
    echo "  kill $PID"
    echo ""

    exit 0
fi

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to cleanup on error
cleanup_on_error() {
    error "Pipeline failed. Cleaning up..."
    # Don't delete PDFs on error - we may want to retry
    exit 1
}

trap cleanup_on_error ERR

# Parse arguments
if [ $# -lt 2 ]; then
    error "Usage: $0 <min_doc_id> <max_doc_id> [batch_size] [--background|-b] [--skip-download|-s]"
    echo ""
    echo "Examples:"
    echo "  $0 27000 30000 500                # Run interactively"
    echo "  $0 27000 30000 500 --background   # Run in background with nohup"
    echo "  $0 27000 30000 500 --skip-download # Skip document fetching step"
    exit 1
fi

MIN_DOC_ID=$1
MAX_DOC_ID=$2
BATCH_SIZE=${3:-500}  # Default to 500 if not specified

# Validate arguments
if ! [[ "$MIN_DOC_ID" =~ ^[0-9]+$ ]] || ! [[ "$MAX_DOC_ID" =~ ^[0-9]+$ ]]; then
    error "min_doc_id and max_doc_id must be integers"
    exit 1
fi

if [ "$MIN_DOC_ID" -ge "$MAX_DOC_ID" ]; then
    error "min_doc_id must be less than max_doc_id"
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -le 0 ]; then
    error "batch_size must be a positive integer"
    exit 1
fi

# Check required commands
for cmd in python3; do
    if ! command_exists "$cmd"; then
        error "Required command not found: $cmd"
        exit 1
    fi
done

# Display configuration
log "=========================================="
log "PDF Processing Pipeline"
log "=========================================="
info "Document ID range: $MIN_DOC_ID to $MAX_DOC_ID"
info "Batch size: $BATCH_SIZE documents"
info "Skip download: $SKIP_DOWNLOAD"
info "Working directory: $(pwd)"
log "=========================================="
echo ""

# Step 1: Fetch documents (optional)
if [ "$SKIP_DOWNLOAD" = false ]; then
    log "STEP 1: Fetching documents (ID range: $MIN_DOC_ID - $MAX_DOC_ID)"
    python3 1_fetch_documents.py \
        --min-doc-id "$MIN_DOC_ID" \
        --max-doc-id "$MAX_DOC_ID"

    if [ $? -ne 0 ]; then
        error "Document fetching failed"
        exit 1
    fi

    log "✓ Document fetching complete"
    echo ""
else
    log "STEP 1: Skipping document fetch (--skip-download flag set)"
    echo ""
fi

# Count total PDFs to process
TOTAL_PDFS=$(find data/to_process -name "doc_*.pdf" 2>/dev/null | wc -l | tr -d ' ')

# Count markdowns that need image processing
PROCESSED_MDS=$(find data/processed -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
PROCESSED_IMAGES_MDS=$(find data/processed_images -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
NEED_IMAGE_PROCESSING=$(( PROCESSED_MDS - PROCESSED_IMAGES_MDS ))

if [ "$TOTAL_PDFS" -eq 0 ] && [ "$NEED_IMAGE_PROCESSING" -le 0 ]; then
    warn "No PDFs found to process and no markdowns need image processing. Exiting."
    exit 0
fi

if [ "$TOTAL_PDFS" -eq 0 ]; then
    info "No PDFs to process, but $NEED_IMAGE_PROCESSING markdown(s) need image processing"
    info "Skipping PDF conversion, proceeding to image description pipeline"
fi

if [ "$TOTAL_PDFS" -gt 0 ]; then
    info "Found $TOTAL_PDFS PDFs to process"
    NUM_BATCHES=$(( (TOTAL_PDFS + BATCH_SIZE - 1) / BATCH_SIZE ))
    info "Will process in $NUM_BATCHES batch(es) of up to $BATCH_SIZE documents each"
else
    NUM_BATCHES=1  # One batch for image processing only
fi
echo ""

# Process in batches
BATCH_NUM=1
PROCESSED_COUNT=0

# Run at least once if there's work to do (PDFs or image processing needed)
while [ "$PROCESSED_COUNT" -lt "$TOTAL_PDFS" ] || [ "$BATCH_NUM" -eq 1 -a "$NEED_IMAGE_PROCESSING" -gt 0 ]; do
    log "=========================================="
    log "BATCH $BATCH_NUM/$NUM_BATCHES"
    log "=========================================="

    # Step 2: Convert PDFs to Markdown (only if there are PDFs)
    if [ "$TOTAL_PDFS" -gt 0 ]; then
        # Count remaining PDFs
        REMAINING_PDFS=$(find data/to_process -name "doc_*.pdf" 2>/dev/null | wc -l | tr -d ' ')
        CURRENT_BATCH_SIZE=$(( REMAINING_PDFS < BATCH_SIZE ? REMAINING_PDFS : BATCH_SIZE ))

        info "Processing $CURRENT_BATCH_SIZE documents in this batch"
        info "Progress: $PROCESSED_COUNT/$TOTAL_PDFS documents processed so far"
        echo ""

        log "STEP 2: Converting PDFs to Markdown (batch $BATCH_NUM)"

        # Create a temporary marker to track which PDFs are in this batch
        BATCH_PDFS=$(find data/to_process -name "doc_*.pdf" 2>/dev/null | head -n "$BATCH_SIZE")
        BATCH_PDF_COUNT=$(echo "$BATCH_PDFS" | grep -c "doc_" || true)

        if [ "$BATCH_PDF_COUNT" -eq 0 ]; then
            warn "No PDFs left to process in this batch"
            break
        fi

        python3 2_batch_convert_pdfs.py \
            data/to_process \
            data/processed \
            --batch-size 2 \
            --extract-images

        if [ $? -ne 0 ]; then
            error "PDF conversion failed for batch $BATCH_NUM"
            exit 1
        fi

        log "✓ PDF conversion complete for batch $BATCH_NUM"
        echo ""
    else
        log "STEP 2: Skipping PDF conversion (no PDFs to process)"
        CURRENT_BATCH_SIZE=0
        echo ""
    fi

    # Step 3: Image Description Pipeline
    log "STEP 3: Running image description pipeline (batch $BATCH_NUM)"

    # 3a: Prepare image batches
    log "  3a: Preparing image batches..."
    python3 3a_prepare_image_batches.py

    # 3b: Upload batches to Gemini
    log "  3b: Uploading batches to Gemini..."
    python3 3b_upload_batches.py

    # 3c: Monitor batch progress (with retry)
    log "  3c: Monitoring batch progress..."
    MAX_RETRIES=60  # 60 retries = 60 minutes max wait
    RETRY_COUNT=0
    WAIT_TIME=60  # 60 seconds between checks

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        MONITOR_OUTPUT=$(python3 3c_monitor_batches.py 2>&1)
        echo "$MONITOR_OUTPUT"

        # Check if all jobs completed successfully
        if echo "$MONITOR_OUTPUT" | grep -q "✅ All batch jobs completed successfully"; then
            log "  ✓ All batch jobs completed!"
            break
        fi

        # Check if any jobs failed
        if echo "$MONITOR_OUTPUT" | grep -q "❌ Some batch jobs failed"; then
            error "Some batch jobs failed"
            exit 1
        fi

        # Jobs still processing
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            info "  Jobs still processing... waiting ${WAIT_TIME}s before retry $RETRY_COUNT/$MAX_RETRIES"
            sleep $WAIT_TIME
        else
            error "Timeout waiting for batch jobs to complete after $((MAX_RETRIES * WAIT_TIME / 60)) minutes"
            exit 1
        fi
    done

    # 3d: Download results
    log "  3d: Downloading batch results..."
    python3 3d_download_batch_results.py

    log "✓ Image description pipeline complete for batch $BATCH_NUM"
    echo ""

    # Step 4: Filter Pipeline
    log "STEP 4: Running filter pipeline (batch $BATCH_NUM)"

    # 4a: Prepare filter batches
    log "  4a: Preparing filter batches..."
    python3 4a_prepare_filter_batches.py

    # 4b: Upload filter batches
    log "  4b: Uploading filter batches..."
    python3 4b_upload_filter_batches.py

    # 4c: Monitor filter progress (with retry)
    log "  4c: Monitoring filter progress..."
    MAX_RETRIES=60  # 60 retries = 60 minutes max wait
    RETRY_COUNT=0
    WAIT_TIME=60  # 60 seconds between checks

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        MONITOR_OUTPUT=$(python3 4c_monitor_filter_batches.py 2>&1)
        echo "$MONITOR_OUTPUT"

        # Check if all jobs completed successfully
        if echo "$MONITOR_OUTPUT" | grep -q "✅ All batch jobs completed successfully"; then
            log "  ✓ All filter jobs completed!"
            break
        fi

        # Check if any jobs failed
        if echo "$MONITOR_OUTPUT" | grep -q "❌ Some batch jobs failed"; then
            error "Some filter jobs failed"
            exit 1
        fi

        # Jobs still processing
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            info "  Filter jobs still processing... waiting ${WAIT_TIME}s before retry $RETRY_COUNT/$MAX_RETRIES"
            sleep $WAIT_TIME
        else
            error "Timeout waiting for filter jobs to complete after $((MAX_RETRIES * WAIT_TIME / 60)) minutes"
            exit 1
        fi
    done

    # 4d: Download filter results
    log "  4d: Downloading filter results..."
    python3 4d_download_filter_results.py

    log "✓ Filter pipeline complete for batch $BATCH_NUM"
    echo ""

    # Step 5: Integrate descriptions
    log "STEP 5: Integrating image descriptions (batch $BATCH_NUM)"
    python3 5_integrate_descriptions.py

    if [ $? -ne 0 ]; then
        error "Description integration failed for batch $BATCH_NUM"
        exit 1
    fi

    log "✓ Description integration complete for batch $BATCH_NUM"
    echo ""

    # Step 6: Upload to S3 and Supabase
    log "STEP 6: Uploading to S3 and Supabase (batch $BATCH_NUM)"
    python3 5a_upload.py

    if [ $? -ne 0 ]; then
        error "Upload failed for batch $BATCH_NUM"
        exit 1
    fi

    log "✓ Upload complete for batch $BATCH_NUM"
    echo ""

    # Step 7: Cleanup
    log "STEP 7: Cleaning up batch $BATCH_NUM"

    # Delete .generated directory
    if [ -d ".generated" ]; then
        log "  Removing .generated directory..."
        rm -rf .generated
        log "  ✓ Removed .generated"
    fi

    # Delete images directory
    if [ -d "data/images" ]; then
        log "  Removing data/images directory..."
        rm -rf data/images/*
        log "  ✓ Cleared data/images"
    fi

    # Delete the PDFs that were just processed
    # Only delete PDFs that have corresponding markdown files in data/processed
    log "  Removing processed PDFs..."
    DELETED_COUNT=0
    for pdf in data/to_process/doc_*.pdf; do
        if [ -f "$pdf" ]; then
            # Extract document ID
            DOC_ID=$(basename "$pdf" .pdf)
            # Check if corresponding markdown exists
            if [ -f "data/processed/${DOC_ID}.md" ]; then
                rm -f "$pdf"
                DELETED_COUNT=$((DELETED_COUNT + 1))
            fi
        fi
    done
    log "  ✓ Removed $DELETED_COUNT processed PDFs"

    log "✓ Cleanup complete for batch $BATCH_NUM"
    echo ""

    # Update counters
    PROCESSED_COUNT=$((PROCESSED_COUNT + CURRENT_BATCH_SIZE))
    BATCH_NUM=$((BATCH_NUM + 1))

    # If we're only doing image processing (no PDFs), exit after one iteration
    if [ "$TOTAL_PDFS" -eq 0 ]; then
        log "Image processing complete (no PDFs to process)!"
        break
    fi

    # Check if there are more PDFs to process
    REMAINING_PDFS=$(find data/to_process -name "doc_*.pdf" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$REMAINING_PDFS" -eq 0 ]; then
        log "All PDFs processed!"
        break
    fi

    # Brief pause between batches
    log "Pausing 10 seconds before next batch..."
    sleep 10
    echo ""
done

# Final summary
log "=========================================="
log "PIPELINE COMPLETE"
log "=========================================="
info "Total documents processed: $PROCESSED_COUNT"
info "Total batches: $BATCH_NUM"
log "=========================================="
echo ""

log "✓ All done! Pipeline completed successfully."
