#!/bin/bash

################################################################################
# PDF Processing Pipeline - Images Only Mode
#
# This script processes documents that have DOCLING but are missing DOCLING_IMG
# representation. It fetches and processes documents in batches to avoid memory
# issues and downloads getting out of control.
#
# Usage: ./run_pipeline_images_only.sh [batch_size] [--batch-size <workers>] [--background] [--cpu]
#
# Examples:
#   ./run_pipeline_images_only.sh 500                    # Run interactively with batch size 500
#   ./run_pipeline_images_only.sh 500 --batch-size 4     # Use 4 parallel workers
#   ./run_pipeline_images_only.sh 500 --background       # Run in background with nohup
#   ./run_pipeline_images_only.sh 500 --cpu --batch-size 8  # CPU mode with 8 workers
################################################################################

# Parse flags
BACKGROUND_MODE=false
CPU_MODE=false
WORKERS=2  # Default to 2 workers
ARGS=()

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --background|-b)
            BACKGROUND_MODE=true
            ;;
        --cpu|-c)
            CPU_MODE=true
            ;;
        --batch-size)
            ((i++))
            WORKERS="${!i}"
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
    ((i++))
done

# Reset positional parameters to non-flag arguments
set -- "${ARGS[@]}"

# If background mode and this is the parent process, relaunch with nohup
if [ "$BACKGROUND_MODE" = true ] && [ -z "${PIPELINE_CHILD:-}" ]; then
    LOG_FILE="pipeline_images_only_$(date +%Y%m%d_%H%M%S).log"
    echo "🚀 Starting images-only pipeline in background mode..."
    echo "📝 Log file: $LOG_FILE"
    echo ""

    # Relaunch with nohup, passing through flags
    EXTRA_FLAGS=""
    [ "$CPU_MODE" = true ] && EXTRA_FLAGS="$EXTRA_FLAGS --cpu"
    [ "$WORKERS" != "2" ] && EXTRA_FLAGS="$EXTRA_FLAGS --batch-size $WORKERS"
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
    
    # Log session failure if we have a session ID
    if [ -n "${PIPELINE_SESSION_ID:-}" ]; then
        SESSION_ERROR_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$SESSION_ERROR_TIME | Session: $PIPELINE_SESSION_ID | Batch: ${BATCH_NUM:-?} | Status: FAILED" >> "${SESSION_LOG_FILE:-pipeline_sessions.log}"
    fi
    
    exit 1
}

trap cleanup_on_error ERR

# Parse arguments
if [ $# -lt 1 ]; then
    error "Usage: $0 <batch_size> [--background|-b] [--cpu|-c] [--batch-size <workers>]"
    echo ""
    echo "Examples:"
    echo "  $0 500                  # Run interactively with batch size 500"
    echo "  $0 500 --background     # Run in background with nohup"
    echo "  $0 500 --cpu            # CPU mode (no GPU)"
    exit 1
fi

BATCH_SIZE=${1:-500}  # Default to 500 if not specified

# Validate arguments
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
log "Images-Only Pipeline"
log "=========================================="
info "Batch size: $BATCH_SIZE documents per batch"
info "Parallel workers: $WORKERS"
info "Mode: $([ "$CPU_MODE" = true ] && echo "CPU" || echo "GPU")"
info "Working directory: $(pwd)"
log "=========================================="
echo ""

# Main loop - keep processing batches until no more documents are found
BATCH_NUM=1
TOTAL_PROCESSED=0
SESSION_LOG_FILE="pipeline_images_only_sessions.log"

while true; do
    log "=========================================="
    log "BATCH $BATCH_NUM"
    log "=========================================="
    
    # Step 1: Fetch batch of documents missing DOCLING_IMG
    log "STEP 1: Fetching batch of $BATCH_SIZE documents missing DOCLING_IMG"
    python3 1_fetch_documents.py \
        --min-doc-id 0 \
        --max-doc-id 99999999 \
        --run-all-images \
        --limit "$BATCH_SIZE"
    
    if [ $? -ne 0 ]; then
        error "Document fetching failed"
        exit 1
    fi
    
    # Check if any PDFs were downloaded
    PDFS_IN_BATCH=$(find data/to_process -name "doc_*.pdf" 2>/dev/null | wc -l | tr -d ' ')
    
    if [ "$PDFS_IN_BATCH" -eq 0 ]; then
        log "✓ No more documents to process! Pipeline complete."
        break
    fi
    
    info "Downloaded $PDFS_IN_BATCH PDFs in this batch"
    echo ""
    
    # Step 2: Extract images from PDFs (skip markdown conversion since we already have DOCLING)
    log "STEP 2: Extracting images from PDFs"
    
    GPU_FLAG=""
    [ "$CPU_MODE" = true ] && GPU_FLAG="--no-gpu"
    
    # We need to extract images from the PDFs we just downloaded
    # Use batch_convert with --extract-images-only flag (if it exists) or full conversion
    python3 2_batch_convert_pdfs.py \
        data/to_process \
        data/processed \
        --batch-size "$WORKERS" \
        --max-docs "$PDFS_IN_BATCH" \
        --extract-images \
        $GPU_FLAG
    
    if [ $? -ne 0 ]; then
        error "Image extraction failed for batch $BATCH_NUM"
        exit 1
    fi
    
    log "✓ Image extraction complete for batch $BATCH_NUM"
    echo ""
    
    # Step 3: Image Description Pipeline
    log "STEP 3: Running image description pipeline (batch $BATCH_NUM)"
    
    # Generate unique session ID for this pipeline run
    PIPELINE_SESSION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4())[:8])")
    log "  🔑 Pipeline Session ID: $PIPELINE_SESSION_ID"
    
    # Log session info to file
    SESSION_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$SESSION_START_TIME | Session: $PIPELINE_SESSION_ID | Batch: $BATCH_NUM | Status: STARTED" >> "$SESSION_LOG_FILE"
    log "  📝 Session logged to: $SESSION_LOG_FILE"
    
    # 3a: Prepare image batches
    log "  3a: Preparing image batches..."
    python3 3a_prepare_image_batches.py --session-id "$PIPELINE_SESSION_ID"
    
    # 3b: Upload batches to Gemini
    log "  3b: Uploading batches to Gemini..."
    python3 3b_upload_batches.py --session-id "$PIPELINE_SESSION_ID"
    
    # 3c: Monitor batch progress (with retry)
    log "  3c: Monitoring batch progress..."
    MAX_RETRIES=60  # 60 retries = 60 minutes max wait
    RETRY_COUNT=0
    WAIT_TIME=120  # 120 seconds between checks
    
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
    python3 3d_download_batch_results.py --session-id "$PIPELINE_SESSION_ID"
    
    log "✓ Image description pipeline complete for batch $BATCH_NUM"
    echo ""
    
    # Step 4: Integrate descriptions
    log "STEP 4: Integrating image descriptions (batch $BATCH_NUM)"
    python3 5_integrate_descriptions.py
    
    if [ $? -ne 0 ]; then
        error "Description integration failed for batch $BATCH_NUM"
        exit 1
    fi
    
    log "✓ Description integration complete for batch $BATCH_NUM"
    echo ""
    
    # Log session completion
    SESSION_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$SESSION_END_TIME | Session: $PIPELINE_SESSION_ID | Batch: $BATCH_NUM | Status: COMPLETED" >> "$SESSION_LOG_FILE"
    
    # Step 5: Upload DOCLING_IMG to S3 and Supabase
    log "STEP 5: Uploading DOCLING_IMG to S3 and Supabase (batch $BATCH_NUM)"
    python3 5a_upload.py
    
    if [ $? -ne 0 ]; then
        error "Upload failed for batch $BATCH_NUM"
        exit 1
    fi
    
    log "✓ Upload complete for batch $BATCH_NUM"
    echo ""
    
    # Step 6: Cleanup
    log "STEP 6: Cleaning up batch $BATCH_NUM"
    
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
    # Only delete PDFs that have corresponding markdown files in data/processed_images
    log "  Removing processed PDFs..."
    DELETED_COUNT=0
    for pdf in data/to_process/doc_*.pdf; do
        if [ -f "$pdf" ]; then
            # Extract document ID
            DOC_ID=$(basename "$pdf" .pdf)
            # Check if corresponding markdown exists in processed_images
            if [ -f "data/processed_images/${DOC_ID}.md" ]; then
                rm -f "$pdf"
                DELETED_COUNT=$((DELETED_COUNT + 1))
            fi
        fi
    done
    log "  ✓ Removed $DELETED_COUNT processed PDFs"
    
    log "✓ Cleanup complete for batch $BATCH_NUM"
    echo ""
    
    # Update counters
    TOTAL_PROCESSED=$((TOTAL_PROCESSED + PDFS_IN_BATCH))
    BATCH_NUM=$((BATCH_NUM + 1))
    
    # Brief pause between batches
    log "Pausing 10 seconds before next batch..."
    sleep 10
    echo ""
done

# Final summary
log "=========================================="
log "IMAGES-ONLY PIPELINE COMPLETE"
log "=========================================="
info "Total batches processed: $((BATCH_NUM - 1))"
info "Total documents processed: $TOTAL_PROCESSED"
log "=========================================="
echo ""

log "✓ All done! All documents now have DOCLING_IMG representations."

