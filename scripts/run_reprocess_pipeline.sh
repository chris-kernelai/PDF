#!/bin/bash
################################################################################
# run_reprocess_pipeline.sh
#
# Reprocess compromised documents through image description and filter pipelines
# This script processes documents from data/processed/ and uploads only docling_img
# files to replace the potentially compromised ones.
#
# Usage:
#   ./scripts/run_reprocess_pipeline.sh [session-id]
#
# If no session-id is provided, one will be generated automatically.
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Check if session ID provided as argument
if [ $# -eq 1 ]; then
    PIPELINE_SESSION_ID="$1"
    log "Using provided session ID: $PIPELINE_SESSION_ID"
else
    # Generate unique session ID for this reprocessing run
    PIPELINE_SESSION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4())[:8])")
    log "Generated session ID: $PIPELINE_SESSION_ID"
fi

# Log session info to file
SESSION_LOG_FILE="reprocess_sessions.log"
SESSION_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "$SESSION_START_TIME | Session: $PIPELINE_SESSION_ID | Type: REPROCESS | Status: STARTED" >> "$SESSION_LOG_FILE"
log "📝 Session logged to: $SESSION_LOG_FILE"

# Check if we have documents to process
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed 2>/dev/null)" ]; then
    error "No documents found in data/processed/"
    error "Please run the download script first: python3 scripts/download_compromised_docling.py document_locations_v2_rows.csv --profile production"
    exit 1
fi

DOC_COUNT=$(find data/processed -name "doc_*.txt" | wc -l)
log "Found $DOC_COUNT documents to reprocess"

# Function to cleanup on error
cleanup_on_error() {
    error "Reprocessing pipeline failed. Cleaning up..."
    
    # Log session failure
    SESSION_ERROR_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$SESSION_ERROR_TIME | Session: $PIPELINE_SESSION_ID | Type: REPROCESS | Status: FAILED" >> "$SESSION_LOG_FILE"
    
    exit 1
}

# Set trap for error handling
trap cleanup_on_error ERR

log "🚀 Starting reprocessing pipeline for session: $PIPELINE_SESSION_ID"
echo ""

# Step 1: Convert docling files to images (if needed)
log "STEP 1: Preparing images from docling files"
if [ ! -d "data/images" ]; then
    log "  Creating images directory..."
    mkdir -p data/images
fi

# Check if images already exist
if [ -z "$(ls -A data/images 2>/dev/null)" ]; then
    log "  Converting docling files to images..."
    python3 2_batch_convert_pdfs.py --input-dir data/processed --output-dir data/images
    success "Image conversion complete"
else
    log "  Images already exist, skipping conversion"
fi
echo ""

# Step 2: Image Description Pipeline
log "STEP 2: Running image description pipeline"

# 2a: Prepare image batches
log "  2a: Preparing image batches..."
python3 3a_prepare_image_batches.py --session-id "$PIPELINE_SESSION_ID"

# 2b: Upload batches to Gemini
log "  2b: Uploading batches to Gemini..."
python3 3b_upload_batches.py --session-id "$PIPELINE_SESSION_ID"

# 2c: Monitor progress (with retry)
log "  2c: Monitoring batch progress..."
MAX_RETRIES=60  # 60 retries = 60 minutes max wait
RETRY_COUNT=0
WAIT_TIME=60  # 60 seconds between checks

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    MONITOR_OUTPUT=$(python3 3c_monitor_batches.py --session-id "$PIPELINE_SESSION_ID" 2>&1)
    echo "$MONITOR_OUTPUT"
    
    if echo "$MONITOR_OUTPUT" | grep -q "All batches completed"; then
        success "All image description batches completed"
        break
    elif echo "$MONITOR_OUTPUT" | grep -q "Some batches failed"; then
        error "Some image description batches failed"
        exit 1
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        log "  Waiting $WAIT_TIME seconds before next check... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep $WAIT_TIME
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    error "Timeout waiting for image description batches to complete"
    exit 1
fi

# 2d: Download results
log "  2d: Downloading batch results..."
python3 3d_download_batch_results.py --session-id "$PIPELINE_SESSION_ID"

success "Image description pipeline complete"
echo ""

# Step 3: Filter Pipeline
log "STEP 3: Running filter pipeline"

# 3a: Prepare filter batches
log "  3a: Preparing filter batches..."
python3 4a_prepare_filter_batches.py --session-id "$PIPELINE_SESSION_ID"

# 3b: Upload filter batches
log "  3b: Uploading filter batches..."
python3 4b_upload_filter_batches.py --session-id "$PIPELINE_SESSION_ID"

# 3c: Monitor filter progress (with retry)
log "  3c: Monitoring filter progress..."
MAX_RETRIES=60  # 60 retries = 60 minutes max wait
RETRY_COUNT=0
WAIT_TIME=60  # 60 seconds between checks

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    MONITOR_OUTPUT=$(python3 4c_monitor_filter_batches.py --session-id "$PIPELINE_SESSION_ID" 2>&1)
    echo "$MONITOR_OUTPUT"
    
    if echo "$MONITOR_OUTPUT" | grep -q "All batches completed"; then
        success "All filter batches completed"
        break
    elif echo "$MONITOR_OUTPUT" | grep -q "Some batches failed"; then
        error "Some filter batches failed"
        exit 1
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        log "  Waiting $WAIT_TIME seconds before next check... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep $WAIT_TIME
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    error "Timeout waiting for filter batches to complete"
    exit 1
fi

# 3d: Download filter results
log "  3d: Downloading filter results..."
python3 4d_download_filter_results.py --session-id "$PIPELINE_SESSION_ID"

success "Filter pipeline complete"
echo ""

# Step 4: Integrate descriptions
log "STEP 4: Integrating image descriptions"
python3 5_integrate_descriptions.py
success "Description integration complete"
echo ""

# Step 5: Upload only docling_img files
log "STEP 5: Uploading docling_img files to replace compromised ones"
python3 scripts/upload_docling_img_only.py --session-id "$PIPELINE_SESSION_ID" --profile production
success "Docling_img upload complete"
echo ""

# Log session completion
SESSION_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "$SESSION_END_TIME | Session: $PIPELINE_SESSION_ID | Type: REPROCESS | Status: COMPLETED" >> "$SESSION_LOG_FILE"

success "🎉 Reprocessing pipeline complete for session: $PIPELINE_SESSION_ID"
log "📝 Session completion logged to: $SESSION_LOG_FILE"
