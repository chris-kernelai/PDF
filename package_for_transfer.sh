#!/bin/bash

################################################################################
# Package Pipeline for Transfer to Another Machine
#
# Creates a tarball with all necessary files for deployment
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}$1${NC}"; }
info() { echo -e "${BLUE}$1${NC}"; }

PACKAGE_NAME="pdf-pipeline-$(date +%Y%m%d_%H%M%S).tar.gz"

echo ""
log "=========================================="
log "Packaging PDF Pipeline for Transfer"
log "=========================================="
echo ""

# Check required files exist
log "Checking required files..."
if [ ! -f ".env" ]; then
    echo "❌ .env not found"
    exit 1
fi

if [ ! -f "~/gcp-service-account-key.json" ]; then
    echo "⚠️  GCP service account key not found at ~/gcp-service-account-key.json"
    echo "   Pipeline will require manual GCP authentication"
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cp -r . "$TEMP_DIR/PDF"
cd "$TEMP_DIR/PDF"

# Copy GCP key if it exists
if [ -f ~/gcp-service-account-key.json ]; then
    cp ~/gcp-service-account-key.json .
    log "✓ Included GCP service account key"
fi

# Remove unnecessary files
log "Cleaning up unnecessary files..."
rm -rf venv __pycache__ .git
rm -rf data/to_process/*.pdf
rm -rf data/images/*
rm -rf .generated
rm -rf *.log
rm -f processing_log.csv processed_documents.txt

# Create archive
cd ..
tar czf "$PACKAGE_NAME" PDF

# Move to original directory
mv "$PACKAGE_NAME" "$OLDPWD/"
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

FILE_SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)

echo ""
log "=========================================="
log "✓ Package Created Successfully"
log "=========================================="
info "File: $PACKAGE_NAME"
info "Size: $FILE_SIZE"
echo ""

log "Transfer to target machine:"
echo "  scp $PACKAGE_NAME user@target-machine:~/"
echo ""

log "On target machine, run:"
echo "  tar xzf $PACKAGE_NAME"
echo "  cd PDF"
echo "  ./deploy.sh"
echo ""

log "=========================================="
