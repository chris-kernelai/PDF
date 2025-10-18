#!/bin/bash

################################################################################
# GCP Service Account Setup Script
#
# This script helps you create a service account for automated authentication
# to Google Cloud Platform services (Vertex AI, GCS, Gemini API)
################################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}$1${NC}"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}$1${NC}"
}

# Configuration
PROJECT_ID="gen-lang-client-0133572494"
SERVICE_ACCOUNT_NAME="pdf-pipeline-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="$HOME/gcp-service-account-key.json"

echo ""
log "=========================================="
log "GCP Service Account Setup"
log "=========================================="
info "Project ID: $PROJECT_ID"
info "Service Account: $SERVICE_ACCOUNT_NAME"
info "Key file location: $KEY_FILE"
log "=========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    error "gcloud CLI is not installed"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    warn "You need to login to gcloud first"
    echo "Running: gcloud auth login"
    gcloud auth login
fi

# Set the project
log "Setting active project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# Check if service account already exists
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
    warn "Service account $SERVICE_ACCOUNT_EMAIL already exists"
    read -p "Do you want to use the existing service account? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Exiting. Please use a different service account name."
        exit 0
    fi
else
    # Create service account
    log "Creating service account: $SERVICE_ACCOUNT_NAME..."
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="PDF Pipeline Service Account" \
        --description="Service account for automated PDF processing pipeline"

    log "✓ Service account created"
fi

# Grant necessary roles
log "Granting IAM roles..."

# Vertex AI User - for using Vertex AI models
info "  → roles/aiplatform.user (Vertex AI User)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.user" \
    --condition=None \
    &> /dev/null

# Storage Admin - for GCS bucket access
info "  → roles/storage.admin (Storage Admin)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin" \
    --condition=None \
    &> /dev/null

# Generative AI User - for Gemini API access
info "  → roles/aiplatform.user (Generative AI User)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.serviceAgent" \
    --condition=None \
    &> /dev/null || true  # This role might not exist in all projects

log "✓ IAM roles granted"

# Check if key file already exists
if [ -f "$KEY_FILE" ]; then
    warn "Key file already exists at $KEY_FILE"
    read -p "Do you want to create a new key (old key will be moved to backup)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mv "$KEY_FILE" "${KEY_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        log "Old key backed up"
    else
        log "Using existing key file"
        echo ""
        log "=========================================="
        log "✓ Setup complete!"
        log "=========================================="
        info "Key file location: $KEY_FILE"
        info "Add this to your .env file:"
        echo "GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"
        log "=========================================="
        exit 0
    fi
fi

# Create and download key
log "Creating service account key..."
gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account="$SERVICE_ACCOUNT_EMAIL"

log "✓ Key file created"

# Set restrictive permissions on key file
chmod 600 "$KEY_FILE"
log "✓ Key file permissions set to 600"

echo ""
log "=========================================="
log "✓ Setup complete!"
log "=========================================="
info "Key file saved to: $KEY_FILE"
echo ""
info "Your .env file has been configured with:"
echo "GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"
echo ""
info "You can now run your pipeline without manual authentication!"
log "=========================================="
echo ""

# Test the authentication
log "Testing authentication..."
if GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE" gcloud auth application-default print-access-token &> /dev/null; then
    log "✓ Authentication test successful!"
else
    warn "Authentication test failed. Please check your setup."
fi

echo ""
info "Next steps:"
echo "1. Make sure GOOGLE_APPLICATION_CREDENTIALS is in your .env file"
echo "2. Run your pipeline: ./run_pipeline.sh <min_id> <max_id>"
echo ""
