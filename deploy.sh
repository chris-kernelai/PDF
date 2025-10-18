#!/bin/bash

################################################################################
# Automated Deployment Script for PDF Processing Pipeline
#
# This script automates the deployment of the pipeline to a new machine
################################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}$1${NC}"; }
error() { echo -e "${RED}ERROR: $1${NC}" >&2; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}"; }
info() { echo -e "${BLUE}$1${NC}"; }

echo ""
log "=========================================="
log "PDF Pipeline Deployment"
log "=========================================="
echo ""

# Check if running on correct directory
if [ ! -f "run_pipeline.sh" ]; then
    error "Must run from the PDF pipeline directory"
    exit 1
fi

# Step 1: Check Python
log "Step 1: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found. Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
info "Found Python $PYTHON_VERSION"

# Step 2: Create virtual environment
log "Step 2: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log "✓ Virtual environment created"
else
    warn "Virtual environment already exists, skipping"
fi

# Activate venv
source venv/bin/activate

# Step 3: Install dependencies
log "Step 3: Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
log "✓ Dependencies installed"

# Step 4: Check for .env file
log "Step 4: Checking configuration files..."
if [ ! -f ".env" ]; then
    warn ".env file not found"
    info "Please create .env file with required environment variables"
    info "See .env.example for template"
    exit 1
fi
log "✓ .env file found"

# Step 5: Check for GCP service account key
if grep -q "GOOGLE_APPLICATION_CREDENTIALS" .env; then
    KEY_FILE=$(grep "GOOGLE_APPLICATION_CREDENTIALS" .env | cut -d'=' -f2)
    if [ ! -f "$KEY_FILE" ]; then
        warn "GCP service account key file not found at: $KEY_FILE"
        info "Please copy your gcp-service-account-key.json file"
        info "Or run: ./setup_gcp_auth.sh"
    else
        log "✓ GCP service account key found"
    fi
fi

# Step 6: Create required directories
log "Step 5: Creating required directories..."
mkdir -p data/{to_process,processed,processed_raw,processed_images,images}
mkdir -p .generated
log "✓ Directories created"

# Step 7: Make scripts executable
log "Step 6: Making scripts executable..."
chmod +x run_pipeline.sh setup_gcp_auth.sh 2>/dev/null || true
chmod +x *.py 2>/dev/null || true
log "✓ Scripts made executable"

# Step 8: Test imports
log "Step 7: Testing Python imports..."
python3 << 'EOF'
try:
    import asyncio
    import asyncpg
    import aioboto3
    import aiohttp
    from google.cloud import storage
    print("✓ All required modules imported successfully")
except ImportError as e:
    print(f"❌ Missing module: {e}")
    exit(1)
EOF

# Step 9: Test authentication
log "Step 8: Testing authentication..."

# Test GCP
if [ -f "$KEY_FILE" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"
    if python3 -c "from google.cloud import storage; storage.Client()" 2>/dev/null; then
        log "✓ GCP authentication successful"
    else
        warn "GCP authentication test failed"
    fi
fi

# Test AWS (if configured)
if command -v aws &> /dev/null; then
    if aws sts get-caller-identity &>/dev/null; then
        log "✓ AWS authentication successful"
    else
        warn "AWS authentication not configured or failed"
    fi
fi

echo ""
log "=========================================="
log "✓ Deployment Complete!"
log "=========================================="
echo ""

info "Next steps:"
echo "1. Verify .env configuration: cat .env"
echo "2. Test with small batch: ./run_pipeline.sh 27000 27010 10"
echo "3. Run production: ./run_pipeline.sh <min_id> <max_id> 500"
echo ""

info "For background execution:"
echo "  nohup ./run_pipeline.sh 30000 35000 500 > pipeline.log 2>&1 &"
echo ""

log "=========================================="
