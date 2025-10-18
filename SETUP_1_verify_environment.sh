#!/bin/bash

################################################################################
# SETUP_1_verify_environment.sh
#
# Run this ON AWS instance to verify environment is ready
#
# This checks:
#   - Virtual environment exists and has packages
#   - Required directories exist
#   - GPU is available
#   - Environment variables are set
#   - GCP authentication works
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "AWS Environment Verification"
echo "=========================================="
echo ""

ERRORS=0

# Check 1: Virtual environment
echo -n "Checking virtual environment... "
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ venv directory not found${NC}"
    echo "  Run: python3 -m venv venv"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Activate venv and check packages
if [ -d "venv" ]; then
    source venv/bin/activate

    echo -n "Checking Python packages... "
    if python3 -c "import docling, boto3, psycopg2, google.cloud.aiplatform" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Missing packages${NC}"
        echo "  Run: pip install -r requirements.txt"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check 3: Required directories
echo -n "Checking directories... "
REQUIRED_DIRS="data/to_process data/processed data/processed_raw data/processed_images .generated"
MISSING_DIRS=""
for dir in $REQUIRED_DIRS; do
    if [ ! -d "$dir" ]; then
        MISSING_DIRS="$MISSING_DIRS $dir"
    fi
done

if [ -z "$MISSING_DIRS" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Missing:$MISSING_DIRS${NC}"
    echo "  Run: mkdir -p $REQUIRED_DIRS"
    ERRORS=$((ERRORS + 1))
fi

# Check 4: GPU
echo -n "Checking GPU... "
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
        echo -e "${GREEN}✓ $GPU_NAME${NC}"
    else
        echo -e "${YELLOW}⚠ nvidia-smi failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ nvidia-smi not found${NC}"
fi

# Check 5: Environment file
echo -n "Checking .env file... "
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC}"

    # Check required variables
    source .env
    REQUIRED_VARS="AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY DB_HOST DB_USER GCP_PROJECT_ID LIBRARIAN_API_KEY"
    MISSING_VARS=""
    for var in $REQUIRED_VARS; do
        if [ -z "${!var}" ]; then
            MISSING_VARS="$MISSING_VARS $var"
        fi
    done

    if [ -n "$MISSING_VARS" ]; then
        echo -e "${RED}  ✗ Missing variables:$MISSING_VARS${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗ .env not found${NC}"
    echo "  Deploy with: ./LOCAL_2_aws_helper.sh deploy-config"
    ERRORS=$((ERRORS + 1))
fi

# Check 6: GCP service account key
echo -n "Checking GCP authentication... "
if [ -f "gcp-service-account-key.json" ]; then
    echo -e "${GREEN}✓${NC}"
elif gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
    echo -e "${GREEN}✓ (gcloud)${NC}"
else
    echo -e "${RED}✗ No GCP authentication found${NC}"
    echo "  Deploy with: ./LOCAL_2_aws_helper.sh deploy-config"
    ERRORS=$((ERRORS + 1))
fi

# Check 7: Config file
echo -n "Checking config.yaml... "
if [ -f "config.yaml" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ config.yaml not found${NC}"
fi

# Check 8: Pipeline scripts
echo -n "Checking pipeline scripts... "
REQUIRED_SCRIPTS="1_fetch_documents.py 2_batch_convert_pdfs.py 3a_prepare_image_batches.py 5_integrate_descriptions.py run_pipeline.sh"
MISSING_SCRIPTS=""
for script in $REQUIRED_SCRIPTS; do
    if [ ! -f "$script" ]; then
        MISSING_SCRIPTS="$MISSING_SCRIPTS $script"
    fi
done

if [ -z "$MISSING_SCRIPTS" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Missing:$MISSING_SCRIPTS${NC}"
    echo "  Deploy with: ./LOCAL_2_aws_helper.sh sync-code"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Ready to run pipeline:"
    echo "  ./run_pipeline.sh <min_id> <max_id> [batch_size]"
    echo ""
    echo "Example:"
    echo "  ./run_pipeline.sh 30000 35000 500"
else
    echo -e "${RED}✗ Found $ERRORS issue(s)${NC}"
    echo ""
    echo "Fix issues above, then try again"
fi
echo "=========================================="
echo ""
