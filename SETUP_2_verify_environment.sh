#!/bin/bash

################################################################################
# SETUP_2_verify_environment.sh
#
# STEP 2: Run this ON AWS instance to verify environment is ready
#
# Run this AFTER SETUP_1_python_venv.sh to check everything is working
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
if [ -d "workspace/venv" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ workspace/venv directory not found${NC}"
    echo "  Run: ./SETUP_1_python_venv.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Activate venv and check packages
if [ -d "workspace/venv" ]; then
    source workspace/venv/bin/activate

    echo -n "Checking Python packages... "
    MISSING_PACKAGES=""

    # Check each package individually
    python3 -c "import docling" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES docling"
    python3 -c "import psycopg2" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES psycopg2-binary"
    python3 -c "import google.cloud.aiplatform" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES google-cloud-aiplatform"
    python3 -c "import google.genai" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES google-generativeai"
    python3 -c "import aiohttp" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES aiohttp"
    python3 -c "import aiofiles" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES aiofiles"
    python3 -c "import dotenv" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES python-dotenv"
    python3 -c "import yaml" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES pyyaml"
    python3 -c "import PIL" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES Pillow"
    python3 -c "import boto3" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES boto3"
    python3 -c "import aioboto3" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES aioboto3"
    python3 -c "import asyncpg" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES asyncpg"
    python3 -c "import httpx" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES httpx"

    if [ -z "$MISSING_PACKAGES" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Missing:$MISSING_PACKAGES${NC}"
        echo "  Run: pip install -r workspace/configs/requirements.txt"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check 3: Required directories
echo -n "Checking directories... "
REQUIRED_DIRS="workspace/data/to_process workspace/data/processed workspace/data/processed_raw workspace/data/processed_images workspace/.generated"
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

    # Check required variables (checking actual variable names used in codebase)
    source .env
    REQUIRED_VARS="DB_HOST DB_USER GCP_PROJECT API_KEY"
    MISSING_VARS=""
    for var in $REQUIRED_VARS; do
        if [ -z "${!var}" ]; then
            MISSING_VARS="$MISSING_VARS $var"
        fi
    done

    # Check AWS credentials: either environment vars OR IAM role
    if [ -z "$AWS_PROFILE" ] && [ -z "$AWS_ACCESS_KEY_ID" ]; then
        # Check if instance has IAM role (EC2 metadata service)
        if curl -s --connect-timeout 1 http://169.254.169.254/latest/meta-data/iam/security-credentials/ &> /dev/null; then
            echo -e "${GREEN}  ✓ Using IAM role for AWS authentication${NC}"
        else
            MISSING_VARS="$MISSING_VARS AWS_PROFILE/AWS_ACCESS_KEY_ID"
        fi
    fi

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
if [ -f "workspace/configs/config.yaml" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ workspace/configs/config.yaml not found${NC}"
fi

# Check 8: Pipeline scripts
echo -n "Checking pipeline scripts... "
REQUIRED_SCRIPTS="run_pipeline.py run_pipeline.sh run_pipeline_md_only.sh run_pipeline_images_only.sh"
MISSING_SCRIPTS=""
for script in $REQUIRED_SCRIPTS; do
    if [ ! -f "$script" ]; then
        MISSING_SCRIPTS="$MISSING_SCRIPTS $script"
    fi
done

# Also check workspace structure
if [ ! -d "workspace/src" ] || [ ! -d "workspace/scripts" ]; then
    MISSING_SCRIPTS="$MISSING_SCRIPTS workspace/src workspace/scripts"
fi

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
