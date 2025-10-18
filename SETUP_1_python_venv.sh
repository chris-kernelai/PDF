#!/bin/bash

################################################################################
# SETUP_1_python_venv.sh
#
# STEP 1: Run ON AWS instance to create virtual environment and install packages
#
# This script:
#   - Creates Python virtual environment
#   - Installs all dependencies from requirements.txt
#   - Creates required directories
#
# Run this FIRST, then run SETUP_2_verify_environment.sh
################################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "Python Virtual Environment Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found${NC}"
    echo "Make sure you're in the pdf_pipeline directory"
    exit 1
fi

# Remove existing venv if requested
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf venv
    else
        echo "Keeping existing venv, will reinstall packages..."
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Using existing virtual environment"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip -q

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
echo "This may take several minutes..."
pip install -r requirements.txt

echo -e "${GREEN}✓${NC} All packages installed"

# Create required directories
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p data/{to_process,processed,processed_raw,processed_images,images}
mkdir -p .generated

echo -e "${GREEN}✓${NC} Directories created"

# Show Python version and key packages
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Python version:"
python --version
echo ""
echo "Key packages installed:"
pip show docling 2>/dev/null | grep "Name:\|Version:" || echo "  docling: not found"
pip show psycopg2-binary 2>/dev/null | grep "Name:\|Version:" || echo "  psycopg2-binary: not found"
pip show google-cloud-aiplatform 2>/dev/null | grep "Name:\|Version:" || echo "  google-cloud-aiplatform: not found"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To verify setup:"
echo "  ./SETUP_2_verify_environment.sh"
echo ""
echo "=========================================="
echo ""
