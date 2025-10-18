#!/bin/bash
################################################################################
# setup_aws_gpu.sh
#
# Setup script to deploy PDF processing pipeline to AWS GPU instance
#
# Usage:
#   ./setup_aws_gpu.sh
#
################################################################################

set -e

# AWS Instance details
INSTANCE_IP="52.53.182.181"
INSTANCE_USER="ubuntu"  # Default for Ubuntu AMI, change to "ec2-user" for Amazon Linux
PEM_KEY="PDF_key.pem"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AWS GPU Instance Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Instance: $INSTANCE_IP"
echo "User: $INSTANCE_USER"
echo ""

# Check PEM file exists and has correct permissions
if [ ! -f "$PEM_KEY" ]; then
    echo "❌ PEM key file not found: $PEM_KEY"
    exit 1
fi

# Fix PEM permissions if needed
chmod 400 "$PEM_KEY"
echo -e "${GREEN}✅ PEM key permissions set${NC}"

# Test SSH connection
echo ""
echo -e "${BLUE}Testing SSH connection...${NC}"
if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "echo 'Connection successful'" 2>/dev/null; then
    echo "❌ Cannot connect to instance"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check if instance is running"
    echo "  2. Check security group allows SSH (port 22) from your IP"
    echo "  3. Try different user: 'ubuntu' or 'ec2-user'"
    echo ""
    echo "To manually test connection:"
    echo "  ssh -i $PEM_KEY ${INSTANCE_USER}@${INSTANCE_IP}"
    exit 1
fi

echo -e "${GREEN}✅ SSH connection successful${NC}"
echo ""

# Create archive of repository (excluding venv and large files)
echo -e "${BLUE}Creating repository archive...${NC}"
ARCHIVE_NAME="pdf_pipeline.tar.gz"

tar -czf "$ARCHIVE_NAME" \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.generated' \
    --exclude='processed' \
    --exclude='processed_raw' \
    --exclude='processed_images' \
    --exclude='images' \
    --exclude='pdfs_processed' \
    --exclude='to_process/*.pdf' \
    --exclude="$ARCHIVE_NAME" \
    .

echo -e "${GREEN}✅ Archive created: $ARCHIVE_NAME${NC}"
echo ""

# Copy archive to instance
echo -e "${BLUE}Copying repository to instance...${NC}"
scp -i "$PEM_KEY" "$ARCHIVE_NAME" "${INSTANCE_USER}@${INSTANCE_IP}:~/"
echo -e "${GREEN}✅ Repository copied${NC}"
echo ""

# Extract and setup on instance
echo -e "${BLUE}Setting up on instance...${NC}"
ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" bash << 'ENDSSH'
set -e

echo "Extracting repository..."
mkdir -p ~/pdf_pipeline
cd ~/pdf_pipeline
tar -xzf ~/pdf_pipeline.tar.gz
rm ~/pdf_pipeline.tar.gz

echo "Updating system packages..."
sudo apt-get update -qq

echo "Installing system dependencies..."
sudo apt-get install -y python3-pip python3-venv git curl

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt || echo "Warning: Some dependencies may have failed"

echo "Creating necessary directories..."
mkdir -p to_process processed processed_raw processed_images images pdfs_processed .generated

echo "Setup complete!"
echo ""
echo "Repository location: ~/pdf_pipeline"
ENDSSH

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "To connect to instance:"
echo -e "${YELLOW}  ssh -i $PEM_KEY ${INSTANCE_USER}@${INSTANCE_IP}${NC}"
echo ""
echo "To activate environment on instance:"
echo -e "${YELLOW}  cd ~/pdf_pipeline${NC}"
echo -e "${YELLOW}  source venv/bin/activate${NC}"
echo ""
echo "To upload PDFs to instance:"
echo -e "${YELLOW}  scp -i $PEM_KEY your_file.pdf ${INSTANCE_USER}@${INSTANCE_IP}:~/pdf_pipeline/to_process/${NC}"
echo ""
echo "To run pipeline on instance:"
echo -e "${YELLOW}  ssh -i $PEM_KEY ${INSTANCE_USER}@${INSTANCE_IP}${NC}"
echo -e "${YELLOW}  cd ~/pdf_pipeline && source venv/bin/activate${NC}"
echo -e "${YELLOW}  ./run_pipeline.sh${NC}"
echo ""
echo "To download results from instance:"
echo -e "${YELLOW}  scp -i $PEM_KEY -r ${INSTANCE_USER}@${INSTANCE_IP}:~/pdf_pipeline/processed_images/ .${NC}"
echo ""

# Clean up local archive
rm "$ARCHIVE_NAME"
