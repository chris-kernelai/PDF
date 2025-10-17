#!/bin/bash
################################################################################
# Instance Setup Script (Run this ON the GCP instance after SSH)
#
# This script sets up the PDF processing environment on the GPU instance.
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setting Up PDF Processing Environment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Install NVIDIA drivers if not present
echo -e "${BLUE}Checking GPU drivers...${NC}"
if ! nvidia-smi &>/dev/null; then
    echo -e "${YELLOW}Installing NVIDIA drivers...${NC}"
    echo "This may take 5-10 minutes..."

    # Install CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -q
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update -qq

    # Install NVIDIA driver and CUDA toolkit
    sudo apt-get install -y -qq nvidia-driver-535 cuda-toolkit-12-2

    echo -e "${YELLOW}Driver installed. Waiting for GPU to initialize...${NC}"
    sleep 10

    # Verify again
    if ! nvidia-smi &>/dev/null; then
        echo -e "${RED}❌ GPU still not detected. May need reboot.${NC}"
        echo "Run: sudo reboot"
        echo "Then SSH back in and run this script again"
        exit 1
    fi
fi

echo -e "${GREEN}✅ GPU detected${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Update system
echo -e "${BLUE}Updating system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip python3-venv wget curl

# Check if repo already exists
if [ -d "PDF" ]; then
    echo -e "${YELLOW}⚠️  PDF directory already exists${NC}"
    read -p "Pull latest changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd PDF
        git pull
        cd ..
    fi
else
    # Get repo URL
    echo ""
    echo -e "${YELLOW}Enter your Git repository URL:${NC}"
    echo "  (e.g., https://github.com/username/PDF.git)"
    echo "  Or press Enter to skip and set up manually"
    read -p "Repo URL: " REPO_URL

    if [ ! -z "$REPO_URL" ]; then
        echo -e "${BLUE}Cloning repository...${NC}"
        git clone $REPO_URL PDF
    else
        echo -e "${YELLOW}⚠️  Skipping git clone - set up manually${NC}"
        mkdir -p PDF
    fi
fi

cd PDF

# Create virtual environment
echo ""
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip -q

# Install requirements
if [ -f "requirements.txt" ]; then
    echo -e "${BLUE}Installing requirements...${NC}"
    pip install -r requirements.txt -q
fi

# Install additional required packages
echo -e "${BLUE}Installing docling and dependencies...${NC}"
pip install docling google-generativeai python-dotenv -q

# Create directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p to_process processed pdfs_processed images processed_raw

# Set up environment variables
echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}API Configuration${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

if [ ! -f ".env" ]; then
    echo "Enter your Gemini API key (or press Enter to skip):"
    read -s GEMINI_KEY

    if [ ! -z "$GEMINI_KEY" ]; then
        echo "GOOGLE_API_KEY=$GEMINI_KEY" > .env
        echo -e "${GREEN}✅ API key saved to .env${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipped - you can add it later to .env${NC}"
    fi
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Virtual environment activated: $(which python)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Upload PDFs to process:"
echo -e "   From your local machine:"
echo -e "   ${BLUE}gcloud compute scp /path/to/pdfs/* $HOSTNAME:~/PDF/to_process/ --zone=\$(gcloud config get-value compute/zone)${NC}"
echo ""
echo "2. Run the pipeline:"
echo -e "   ${BLUE}./run_pipeline.sh${NC}"
echo ""
echo "3. Download results:"
echo -e "   ${BLUE}gcloud compute scp --recurse $HOSTNAME:~/PDF/processed /local/path/ --zone=\$(gcloud config get-value compute/zone)${NC}"
echo ""
echo "4. Stop instance when done:"
echo -e "   ${BLUE}sudo shutdown -h now${NC}"
echo ""
