#!/bin/bash
################################################################################
# GCP GPU Instance Setup Script
#
# This script creates a GPU instance on Google Cloud and sets it up for
# PDF processing with Docling.
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GCP GPU Instance Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration - EDIT THESE
INSTANCE_NAME="pdf-processor"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="200GB"
USE_PREEMPTIBLE=false  # Set to true for 70% cost savings (but can be shut down anytime)

echo "Configuration:"
echo "  Instance name: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: $GPU_TYPE x $GPU_COUNT"
echo "  Disk: $BOOT_DISK_SIZE"
echo "  Preemptible: $USE_PREEMPTIBLE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found!${NC}"
    echo "Install with: brew install --cask google-cloud-sdk"
    echo "Then run: gcloud init"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo -e "${RED}❌ Not authenticated with gcloud${NC}"
    echo "Run: gcloud init"
    exit 1
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No project set${NC}"
    echo "Run: gcloud init"
    exit 1
fi

echo -e "${GREEN}✅ Project: $PROJECT_ID${NC}"
echo ""

# Enable required APIs
echo -e "${BLUE}Enabling required APIs...${NC}"
gcloud services enable compute.googleapis.com --quiet

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo -e "${YELLOW}⚠️  Instance '$INSTANCE_NAME' already exists!${NC}"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    else
        echo "Exiting. Use a different INSTANCE_NAME or delete manually."
        exit 0
    fi
fi

# Build create command
CREATE_CMD="gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --boot-disk-type=pd-standard \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True"

if [ "$USE_PREEMPTIBLE" = true ]; then
    CREATE_CMD="$CREATE_CMD --preemptible"
fi

# Create instance
echo -e "${BLUE}Creating GPU instance...${NC}"
echo "This may take 2-3 minutes..."
echo ""

eval $CREATE_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Instance created successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to create instance${NC}"
    exit 1
fi

# Wait for instance to be ready
echo ""
echo -e "${BLUE}Waiting for instance to be ready...${NC}"
sleep 10

# Configure SSH
echo -e "${BLUE}Configuring SSH...${NC}"
gcloud compute config-ssh --quiet

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Instance Ready!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "SSH into instance:"
echo -e "  ${BLUE}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE${NC}"
echo ""
echo "Or use the setup script on the instance:"
echo -e "  ${BLUE}./gcp_instance_setup.sh${NC}"
echo ""
echo "View instance in console:"
echo "  https://console.cloud.google.com/compute/instances?project=$PROJECT_ID"
echo ""
echo -e "${YELLOW}💰 Cost estimate:${NC}"
if [ "$USE_PREEMPTIBLE" = true ]; then
    echo "  ~\$0.21/hour (preemptible)"
else
    echo "  ~\$0.73/hour (regular)"
fi
echo ""
echo -e "${YELLOW}Stop instance when not in use:${NC}"
echo -e "  ${BLUE}gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE${NC}"
echo ""
echo -e "${YELLOW}Delete instance (and disk):${NC}"
echo -e "  ${BLUE}gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE${NC}"
echo ""
