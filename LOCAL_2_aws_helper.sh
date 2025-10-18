#!/bin/bash
################################################################################
# LOCAL_2_aws_helper.sh
#
# AWS GPU instance operations - Run this FROM your LOCAL machine
#
# Usage:
#   ./LOCAL_2_aws_helper.sh setup                # Complete first-time setup
#   ./LOCAL_2_aws_helper.sh deploy-config        # Deploy .env and GCP key
#   ./LOCAL_2_aws_helper.sh sync-code            # Sync code changes to instance
#   ./LOCAL_2_aws_helper.sh connect              # SSH into instance
#   ./LOCAL_2_aws_helper.sh upload <file>        # Upload file to to_process/
#   ./LOCAL_2_aws_helper.sh upload-dir <dir>     # Upload directory of PDFs
#   ./LOCAL_2_aws_helper.sh run                  # Run pipeline on instance
#   ./LOCAL_2_aws_helper.sh download             # Download processed results
#   ./LOCAL_2_aws_helper.sh logs                 # View processing logs
#   ./LOCAL_2_aws_helper.sh status               # Check instance status
#   ./LOCAL_2_aws_helper.sh clean                # Remove processed files
#
################################################################################

# AWS Instance details
INSTANCE_IP="13.57.236.223"
INSTANCE_USER="ubuntu"
PEM_KEY="PDF_key.pem"
REMOTE_DIR="~/pdf_pipeline"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}❌ PEM key not found: $PEM_KEY${NC}"
    exit 1
fi

chmod 400 "$PEM_KEY"

case "$1" in
    connect)
        echo -e "${BLUE}Connecting to AWS instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source venv/bin/activate && bash"
        ;;

    upload)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify a file to upload${NC}"
            echo "Usage: $0 upload <file.pdf>"
            exit 1
        fi

        if [ ! -f "$2" ]; then
            echo -e "${RED}❌ File not found: $2${NC}"
            exit 1
        fi

        echo -e "${BLUE}Uploading $2 to instance...${NC}"
        scp -i "$PEM_KEY" "$2" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/to_process/"
        echo -e "${GREEN}✅ Upload complete${NC}"
        ;;

    upload-dir)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please specify a directory to upload${NC}"
            echo "Usage: $0 upload-dir <directory>"
            exit 1
        fi

        if [ ! -d "$2" ]; then
            echo -e "${RED}❌ Directory not found: $2${NC}"
            exit 1
        fi

        echo -e "${BLUE}Uploading PDFs from $2 to instance...${NC}"
        scp -i "$PEM_KEY" "$2"/*.pdf "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/to_process/" 2>/dev/null || {
            echo -e "${RED}❌ No PDF files found in $2${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ Upload complete${NC}"
        ;;

    download)
        echo -e "${BLUE}Downloading results from instance...${NC}"
        mkdir -p ./aws_results

        echo "Downloading processed_images/..."
        scp -i "$PEM_KEY" -r "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/processed_images/" ./aws_results/ 2>/dev/null || echo "No processed_images found"

        echo "Downloading processing_log.csv..."
        scp -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/processing_log.csv" ./aws_results/ 2>/dev/null || echo "No processing_log.csv found"

        echo "Downloading images/..."
        scp -i "$PEM_KEY" -r "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/images/" ./aws_results/ 2>/dev/null || echo "No images found"

        echo -e "${GREEN}✅ Results downloaded to ./aws_results/${NC}"
        ;;

    run)
        echo -e "${BLUE}Running pipeline on instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source venv/bin/activate && ./run_pipeline.sh"
        ;;

    run-md-only)
        echo -e "${BLUE}Running pipeline (MD only) on instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source venv/bin/activate && ./run_pipeline.sh --md-only"
        ;;

    logs)
        echo -e "${BLUE}Viewing processing logs...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source venv/bin/activate && python3 view_processing_log.py --summary"
        ;;

    sync-code)
        echo -e "${BLUE}Syncing code changes to instance...${NC}"

        # Sync Python scripts
        scp -i "$PEM_KEY" *.py "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"

        # Sync shell scripts
        scp -i "$PEM_KEY" *.sh "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"

        # Sync src directory if it exists
        if [ -d "src" ]; then
            scp -i "$PEM_KEY" -r src "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
        fi

        # Make scripts executable
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "cd $REMOTE_DIR && chmod +x *.sh"

        echo -e "${GREEN}✅ Code synced${NC}"
        ;;

    deploy-config)
        echo -e "${BLUE}Deploying configuration to instance...${NC}"

        # Check if .env exists
        if [ ! -f ".env" ]; then
            echo -e "${RED}❌ .env file not found${NC}"
            exit 1
        fi

        # Copy .env
        echo "Copying .env..."
        scp -i "$PEM_KEY" .env "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"

        # Copy GCP service account key if it exists
        GCP_KEY="$HOME/gcp-service-account-key.json"
        if [ -f "$GCP_KEY" ]; then
            echo "Copying GCP service account key..."
            scp -i "$PEM_KEY" "$GCP_KEY" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
        else
            echo -e "${YELLOW}⚠️  GCP service account key not found at $GCP_KEY${NC}"
            echo -e "${YELLOW}   Pipeline will require manual gcloud authentication${NC}"
        fi

        # Copy config.yaml if it exists
        if [ -f "config.yaml" ]; then
            echo "Copying config.yaml..."
            scp -i "$PEM_KEY" config.yaml "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
        fi

        # Copy requirements.txt if it exists
        if [ -f "requirements.txt" ]; then
            echo "Copying requirements.txt..."
            scp -i "$PEM_KEY" requirements.txt "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
        fi

        # Fix GOOGLE_APPLICATION_CREDENTIALS path in remote .env
        echo "Fixing GOOGLE_APPLICATION_CREDENTIALS path in remote .env..."
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline
if [ -f ".env" ] && [ -f "gcp-service-account-key.json" ]; then
    # Update the path to point to the remote location
    sed -i 's|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=~/pdf_pipeline/gcp-service-account-key.json|g' .env
    echo "✓ Updated GOOGLE_APPLICATION_CREDENTIALS path"
fi
ENDSSH

        # Set proper permissions on remote
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "cd $REMOTE_DIR && chmod 600 .env gcp-service-account-key.json 2>/dev/null || true"

        echo -e "${GREEN}✅ Configuration deployed${NC}"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "  1. Connect to instance: $0 connect"
        echo "  2. Install dependencies: pip install -r requirements.txt"
        echo "  3. Run pipeline: ./run_pipeline.sh <min_id> <max_id>"
        ;;

    setup)
        echo -e "${BLUE}Setting up AWS instance from scratch...${NC}"
        echo ""

        # First, create the directory on AWS
        echo -e "${BLUE}Step 1: Creating directory on AWS...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "mkdir -p ~/pdf_pipeline"
        echo -e "${GREEN}✅ Directory created${NC}"

        echo ""
        echo -e "${BLUE}Step 2: Deploying configuration...${NC}"
        "$0" deploy-config

        echo ""
        echo -e "${BLUE}Step 3: Syncing code...${NC}"
        "$0" sync-code

        echo ""
        echo -e "${BLUE}Step 4: Installing dependencies on instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create required directories
echo "Creating directories..."
mkdir -p data/{to_process,processed,processed_raw,processed_images,images}
mkdir -p .generated

echo "✅ Setup complete on remote instance"
ENDSSH

        echo ""
        echo -e "${GREEN}✅ Instance setup complete!${NC}"
        echo ""
        echo -e "${BLUE}Test the setup:${NC}"
        echo "  $0 connect"
        echo "  ./run_pipeline.sh 27000 27010 10"
        ;;

    download-code)
        echo -e "${BLUE}Downloading code from instance...${NC}"

        # Create backup directory with timestamp
        BACKUP_DIR="./code_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"

        echo "Backing up current local code to $BACKUP_DIR..."
        cp *.py "$BACKUP_DIR/" 2>/dev/null || true
        cp *.sh "$BACKUP_DIR/" 2>/dev/null || true

        # Download Python scripts
        echo "Downloading Python scripts..."
        scp -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/*.py" ./ 2>/dev/null || echo "No Python files to download"

        # Download shell scripts (excluding aws_helper.sh to avoid overwriting this script)
        echo "Downloading shell scripts..."
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "cd ${REMOTE_DIR} && ls *.sh 2>/dev/null | grep -v aws_helper.sh" | while read script; do
            echo "  Downloading $script..."
            scp -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/$script" ./ 2>/dev/null
        done

        # Make shell scripts executable
        chmod +x *.sh 2>/dev/null || true

        echo -e "${GREEN}✅ Code downloaded from AWS (excluding aws_helper.sh)${NC}"
        echo -e "${YELLOW}ℹ️  Local backup saved to: $BACKUP_DIR${NC}"
        ;;

    status)
        echo -e "${BLUE}Checking instance status...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline
echo "=========================================="
echo "PDFs to process:"
ls -1 to_process/*.pdf 2>/dev/null | wc -l
echo ""
echo "Processed markdown files:"
ls -1 processed/*.md 2>/dev/null | wc -l
echo ""
echo "Enhanced markdown files:"
ls -1 processed_images/*.md 2>/dev/null | wc -l
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info not available"
echo "=========================================="
ENDSSH
        ;;

    clean)
        echo -e "${YELLOW}⚠️  This will remove all processed files on the instance${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Cleaning instance...${NC}"
            ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline
rm -rf processed/* processed_raw/* processed_images/* images/* pdfs_processed/* .generated/*
rm -f processing_log.csv
echo "Cleaned all processed files"
ENDSSH
            echo -e "${GREEN}✅ Clean complete${NC}"
        fi
        ;;

    *)
        echo "AWS GPU Instance Helper"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Setup Commands:"
        echo "  setup                Complete setup (config + code + deps)"
        echo "  deploy-config        Deploy .env and GCP key to instance"
        echo "  sync-code            Sync code changes to instance"
        echo ""
        echo "Operation Commands:"
        echo "  connect              SSH into instance"
        echo "  upload <file>        Upload PDF to to_process/"
        echo "  upload-dir <dir>     Upload all PDFs from directory"
        echo "  run                  Run full pipeline on instance"
        echo "  run-md-only          Run pipeline (stop after markdown)"
        echo "  status               Check instance status"
        echo "  logs                 View processing logs"
        echo ""
        echo "Download Commands:"
        echo "  download             Download results to ./aws_results/"
        echo "  download-code        Download code from AWS (excludes aws_helper.sh)"
        echo ""
        echo "Maintenance Commands:"
        echo "  clean                Remove all processed files on instance"
        echo ""
        echo "Examples:"
        echo "  $0 setup                    # First-time setup"
        echo "  $0 deploy-config            # Update config files only"
        echo "  $0 sync-code                # Update code only"
        echo "  $0 upload-dir ./pdfs/       # Upload PDFs"
        echo "  $0 run                      # Run pipeline"
        echo "  $0 download                 # Download results"
        exit 1
        ;;
esac
