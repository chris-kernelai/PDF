#!/bin/bash
################################################################################
# LOCAL_2_aws_helper.sh
#
# AWS instance operations - Run this FROM your LOCAL machine
#
# Usage:
#   ./LOCAL_2_aws_helper.sh setup [instance]           # Complete first-time setup
#   ./LOCAL_2_aws_helper.sh deploy-config [instance]   # Deploy .env and GCP key
#   ./LOCAL_2_aws_helper.sh sync-code [instance]       # Sync code changes to instance
#   ./LOCAL_2_aws_helper.sh connect [instance]         # SSH into instance
#   ./LOCAL_2_aws_helper.sh upload <file> [instance]   # Upload file to to_process/
#   ./LOCAL_2_aws_helper.sh upload-dir <dir> [instance] # Upload directory of PDFs
#   ./LOCAL_2_aws_helper.sh run [instance]             # Run pipeline on instance
#   ./LOCAL_2_aws_helper.sh download [instance]        # Download processed results
#   ./LOCAL_2_aws_helper.sh logs [instance]            # View processing logs
#   ./LOCAL_2_aws_helper.sh status [instance]          # Check instance status
#   ./LOCAL_2_aws_helper.sh clean [instance]           # Remove processed files
#
# Available instances:
#   PDF (default): 3.101.112.7 - GPU (Tesla T4) - US West
#   PDF-London:    35.178.204.146 - GPU (Tesla T4) - London
#
################################################################################

# Usage helper
usage() {
    cat <<'EOF'
Usage: ./LOCAL_2_aws_helper.sh <command> [instance|ip] [args]

Commands:
  setup [instance]            Complete first-time setup
  deploy-config [instance]    Deploy .env and GCP key
  sync-code [instance]        Sync code changes to instance
  connect [instance]          SSH into instance
  upload <file> [instance]    Upload single PDF to to_process/
  upload-dir <dir> [instance] Upload directory of PDFs
  run [instance]              Run legacy pipeline script
  download [instance]         Download processed results
  logs [instance]             View processing logs
  status [instance]           Check instance status
  clean [instance]            Remove processed files

Instances:
  PDF (default)
  PDF-London
  or provide a raw IP address
EOF
}

# Function to get instance configuration
get_instance_config() {
    local instance=$1
    case "$instance" in
        PDF)
            INSTANCE_IP="204.236.163.8"
            INSTANCE_ID="i-09f9f69a561efe64c"
            INSTANCE_REGION="us-west-1"
            PEM_KEY="workspace/configs/keys/PDF.pem"
            INSTANCE_TYPE="GPU (Tesla T4)"
            ;;
        PDF-London)
            INSTANCE_IP="35.178.204.146"
            INSTANCE_ID="i-0131ec0698e8c7bbf"
            INSTANCE_REGION="eu-west-2"
            PEM_KEY="workspace/configs/keys/PDF-London.pem"
            INSTANCE_TYPE="GPU (Tesla T4 London)"
            ;;
        *)
            return 1
            ;;
    esac
    return 0
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

INSTANCE_ARG=""
if [ $# -gt 0 ]; then
    if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || get_instance_config "$1" >/dev/null 2>&1; then
        INSTANCE_ARG="$1"
        shift
    fi
fi

if [[ -n "$INSTANCE_ARG" ]]; then
    if [[ "$INSTANCE_ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        INSTANCE_IP="$INSTANCE_ARG"
        INSTANCE_ID=""
        INSTANCE_REGION=""
        PEM_KEY="workspace/configs/keys/PDF.pem"
        INSTANCE_TYPE="Custom"
    else
        get_instance_config "$INSTANCE_ARG"
    fi
else
    get_instance_config "PDF"
fi

INSTANCE_USER="ubuntu"
REMOTE_DIR="~/pdf_pipeline"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

debug() {
    echo -e "${YELLOW}[debug]${NC} $1"
}

# Show which instance we're using
echo -e "${BLUE}Target: ${INSTANCE_TYPE} instance @ ${INSTANCE_IP}${NC}"

debug "Using PEM key path: $PEM_KEY"
if [ -f "$PEM_KEY" ]; then
    debug "PEM key details: $(ls -l "$PEM_KEY")"
fi

# Check PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}❌ PEM key not found: $PEM_KEY${NC}"
    exit 1
fi

chmod 400 "$PEM_KEY"

case "$COMMAND" in
    connect)
        echo -e "${BLUE}Connecting to AWS instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "
            if [ -d $REMOTE_DIR ]; then
                cd $REMOTE_DIR
                if [ -f venv/bin/activate ]; then
                    source venv/bin/activate
                    echo '✓ Activated virtual environment'
                else
                    echo '⚠️  Virtual environment not found - run setup first'
                fi
            else
                echo '⚠️  Pipeline directory not found at $REMOTE_DIR - run setup first'
            fi
            exec bash
        "
        ;;

    upload)
        if [ $# -lt 1 ]; then
            echo -e "${RED}❌ Please specify a file to upload${NC}"
            echo "Usage: $0 upload <file.pdf>"
            exit 1
        fi

        UPLOAD_FILE="$1"
        if [ ! -f "$UPLOAD_FILE" ]; then
            echo -e "${RED}❌ File not found: $UPLOAD_FILE${NC}"
            exit 1
        fi

        echo -e "${BLUE}Uploading $UPLOAD_FILE to instance...${NC}"
        scp -i "$PEM_KEY" "$UPLOAD_FILE" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/data/to_process/"
        echo -e "${GREEN}✅ Upload complete${NC}"
        ;;

    upload-dir)
        if [ $# -lt 1 ]; then
            echo -e "${RED}❌ Please specify a directory to upload${NC}"
            echo "Usage: $0 upload-dir <directory>"
            exit 1
        fi

        UPLOAD_DIR="$1"
        if [ ! -d "$UPLOAD_DIR" ]; then
            echo -e "${RED}❌ Directory not found: $UPLOAD_DIR${NC}"
            exit 1
        fi

        echo -e "${BLUE}Uploading PDFs from $UPLOAD_DIR to instance...${NC}"
        scp -i "$PEM_KEY" "$UPLOAD_DIR"/*.pdf "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/data/to_process/" 2>/dev/null || {
            echo -e "${RED}❌ No PDF files found in $UPLOAD_DIR${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ Upload complete${NC}"
        ;;

    download)
        echo -e "${BLUE}Downloading results from instance...${NC}"
        mkdir -p ./aws_results

        echo "Downloading processed_images/..."
        scp -i "$PEM_KEY" -r "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/data/processed_images/" ./aws_results/processed_images 2>/dev/null || echo "No processed_images found"

        echo "Downloading processing_log.csv..."
        scp -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/logs/processing_log.csv" ./aws_results/ 2>/dev/null || echo "No processing_log.csv found"

        echo "Downloading images/..."
        scp -i "$PEM_KEY" -r "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/data/images/" ./aws_results/images 2>/dev/null || echo "No images found"

        echo -e "${GREEN}✅ Results downloaded to ./aws_results/${NC}"
        ;;

    run)
        echo -e "${BLUE}Running pipeline on instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source workspace/venv/bin/activate && ./workspace/scripts/legacy_pipeline/run_pipeline.sh"
        ;;

    run-md-only)
        echo -e "${BLUE}Running pipeline (MD only) on instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source workspace/venv/bin/activate && ./workspace/scripts/legacy_pipeline/run_pipeline.sh --md-only"
        ;;

    logs)
        echo -e "${BLUE}Viewing processing logs...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" -t "cd $REMOTE_DIR && source workspace/venv/bin/activate && python3 workspace/scripts/tools/view_processing_log.py --summary"
        ;;

    sync-code)
        echo -e "${BLUE}Syncing code changes to instance...${NC}"

        FILES_TO_COPY=(
            "run_pipeline.py"
            "run_pipeline.sh"
            "run_pipeline_md_only.sh"
            "run_pipeline_images_only.sh"
            "run_pipeline_md_upload.sh"
            "run_integrate_images.sh"
            "SETUP_1_python_venv.sh"
            "SETUP_2_verify_environment.sh"
        )

        for file in "${FILES_TO_COPY[@]}"; do
            if [ -f "$file" ]; then
                debug "Copying $file"
                scp -i "$PEM_KEY" "$file" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
            fi
        done

        SSH_TARGET="${INSTANCE_USER}@${INSTANCE_IP}"
        SYNC_OPTS=(-az --delete -e "ssh -i $PEM_KEY")

        if [ -d "workspace/src" ]; then
            debug "Syncing workspace/src"
            rsync "${SYNC_OPTS[@]}" --exclude '__pycache__' --exclude '*.pyc' workspace/src/ "${SSH_TARGET}:${REMOTE_DIR}/workspace/src/"
        fi

        if [ -d "workspace/scripts" ]; then
            debug "Syncing workspace/scripts"
            rsync "${SYNC_OPTS[@]}" --include '*/' --include '*.py' --include '*.sh' --exclude '*' workspace/scripts/ "${SSH_TARGET}:${REMOTE_DIR}/workspace/scripts/"
        fi

        if [ -d "workspace/configs" ]; then
            debug "Syncing workspace/configs"
            rsync "${SYNC_OPTS[@]}" workspace/configs/ "${SSH_TARGET}:${REMOTE_DIR}/workspace/configs/"
        fi

        debug "Setting execute permissions on remote wrappers"
        ssh -i "$PEM_KEY" "$SSH_TARGET" "cd $REMOTE_DIR && chmod +x run_pipeline.sh run_pipeline_md_only.sh run_pipeline_images_only.sh run_pipeline_md_upload.sh run_integrate_images.sh SETUP_1_python_venv.sh SETUP_2_verify_environment.sh 2>/dev/null || true"

        echo -e "${GREEN}✅ Code synced (excludes data and venv)${NC}"
        ;;

    deploy-config)
        echo -e "${BLUE}Deploying configuration to instance...${NC}"

        # Check if .env exists
        if [ ! -f ".env" ]; then
            echo -e "${RED}❌ .env file not found${NC}"
            exit 1
        fi

        debug "Copying .env to remote"
        scp -i "$PEM_KEY" .env "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"

        # Copy GCP service account key if it exists
        GCP_KEY="$HOME/gcp-service-account-key.json"
        if [ -f "$GCP_KEY" ]; then
            debug "Copying GCP service account key"
            scp -i "$PEM_KEY" "$GCP_KEY" "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
        else
            echo -e "${YELLOW}⚠️  GCP service account key not found at $GCP_KEY${NC}"
            echo -e "${YELLOW}   Pipeline will require manual gcloud authentication${NC}"
        fi

        # Copy config.yaml if it exists
        if [ -f "workspace/configs/config.yaml" ]; then
            debug "Copying config.yaml"
            scp -i "$PEM_KEY" workspace/configs/config.yaml "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/configs/"
        fi

        # Copy requirements.txt if it exists
        if [ -f "workspace/configs/requirements.txt" ]; then
            debug "Copying requirements.txt"
            scp -i "$PEM_KEY" workspace/configs/requirements.txt "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/workspace/configs/"
        fi

        # Fix GOOGLE_APPLICATION_CREDENTIALS path in remote .env and authenticate gcloud
        echo "Fixing GOOGLE_APPLICATION_CREDENTIALS path and authenticating gcloud..."
        debug "Executing remote gcloud auth script"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline
if [ -f ".env" ] && [ -f "gcp-service-account-key.json" ]; then
    # Update the path to point to the remote location (use full path, not ~)
    sed -i 's|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=/home/ubuntu/pdf_pipeline/gcp-service-account-key.json|g' .env
    echo "✓ Updated GOOGLE_APPLICATION_CREDENTIALS path"

    # Authenticate gcloud with the service account
    gcloud auth activate-service-account --key-file=/home/ubuntu/pdf_pipeline/gcp-service-account-key.json 2>/dev/null && echo "✓ Authenticated gcloud with service account" || echo "⚠️  gcloud authentication skipped (gcloud may not be installed)"
fi
ENDSSH

        # Set proper permissions on remote
        debug "Setting permissions on remote secrets"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "cd $REMOTE_DIR && chmod 600 .env gcp-service-account-key.json 2>/dev/null || true"

        echo -e "${GREEN}✅ Configuration deployed${NC}"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "  1. Connect to instance: $0 connect"
        echo "  2. Install dependencies: pip install -r workspace/configs/requirements.txt"
        echo "  3. Run pipeline: ./run_pipeline.sh <min_id> <max_id>"
        ;;

    setup)
        echo -e "${BLUE}Setting up AWS instance from scratch...${NC}"
        echo ""

        # First, create the directory on AWS
        echo -e "${BLUE}Step 1: Creating directory on AWS...${NC}"
        debug "Running: ssh -i $PEM_KEY ${INSTANCE_USER}@${INSTANCE_IP} mkdir -p ~/pdf_pipeline/workspace/{configs,data}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "mkdir -p ~/pdf_pipeline/workspace/{configs,data}"
        debug "Running: ssh -i $PEM_KEY ${INSTANCE_USER}@${INSTANCE_IP} mkdir -p workspace data subdirs"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "mkdir -p ~/pdf_pipeline/workspace/data/{to_process,processed,processed_raw,processed_images,processed_images_raw,images,pdfs_processed} ~/pdf_pipeline/workspace/logs ~/pdf_pipeline/workspace/state ~/pdf_pipeline/workspace/.generated"
        echo -e "${GREEN}✅ Directory created${NC}"

        echo ""
        echo -e "${BLUE}Step 2: Deploying configuration...${NC}"
        "$0" deploy-config "$INSTANCE_IP"

        echo ""
        echo -e "${BLUE}Step 3: Syncing code...${NC}"
        "$0" sync-code "$INSTANCE_IP"

        echo ""
        echo -e "${BLUE}Step 4: Attaching IAM role for S3 access...${NC}"

        # Check if instance already has IAM role
        HAS_ROLE=$(aws ec2 describe-instances \
            --region "$INSTANCE_REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
            --output text 2>/dev/null)

        if [ "$HAS_ROLE" != "None" ] && [ -n "$HAS_ROLE" ]; then
            echo "✓ IAM role already attached: $HAS_ROLE"
        else
            echo "Attaching IAM role: PDF-Pipeline-EC2-Profile..."

            # Attach the instance profile
            aws ec2 associate-iam-instance-profile \
                --region "$INSTANCE_REGION" \
                --instance-id "$INSTANCE_ID" \
                --iam-instance-profile Name=PDF-Pipeline-EC2-Profile \
                2>/dev/null && echo "✅ IAM role attached" || echo "⚠️  Failed to attach IAM role (may already exist or permissions issue)"
        fi

        echo ""
        echo -e "${BLUE}Step 5: Installing system dependencies (gcloud CLI)...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
# Install gcloud CLI if not already installed
if ! command -v gcloud &> /dev/null; then
    echo "Installing Google Cloud SDK..."

    # Install prerequisites first
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates gnupg curl

    # Import Google Cloud public key BEFORE adding the repo
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

    # NOW add Cloud SDK repo (after key is imported)
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

    # Update and install
    sudo apt-get update
    sudo apt-get install -y google-cloud-cli

    echo "✅ gcloud CLI installed"
else
    echo "✓ gcloud CLI already installed"
fi
ENDSSH

        echo ""
        echo -e "${BLUE}Step 6: Installing Python dependencies on instance...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline

# Create virtual environment if it doesn't exist
if [ ! -d "workspace/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv workspace/venv
fi

# Activate and install dependencies
source workspace/venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r workspace/configs/requirements.txt -q

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

        # First, show what Python files exist on the server
        echo -e "${BLUE}Scanning Python files on server (depth 4)...${NC}"
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" << 'ENDSSH'
cd ~/pdf_pipeline
echo "=== Python files per directory (depth 4) ==="
for depth in 1 2 3 4; do
    find . -maxdepth $depth -name "*.py" -type f ! -path "*/venv/*" ! -path "*/data/*" ! -path "*/.generated/*" | sed 's|^\./||' | sed 's|/[^/]*$||' | sort -u | while read dir; do
        if [ -z "$dir" ]; then
            dir="."
        fi
        count=$(find "./$dir" -maxdepth 1 -name "*.py" -type f | wc -l)
        if [ $count -gt 0 ]; then
            echo "  $dir: $count files"
        fi
    done
done
echo "=== Total Python files (excluding venv/data/.generated) ==="
find . -maxdepth 4 -name "*.py" -type f ! -path "*/venv/*" ! -path "*/data/*" ! -path "*/.generated/*" | wc -l
ENDSSH

        # Create backup directory with timestamp
        BACKUP_DIR="./code_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"

        echo ""
        echo "Backing up current local code to $BACKUP_DIR..."
        rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
            --include '*.py' --include '*.sh' --exclude '*' \
            ./ "$BACKUP_DIR/" 2>/dev/null || true

        # Download all Python files recursively using rsync
        echo ""
        echo -e "${BLUE}Downloading all Python files recursively...${NC}"
        rsync -avz -e "ssh -i $PEM_KEY" \
            --prune-empty-dirs \
            --exclude 'venv' \
            --exclude 'data' \
            --exclude '.generated' \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            --include '*/' \
            --include '*.py' \
            --exclude '*' \
            "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/" ./ 

        # Download shell scripts (excluding aws_helper.sh to avoid overwriting this script)
        echo ""
        echo "Downloading shell scripts..."
        rsync -avz -e "ssh -i $PEM_KEY" \
            --include '*.sh' \
            --exclude 'LOCAL_2_aws_helper.sh' \
            --exclude '*' \
            "${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/" ./ 

        # Make shell scripts executable
        find . -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

        # Show local summary
        echo ""
        echo -e "${BLUE}Local Python files downloaded:${NC}"
        find . -maxdepth 4 -name "*.py" -type f ! -path "*/venv/*" ! -path "*/code_backup*" | wc -l

        echo ""
        echo -e "${GREEN}✅ Code downloaded from AWS (all .py and .sh files recursively)${NC}"
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
rm -f workspace/logs/processing_log.csv
echo "Cleaned all processed files"
ENDSSH
            echo -e "${GREEN}✅ Clean complete${NC}"
        fi
        ;;

    *)
        echo "AWS Instance Helper"
        echo ""
        echo "Usage: $0 <command> [arguments] [instance]"
        echo ""
        echo "Setup Commands:"
        echo "  setup [instance]           Complete setup (config + code + deps)"
        echo "  deploy-config [instance]   Deploy .env and GCP key to instance"
        echo "  sync-code [instance]       Sync code changes to instance"
        echo ""
        echo "Operation Commands:"
        echo "  connect [instance]         SSH into instance"
        echo "  upload <file> [instance]   Upload PDF to to_process/"
        echo "  upload-dir <dir> [instance] Upload all PDFs from directory"
        echo "  run [instance]             Run full pipeline on instance"
        echo "  run-md-only [instance]     Run pipeline (stop after markdown)"
        echo "  status [instance]          Check instance status"
        echo "  logs [instance]            View processing logs"
        echo ""
        echo "Download Commands:"
        echo "  download [instance]        Download results to ./aws_results/"
        echo "  download-code [instance]   Download code from AWS (excludes aws_helper.sh)"
        echo ""
        echo "Maintenance Commands:"
        echo "  clean [instance]           Remove all processed files on instance"
        echo ""
        echo "Available Instances:"
        echo "  PDF (default):  3.101.112.7  - GPU (Tesla T4) - US West"
        echo "  PDF-London:     35.178.204.146 - GPU (Tesla T4) - London"
        echo ""
        echo "Examples:"
        echo "  $0 setup                   # Setup PDF instance (default)"
        echo "  $0 setup PDF-London        # Setup PDF-London instance"
        echo "  $0 connect                 # Connect to PDF instance"
        echo "  $0 connect PDF-London      # Connect to PDF-London instance"
        echo "  $0 sync-code PDF-London    # Sync code to PDF-London instance"
        echo "  $0 run                     # Run on PDF instance"
        echo "  $0 run PDF-London          # Run on PDF-London instance"
        exit 1
        ;;
esac
