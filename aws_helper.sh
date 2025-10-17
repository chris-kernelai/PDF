#!/bin/bash
################################################################################
# aws_helper.sh
#
# Helper script for common AWS GPU instance operations
#
# Usage:
#   ./aws_helper.sh connect              # SSH into instance
#   ./aws_helper.sh upload <file>        # Upload file to to_process/
#   ./aws_helper.sh upload-dir <dir>     # Upload directory of PDFs
#   ./aws_helper.sh download             # Download processed results
#   ./aws_helper.sh run                  # Run pipeline on instance
#   ./aws_helper.sh logs                 # View processing logs
#   ./aws_helper.sh sync-code            # Sync code changes to instance
#
################################################################################

# AWS Instance details
INSTANCE_IP="52.53.182.181"
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

        # Make scripts executable
        ssh -i "$PEM_KEY" "${INSTANCE_USER}@${INSTANCE_IP}" "cd $REMOTE_DIR && chmod +x *.sh"

        echo -e "${GREEN}✅ Code synced${NC}"
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
        echo "Commands:"
        echo "  connect              SSH into instance"
        echo "  upload <file>        Upload PDF to to_process/"
        echo "  upload-dir <dir>     Upload all PDFs from directory"
        echo "  download             Download results to ./aws_results/"
        echo "  download-code        Download code from AWS (excludes aws_helper.sh)"
        echo "  run                  Run full pipeline on instance"
        echo "  run-md-only          Run pipeline (stop after markdown)"
        echo "  logs                 View processing logs"
        echo "  status               Check instance status"
        echo "  sync-code            Sync code changes to instance"
        echo "  clean                Remove all processed files on instance"
        echo ""
        echo "Examples:"
        echo "  $0 upload my_report.pdf"
        echo "  $0 upload-dir ./pdfs/"
        echo "  $0 run"
        echo "  $0 download"
        echo "  $0 download-code"
        exit 1
        ;;
esac
