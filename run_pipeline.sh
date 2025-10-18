#!/bin/bash
################################################################################
# run_pipeline.sh
#
# Complete PDF processing pipeline from Docling conversion to image integration
#
# Usage:
#   ./run_pipeline.sh              # Run full pipeline
#   ./run_pipeline.sh --md-only    # Stop after markdown conversion (for cleaning)
#   ./run_pipeline.sh -h           # Show help
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TO_PROCESS_DIR="to_process"
PROCESSED_DIR="processed"
MODE="developer"  # or "vertex"

# Parse arguments
MD_ONLY=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --md-only)
      MD_ONLY=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --md-only    Stop after markdown conversion (for manual cleaning)"
      echo "  -h, --help   Show this help message"
      echo ""
      echo "Full Pipeline Steps:"
      echo "  1. Convert PDFs to markdown (Docling)"
      echo "  2. Extract images from PDFs"
      echo "  3. Upload batches to Gemini"
      echo "  4. Monitor batch jobs until complete"
      echo "  5. Download batch results"
      echo "  6. Filter descriptions for relevance"
      echo "  7. Integrate descriptions into markdown"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PDF Processing Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for PDFs
if [ ! -d "$TO_PROCESS_DIR" ] || [ -z "$(ls -A $TO_PROCESS_DIR/*.pdf 2>/dev/null)" ]; then
  echo -e "${RED}❌ No PDFs found in $TO_PROCESS_DIR/${NC}"
  echo "   Place PDF files in $TO_PROCESS_DIR/ to begin"
  exit 1
fi

PDF_COUNT=$(ls -1 $TO_PROCESS_DIR/*.pdf 2>/dev/null | wc -l)
echo -e "${GREEN}✅ Found $PDF_COUNT PDF(s) to process${NC}"
echo ""

################################################################################
# Step 1: Convert PDFs to Markdown with Docling
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: Converting PDFs to Markdown${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 batch_docling_converter.py "$TO_PROCESS_DIR" "$PROCESSED_DIR"

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Docling conversion failed${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Step 1 complete: Markdown files created in $PROCESSED_DIR/${NC}"
echo -e "${GREEN}✅ Images extracted to images/${NC}"
echo -e "${GREEN}✅ PDFs moved to pdfs_processed/${NC}"
echo ""

# If --md-only flag is set, stop here
if [ "$MD_ONLY" = true ]; then
  echo -e "${YELLOW}⏸️  Stopped after markdown conversion (--md-only flag)${NC}"
  echo ""
  echo "Next steps:"
  echo "  1. Review/clean markdown files in $PROCESSED_DIR/"
  echo "  2. Run './run_pipeline.sh' (without --md-only) to continue"
  echo ""
  exit 0
fi

# Check if we need PDFs for image extraction
if [ ! -d "pdfs_processed" ] || [ -z "$(ls -A pdfs_processed/*.pdf 2>/dev/null)" ]; then
  echo -e "${RED}❌ No PDFs found in pdfs_processed/ for image extraction${NC}"
  echo "   PDFs should have been moved there by the converter"
  exit 1
fi

################################################################################
# Step 2: Extract images and create batch requests
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Extracting images and creating batches${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Temporarily copy PDFs back to to_process for image extraction
echo "Preparing PDFs for image extraction..."
cp pdfs_processed/*.pdf "$TO_PROCESS_DIR/"

python3 image_description_batch_preparer.py --mode "$MODE"

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Image extraction failed${NC}"
  # Clean up copied PDFs
  rm -f "$TO_PROCESS_DIR"/*.pdf
  exit 1
fi

# Clean up - move PDFs back to pdfs_processed
echo "Cleaning up..."
mv "$TO_PROCESS_DIR"/*.pdf pdfs_processed/ 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ Step 2 complete: Batch files created${NC}"
echo ""

################################################################################
# Step 3: Upload batches to Gemini
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Uploading batches to Gemini${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 gemini_batch_uploader.py "$MODE"

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Batch upload failed${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Step 3 complete: Batches uploaded${NC}"
echo ""

################################################################################
# Step 4: Monitor batch jobs until complete
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: Monitoring batch jobs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

MAX_CHECKS=20
CHECK_INTERVAL=30
check_count=0

while [ $check_count -lt $MAX_CHECKS ]; do
  check_count=$((check_count + 1))
  echo "Check $check_count/$MAX_CHECKS..."

  # Run monitor and capture output
  if python3 gemini_batch_monitor.py "$MODE" | grep -q "All batch jobs completed successfully"; then
    echo ""
    echo -e "${GREEN}✅ Step 4 complete: All batch jobs succeeded${NC}"
    echo ""
    break
  fi

  if [ $check_count -lt $MAX_CHECKS ]; then
    echo "Waiting ${CHECK_INTERVAL}s before next check..."
    sleep $CHECK_INTERVAL
  else
    echo -e "${RED}❌ Batch jobs did not complete within expected time${NC}"
    echo "   Run 'python3 gemini_batch_monitor.py $MODE' to check status"
    echo "   Then resume with: python3 gemini_batch_downloader.py $MODE"
    exit 1
  fi
done

################################################################################
# Step 5: Download batch results
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5: Downloading batch results${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 gemini_batch_downloader.py "$MODE"

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Batch download failed${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Step 5 complete: Results downloaded${NC}"
echo ""

################################################################################
# Step 6: Filter descriptions for relevance
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 6: Filtering descriptions${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 image_description_filter.py

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Description filtering failed${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Step 6 complete: Descriptions filtered${NC}"
echo ""

################################################################################
# Step 7: Integrate descriptions into markdown
################################################################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 7: Integrating descriptions into markdown${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python3 image_description_integrator.py --overwrite

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Integration failed${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Step 7 complete: Descriptions integrated${NC}"
echo ""

################################################################################
# Pipeline Complete
################################################################################
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Pipeline Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📊 View processing log:"
echo "   python3 view_processing_log.py --summary"
echo ""
echo "📁 Output locations:"
echo "   • Markdown files:          processed/"
echo "   • Enhanced markdown:       processed_images/"
echo "   • Extracted images:        images/"
echo "   • Processed PDFs:          pdfs_processed/"
echo "   • Processing log:          processing_log.csv"
echo ""
