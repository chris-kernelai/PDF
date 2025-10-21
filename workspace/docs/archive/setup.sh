#!/bin/bash
# Setup script for Docling PDF Converter

set -e  # Exit on error

echo "Setting up Docling PDF Converter..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary folders
echo "Creating to_process/ and processed/ folders..."
mkdir -p to_process
mkdir -p processed

echo ""
echo "Setup complete!"
echo ""
echo "=========================================="
echo "Usage Options:"
echo "=========================================="
echo ""
echo "1. FULL PIPELINE (Fetch + Convert):"
echo "   Set your API key:"
echo "     export API_KEY='your-api-key-here'"
echo ""
echo "   Run the full pipeline:"
echo "     python run_pipeline.py"
echo ""
echo "   Options:"
echo "     --fetch-only      Only fetch documents, don't convert"
echo "     --convert-only    Only convert existing PDFs"
echo "     --retry-failed    Retry failed documents"
echo "     --stats           Show statistics"
echo ""
echo "2. MANUAL FETCH (API to PDFs):"
echo "   python fetch_documents.py"
echo ""
echo "   Options:"
echo "     --metadata-only        Fetch metadata without downloading PDFs"
echo "     --download-pending     Download pending PDFs from database"
echo "     --stats                Show database statistics"
echo ""
echo "3. MANUAL CONVERT (PDFs to Markdown):"
echo "   python batch_docling_converter.py to_process/ processed/"
echo ""
echo "   Options:"
echo "     --batch-size N          Process N files concurrently (default: 1)"
echo "     --add-page-numbers      Add page numbers to output"
echo "     --keep-processed        Keep PDFs after processing (don't delete)"
echo "     --log-level LEVEL       Set logging level"
echo ""
echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Edit config.yaml to customize:"
echo "  - Countries to include/exclude"
echo "  - Document types (filing, slides)"
echo "  - Date ranges"
echo "  - Batch sizes and rate limits"
echo ""
echo "Metadata database: to_process/metadata.db"
echo "Logs: fetch_documents.log"
