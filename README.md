# PDF Processing Pipeline

A comprehensive pipeline for fetching, converting, and enriching PDF documents with AI-generated image descriptions using Google's Gemini API.

## Overview

This pipeline processes PDF documents through five main stages:

1. **Fetch Documents** (`1_fetch_documents.py`) - Download PDFs from the Librarian API
2. **Convert PDFs** (`2_batch_convert_pdfs.py`) - Extract content and images using Docling
3. **Generate Image Descriptions** (Steps 3a-3d) - Create AI descriptions via Gemini Batch API
4. **Filter Descriptions** (Steps 4a-4d) - Apply quality filtering to descriptions
5. **Integrate Results** (`5_integrate_descriptions.py`) - Merge descriptions into final markdown

## Prerequisites

### Required Authentication

1. **Librarian API Access**
   - API key for the Librarian service
   - Used in Step 1 to fetch documents

2. **Google Cloud & Gemini API**
   - Google Cloud project with Gemini API enabled
   - Service account credentials JSON file
   - Google Cloud Storage bucket for batch processing
   - Used in Steps 3 and 4 for image description generation

### System Requirements

- Python 3.8+
- pip package manager
- For AWS: SSH access with PEM key file

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Authentication

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your Librarian API key:

```
API_KEY=your-librarian-api-key-here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

**Important**: The `.env` file is gitignored to protect your credentials.

### 3. Configure Google Cloud

Set your Google Cloud credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

Or add this to your `.env` file as shown above.

### 4. Configure Pipeline Settings

Edit `config.yaml` to customize:
- API endpoints and timeouts
- Document filters (countries, types, date ranges)
- Download settings (concurrency, rate limits)
- Processing settings (batch size, GPU usage, quality)
- Input/output folder paths

## Running the Pipeline

### Step 1: Fetch Documents

Download PDFs from the Librarian API:

```bash
python 1_fetch_documents.py
```

**What it does:**
- Fetches companies matching your country filters
- Downloads PDFs (filings, slides) for each company
- Saves metadata to SQLite database (`to_process/metadata.db`)
- Stores PDFs in `to_process/` directory
- Skips already downloaded documents

**Configuration:** Edit the `filters` and `download` sections in `config.yaml`

---

### Step 2: Convert PDFs to Markdown

Extract text, tables, and images from PDFs:

```bash
python 2_batch_convert_pdfs.py
```

**What it does:**
- Processes all PDFs in `to_process/`
- Extracts structured content (text, tables, formulas)
- Saves images to `images/` directory
- Outputs markdown to `processed/` directory
- Tracks status in `processing_log.csv`

**Features:**
- Automatic GPU acceleration (CUDA/MPS)
- Parallel processing with configurable batch size
- High-quality table extraction
- OCR for scanned documents

**Configuration:** Edit the `processing` section in `config.yaml`

---

### Step 3: Generate Image Descriptions

Use Google Gemini Batch API to generate AI descriptions for all extracted images.

#### Step 3a: Prepare Image Batches

```bash
python 3a_prepare_image_batches.py
```

Creates batch request files for the Gemini API.

#### Step 3b: Upload Batches

```bash
python 3b_upload_batches.py
```

Uploads batch requests to Google Cloud Storage and submits jobs to Gemini API.

#### Step 3c: Monitor Batches

```bash
python 3c_monitor_batches.py
```

Monitors batch job progress. Polls until all jobs are complete.

#### Step 3d: Download Results

```bash
python 3d_download_batch_results.py
```

Downloads completed results and extracts image descriptions.

**Output:** Enhanced markdown files in `processed_images/` with image descriptions

---

### Step 4: Filter Descriptions (Optional)

Apply quality filtering to remove irrelevant or low-quality descriptions.

#### Step 4a: Prepare Filter Batches

```bash
python 4a_prepare_filter_batches.py
```

Creates batch requests to filter existing descriptions.

#### Step 4b: Upload Filter Batches

```bash
python 4b_upload_filter_batches.py
```

Uploads filter requests to Gemini API.

#### Step 4c: Monitor Filter Jobs

```bash
python 4c_monitor_filter_batches.py
```

Monitors filter job progress until completion.

#### Step 4d: Download Filtered Results

```bash
python 4d_download_filter_results.py
```

Downloads and applies filtered results.

---

### Step 5: Integrate Descriptions

Merge image descriptions into final markdown files:

```bash
python 5_integrate_descriptions.py
```

**What it does:**
- Combines base markdown with image descriptions
- Creates final enhanced documents
- Handles both filtered and unfiltered descriptions
- Updates processing log with completion status

**Output:** Final enhanced markdown files in `processed_images/`

---

## AWS Helper Script

For running the pipeline on AWS GPU instances, use `aws_helper.sh`.

### Configuration

Edit these variables in `aws_helper.sh`:

```bash
INSTANCE_IP="your-instance-ip"
INSTANCE_USER="ubuntu"
PEM_KEY="PDF_key.pem"
REMOTE_DIR="~/pdf_pipeline"
```

### Available Commands

```bash
# Connect to AWS instance
./aws_helper.sh connect

# Upload a single PDF
./aws_helper.sh upload document.pdf

# Upload a directory of PDFs
./aws_helper.sh upload-dir ./my_pdfs/

# Run the full pipeline on AWS
./aws_helper.sh run

# Download processed results
./aws_helper.sh download

# Check instance status (PDFs, markdown, GPU)
./aws_helper.sh status

# Sync local code changes to instance
./aws_helper.sh sync-code

# Download code from AWS (creates backup)
./aws_helper.sh download-code

# View processing logs
./aws_helper.sh logs

# Clean all processed files on instance
./aws_helper.sh clean
```

### Example Workflow

```bash
# 1. Upload PDFs to process
./aws_helper.sh upload-dir ./pdfs/

# 2. Connect and run pipeline
./aws_helper.sh connect
# (On AWS) python 1_fetch_documents.py
# (On AWS) python 2_batch_convert_pdfs.py
# ... etc

# 3. Download results
./aws_helper.sh download
```

---

## Directory Structure

```
.
├── 1_fetch_documents.py             # Step 1: Fetch PDFs from API
├── 2_batch_convert_pdfs.py          # Step 2: Convert to markdown
├── 3a_prepare_image_batches.py      # Step 3a: Prepare batches
├── 3b_upload_batches.py             # Step 3b: Upload to Gemini
├── 3c_monitor_batches.py            # Step 3c: Monitor jobs
├── 3d_download_batch_results.py     # Step 3d: Download results
├── 4a_prepare_filter_batches.py     # Step 4a: Prepare filters
├── 4b_upload_filter_batches.py      # Step 4b: Upload filters
├── 4c_monitor_filter_batches.py     # Step 4c: Monitor filters
├── 4d_download_filter_results.py    # Step 4d: Download filtered
├── 5_integrate_descriptions.py      # Step 5: Final integration
│
├── document_metadata.py             # Metadata & SQLite management
├── docling_converter.py             # Core PDF conversion logic
├── processing_logger.py             # Processing status tracking
│
├── aws_helper.sh                    # AWS operations helper
├── config.yaml                      # Pipeline configuration
├── .env                             # API keys (gitignored)
├── .env.example                     # Environment template
├── requirements.txt                 # Python dependencies
│
├── to_process/                      # Input PDFs & metadata.db
├── processed/                       # Converted markdown (Step 2)
├── images/                          # Extracted images (Step 2)
└── processed_images/                # Final enhanced markdown (Steps 3-5)
```

---

## Configuration Guide

### API Configuration (`config.yaml`)

```yaml
api:
  base_url: "https://librarian.production.primerapp.com/api/v1"
  api_key: "${API_KEY}"  # Loaded from .env
  timeout: 60
  max_retries: 3
```

### Document Filters

```yaml
filters:
  countries:
    - Canada
    - United Kingdom
    - Germany
    # ... add more countries

  document_types:
    - filing
    - slides

  date_range:
    start: "2023-01-01"  # null = no limit
    end: null
```

### Processing Settings

```yaml
processing:
  batch_size: 3              # Parallel PDF conversions
  use_gpu: true              # Auto-detect GPU
  table_mode: "accurate"     # or "fast"
  images_scale: 3.0          # Image quality
  do_cell_matching: true     # Precise table extraction
```

---

## Troubleshooting

### Authentication Errors

**Problem:** `401 Unauthorized` or `403 Forbidden`

**Solution:**
- Verify API key in `.env` file
- Check Google Cloud credentials path
- Ensure service account has necessary permissions (Gemini API, Cloud Storage)

### Batch Jobs Stuck

**Problem:** Jobs remain in `PENDING` or `RUNNING` state

**Solution:**
- Check Gemini API quotas in Google Cloud Console
- Verify Cloud Storage bucket permissions
- Wait longer (batch jobs can take 30+ minutes)
- Monitor with step 3c or 4c scripts

### Out of Memory

**Problem:** Process crashes during PDF conversion

**Solution:**
- Reduce `batch_size` in `config.yaml`
- Disable GPU with `use_gpu: false`
- Process fewer PDFs at once
- Use AWS instance with more RAM

### Missing Images

**Problem:** No images extracted from PDFs

**Solution:**
- Check if PDFs actually contain images (not just text)
- Increase `images_scale` in `config.yaml`
- Review `processing_log.csv` for errors
- Try converting a single PDF manually to debug

---

## Support Files

### Core Python Modules

| File | Purpose |
|------|---------|
| `document_metadata.py` | SQLite database management for document tracking |
| `docling_converter.py` | PDF to markdown conversion using Docling |
| `processing_logger.py` | CSV logging for processing status |

### Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main pipeline configuration |
| `.env` | API keys and credentials (not in git) |
| `.env.example` | Template for creating `.env` |
| `requirements.txt` | Python dependencies |

---

## Notes

- All pipeline steps are **idempotent** (safe to re-run)
- Processing state tracked in `processing_log.csv`
- Batch API jobs may take 30 minutes to several hours
- Pipeline skips already processed documents
- Monitor Google Cloud costs when using Gemini API

---

## Performance Tips

### GPU Acceleration

The pipeline automatically detects and uses:
- NVIDIA GPUs (CUDA) - 2-3x faster
- Apple Silicon (M1/M2/M3 MPS) - 2x faster
- CPU fallback if no GPU available

### Batch Processing

For large volumes:
- Increase `batch_size` in `config.yaml` (if you have GPU)
- Use AWS GPU instances (`g4dn.xlarge` or similar)
- Process during off-peak hours for lower cloud costs

### Rate Limiting

Adjust in `config.yaml`:
```yaml
download:
  concurrent_downloads: 5    # Increase for faster downloads
  rate_limit_delay: 0.5      # Decrease if API allows
```

---

## License

[Add your license information here]
