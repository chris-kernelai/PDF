# Setup Guide

Complete setup instructions for deploying the PDF to Markdown conversion pipeline on a new server.

## Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd PDF

# 2. Run setup script (creates venv and installs dependencies)
./setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Set your API key
export API_KEY='your-librarian-api-key-here'

# 5. Create input/output folders (if not in repo)
mkdir -p to_process processed

# 6. Run the full pipeline
python run_pipeline.py
```

That's it! The pipeline will:
- Fetch documents from Librarian API
- Download PDFs to `to_process/`
- Convert to Markdown → `processed/`
- Remove successfully processed PDFs
- Auto-download all required models on first run

## What Happens on First Run

The first time you run the pipeline, Docling will automatically download required models:

1. **Layout detection models** (~100-200 MB)
   - Downloaded to: `~/.cache/huggingface/`

2. **EasyOCR models** (~100 MB)
   - Downloaded to: `~/.EasyOCR/`

3. **Table structure models** (~50 MB)
   - Downloaded to: `~/.cache/huggingface/`

**Note**: This only happens once. Subsequent runs will use cached models and start immediately.

## Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB free space (for models and temporary files)
- **Network**: Internet connection for initial model downloads

## Optional: GPU Acceleration

GPU acceleration provides 2-3x speedup and is automatically detected. Setup is optional but recommended for high-volume processing.

### For NVIDIA GPUs (CUDA)

```bash
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### For Apple Silicon (M1/M2/M3)

```bash
source venv/bin/activate
pip install torch torchvision

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### For CPU Only

No additional setup needed - the base installation works on CPU. To explicitly force CPU mode:

```bash
python run_pipeline.py --no-gpu
# or
python batch_docling_converter.py to_process/ processed/ --no-gpu
```

## Configuration

Edit `config.yaml` to customize pipeline behavior:

### API Settings

```yaml
api:
  base_url: "https://librarian.production.primerapp.com/api/v1"
  api_key: "${API_KEY}"  # Uses environment variable
  timeout: 60
  max_retries: 3
```

### Document Filters

```yaml
filters:
  # Countries to include (30+ non-US countries included)
  countries:
    - Canada
    - United Kingdom
    - Germany
    # ... (see config.yaml for full list)

  # Document types
  document_types:
    - filing
    - slides

  # Optional date range
  date_range:
    start: "2023-01-01"  # null = no limit
    end: null
```

### Download Settings

```yaml
download:
  company_page_size: 100
  document_page_size: 250
  concurrent_downloads: 5
  rate_limit_delay: 0.5  # seconds between requests
  skip_existing: true
```

## Pipeline Modes

### Full Pipeline (Recommended)

```bash
# Fetch + Convert in one command
python run_pipeline.py

# Only fetch documents
python run_pipeline.py --fetch-only

# Only convert existing PDFs
python run_pipeline.py --convert-only

# Retry failed documents
python run_pipeline.py --retry-failed

# Show statistics
python run_pipeline.py --stats
```

### Manual Fetch

```bash
# Fetch documents and download PDFs
python fetch_documents.py

# Only fetch metadata (no PDF downloads)
python fetch_documents.py --metadata-only

# Download pending PDFs from database
python fetch_documents.py --download-pending

# Show database statistics
python fetch_documents.py --stats
```

### Batch Convert

```bash
# Convert all PDFs in to_process/
python batch_docling_converter.py to_process/ processed/

# With options
python batch_docling_converter.py to_process/ processed/ \
  --batch-size 5 \
  --add-page-numbers \
  --keep-processed \
  --no-gpu \
  --log-level DEBUG
```

### Single PDF Convert

```bash
# Convert one PDF directly
python docling_converter.py document.pdf > output.md
```

## Environment Variables

```bash
# Required
export API_KEY='your-librarian-api-key-here'

# Optional: Customize model storage locations
export HF_HOME='/custom/path/huggingface'  # Hugging Face models
export EASYOCR_MODULE_PATH='/custom/path/easyocr'  # EasyOCR models
```

## Persistent API Key (Optional)

To avoid setting `API_KEY` every time:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

Or use a `.env` file:

```bash
# Create .env file
echo 'API_KEY=your-api-key-here' > .env

# Load in your script
source .env
python run_pipeline.py
```

## Deployment Tips

### Running as a Service (systemd)

```bash
# Create service file: /etc/systemd/system/pdf-pipeline.service
[Unit]
Description=PDF to Markdown Conversion Pipeline
After=network.target

[Service]
Type=oneshot
User=your-user
WorkingDirectory=/path/to/PDF
Environment="API_KEY=your-api-key"
ExecStart=/path/to/PDF/venv/bin/python run_pipeline.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable pdf-pipeline
sudo systemctl start pdf-pipeline

# Check status
sudo systemctl status pdf-pipeline
```

### Running as a Cron Job

```bash
# Edit crontab
crontab -e

# Add line (runs daily at 2 AM)
0 2 * * * cd /path/to/PDF && source venv/bin/activate && API_KEY='your-key' python run_pipeline.py >> /var/log/pdf-pipeline.log 2>&1
```

### Running in Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN ./setup.sh
ENV API_KEY=""

CMD ["python", "run_pipeline.py"]
```

```bash
# Build and run
docker build -t pdf-pipeline .
docker run -e API_KEY='your-key' -v $(pwd)/to_process:/app/to_process -v $(pwd)/processed:/app/processed pdf-pipeline
```

## Performance Tuning

### Batch Size

```bash
# With GPU: Higher batch size for better utilization
python batch_docling_converter.py to_process/ processed/ --batch-size 5

# Without GPU: Lower batch size to reduce memory
python batch_docling_converter.py to_process/ processed/ --batch-size 1
```

### Concurrent Downloads

Edit `config.yaml`:

```yaml
download:
  concurrent_downloads: 10  # Increase parallel downloads
  rate_limit_delay: 0.1     # Reduce delay between requests
```

### Expected Performance

| Configuration | Speed per PDF | Throughput (100 PDFs) |
|--------------|---------------|----------------------|
| CPU (4 cores) | 60-70 sec | ~2 hours |
| Apple M2 (MPS) | 30-35 sec | ~1 hour |
| NVIDIA RTX 3090 (CUDA) | 20-25 sec | ~40 minutes |

**Note**: Times vary based on PDF complexity (images, tables, OCR requirements)

## Troubleshooting

### Issue: "API_KEY environment variable not set"

```bash
export API_KEY='your-api-key-here'
```

### Issue: Models fail to download

**Check internet connection**:
```bash
ping huggingface.co
```

**Check disk space**:
```bash
df -h ~
```

**Manually clear cache and retry**:
```bash
rm -rf ~/.cache/huggingface ~/.EasyOCR
python run_pipeline.py
```

### Issue: Conversion fails with memory errors

**Reduce batch size**:
```bash
python batch_docling_converter.py to_process/ processed/ --batch-size 1
```

**Or force CPU mode**:
```bash
python batch_docling_converter.py to_process/ processed/ --no-gpu
```

### Issue: Database locked

Only run one pipeline instance at a time. SQLite doesn't support concurrent writes.

### Issue: GPU not detected

**Check PyTorch installation**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

**Reinstall PyTorch** with appropriate GPU support (see GPU Acceleration section above).

### Issue: Slower on GPU than CPU

This can happen when:
- Processing very small PDFs (GPU overhead > speedup)
- Limited GPU memory causes swapping
- Batch size is too small

**Solution**: Increase batch size
```bash
python batch_docling_converter.py to_process/ processed/ --batch-size 5
```

## Verification

After setup, verify everything works:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Check Python and packages
python --version
pip list | grep docling

# 3. Test single PDF conversion
echo "Test" > test.txt
python -c "from docling_converter import DoclingConverter; print('Import successful')"

# 4. Check database
python -c "from document_metadata import MetadataManager; m = MetadataManager(); print(m.get_statistics())"

# 5. Verify API connection (requires API_KEY)
python fetch_documents.py --stats
```

## Monitoring

### View Logs

```bash
# Pipeline logs
tail -f fetch_documents.log

# Real-time conversion progress
python batch_docling_converter.py to_process/ processed/ --log-level INFO
```

### Database Statistics

```bash
# From command line
python run_pipeline.py --stats

# Or programmatically
python -c "from document_metadata import MetadataManager; import json; print(json.dumps(MetadataManager().get_statistics(), indent=2))"
```

### Monitor GPU Usage

**NVIDIA GPUs**:
```bash
watch -n 1 nvidia-smi
```

**Apple Silicon**:
```bash
sudo powermetrics --samplers gpu_power -i 1000
```

## Next Steps

1. Review and customize `config.yaml` for your use case
2. Test with a small batch of PDFs first
3. Monitor performance and adjust batch sizes
4. Set up automated scheduling (cron/systemd) if needed
5. Configure log rotation for long-running deployments

## Support

For issues:
1. Check the logs: `fetch_documents.log`
2. Check statistics: `python run_pipeline.py --stats`
3. Check metadata: `to_process/metadata.db` (SQLite browser)
4. Review troubleshooting section above

## See Also

- [README.md](README.md) - Full project documentation
- [GPU_SETUP.md](GPU_SETUP.md) - Detailed GPU acceleration guide
- [config.yaml](config.yaml) - Configuration reference
