# Quick Start Guide

## 5-Minute Setup

```bash
# 1. Setup environment
./setup.sh
source venv/bin/activate

# 2. Set API key
export API_KEY='your-librarian-api-key'

# 3. Run the pipeline
python run_pipeline.py
```

Done! Documents will be fetched from the API, converted to Markdown, and saved to `processed/`.

## What Just Happened?

1. ✅ Fetched all non-US companies from Librarian API
2. ✅ Found all filings & slides for those companies
3. ✅ Downloaded PDFs to `to_process/`
4. ✅ Converted PDFs to Markdown → `processed/`
5. ✅ Removed successfully processed PDFs
6. ✅ Tracked everything in `to_process/metadata.db`

## Check Results

```bash
# Show statistics
python run_pipeline.py --stats

# Check processed markdown files
ls -lh processed/

# Check logs
tail -f fetch_documents.log
```

## Common Commands

```bash
# Full pipeline (default)
python run_pipeline.py

# Only fetch documents (no conversion)
python run_pipeline.py --fetch-only

# Only convert existing PDFs (no fetching)
python run_pipeline.py --convert-only

# Retry failed documents
python run_pipeline.py --retry-failed

# Show statistics
python run_pipeline.py --stats
```

## Manual Mode

```bash
# Step 1: Fetch PDFs from API
python fetch_documents.py

# Step 2: Convert PDFs to Markdown
python batch_docling_converter.py to_process/ processed/
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Add/remove countries
filters:
  countries:
    - Canada
    - United Kingdom
    # Add more...

# Change document types
  document_types:
    - filing
    - slides

# Add date range
  date_range:
    start: "2023-01-01"
    end: "2024-12-31"
```

## Troubleshooting

### "API_KEY not set"
```bash
export API_KEY='your-key-here'
```

### PDFs not downloading
Check the logs for specific errors:
```bash
tail -f fetch_documents.log
```

Common issues:
- Invalid API key
- Network connectivity
- Documents don't have PDFs available
- S3 presigned URL expired (should auto-retry)

### Out of memory
Reduce batch size:
```bash
python batch_docling_converter.py to_process/ processed/ --batch-size 1
```

### Check what's happening
```bash
# View logs
tail -f fetch_documents.log

# Check database
python -c "from document_metadata import MetadataManager; print(MetadataManager().get_statistics())"
```

## Next Steps

1. ✅ PDF download endpoint is implemented and ready
2. Test with a small sample first (use `--stats` to monitor)
3. Review the markdown output quality
4. Adjust configuration as needed (edit `config.yaml`)
5. Set up a cron job for daily runs

## Full Documentation

- **README.md**: Complete documentation
- **IMPLEMENTATION_SUMMARY.md**: Technical details
- **config.yaml**: All configuration options

## Need Help?

Check the logs first:
```bash
cat fetch_documents.log
```

Then check the README.md for detailed troubleshooting.
