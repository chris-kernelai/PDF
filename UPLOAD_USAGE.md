# Upload Functionality Usage Guide

The batch converter now supports automatic uploading of processed PDFs to your API endpoint.

## Basic Usage

```bash
python batch_docling_converter.py to_process/ processed/ \
  --upload \
  --upload-api-url "http://localhost:8000" \
  --upload-api-key "your-api-key" \
  --upload-ticker "AAPL"
```

## Document Type Filtering

Use `--doc-type` to process specific document types:

```bash
# Process only filings (files with "filing" in the name)
python batch_docling_converter.py to_process/ processed/ --doc-type filings

# Process only slides (files with "slide" in the name)
python batch_docling_converter.py to_process/ processed/ --doc-type slides

# Process both (default - all PDF files)
python batch_docling_converter.py to_process/ processed/ --doc-type both
```

## Upload Options

### Required Options (when --upload is enabled)
- `--upload`: Enable uploading to API endpoint
- `--upload-api-url`: Base URL of your API (e.g., `http://localhost:8000`)
- `--upload-api-key`: Your API authentication key
- `--upload-ticker`: Stock ticker symbol for the documents

### Optional Upload Options
- `--upload-document-type {FILING,CALL_TRANSCRIPT}`: Document type (default: FILING)

### Document Filtering Options
- `--doc-type {filings,slides,both}`: Filter documents by type (default: both)
  - `filings`: Only process files with "filing" in the filename
  - `slides`: Only process files with "slide" in the filename
  - `both`: Process all PDF files

## Complete Example

```bash
python batch_docling_converter.py \
  ./input_pdfs \
  ./output_markdown \
  --batch-size 1 \
  --no-gpu \
  --table-mode accurate \
  --images-scale 2.0 \
  --doc-type filings \
  --upload \
  --upload-api-url "http://localhost:8000" \
  --upload-api-key "sk-1234567890abcdef" \
  --upload-ticker "AAPL" \
  --upload-document-type "FILING"
```

### Example: Process only slide presentations without uploading

```bash
python batch_docling_converter.py \
  ./input_pdfs \
  ./output_markdown \
  --doc-type slides \
  --keep-processed
```

## How It Works

1. **Conversion**: PDFs are converted to Markdown as usual
2. **Upload**: After successful conversion, the **original PDF** is uploaded to the API endpoint
3. **Processing**: The API processes the PDF through its pipeline, creating RAW and CLEAN representations
4. **Cleanup**: If `--keep-processed` is not specified AND upload succeeds, the PDF is removed from the input folder

## Upload Endpoint

The converter uploads to: `{upload_api_url}/mgmt/documents/upload-simple`

### Form Data Sent
- `file`: The original PDF file
- `ticker`: Company ticker symbol
- `document_type`: "FILING" or "CALL_TRANSCRIPT"
- `title`: PDF filename (without extension)
- `filing_date`: Current timestamp (ISO format)
- `period_date`: Current timestamp (ISO format)

### Response
On success (status 200), the API returns:
```json
{
  "data": {
    "source_id": "unique-document-id",
    "s3_key": "s3-storage-path",
    "token_count": 12345,
    "title": "Document Title"
  }
}
```

## Statistics

When upload is enabled, you'll see additional statistics:
```
Conversion completed!
Total files: 5
Processed: 5
Skipped: 0
Failed: 0
Uploaded: 5
Upload failed: 0
Removed: 5
```

## Error Handling

- If upload fails, the PDF file is **not removed** (even if `--remove-processed` is enabled)
- Upload failures are logged but don't stop the batch processing
- Each upload has a 300-second (5-minute) timeout

## Programmatic Usage

```python
from batch_docling_converter import convert_folder
import asyncio

stats = asyncio.run(
    convert_folder(
        input_folder="./pdfs",
        output_folder="./markdown",
        upload_enabled=True,
        upload_api_url="http://localhost:8000",
        upload_api_key="your-api-key",
        upload_ticker="AAPL",
        upload_document_type="FILING"
    )
)

print(f"Uploaded: {stats['uploaded_files']}")
print(f"Upload failures: {stats['upload_failed_files']}")
```

## Security Notes

- API keys are passed via command line arguments (consider using environment variables in production)
- Always use HTTPS in production: `https://your-api-domain.com`
- The upload feature uses the `/mgmt/documents/upload-simple` endpoint
- Authentication is via `X-API-Key` header
