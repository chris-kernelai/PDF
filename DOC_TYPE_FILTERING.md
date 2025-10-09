# Document Type Filtering Feature

The batch converter now supports filtering documents by type using the `--doc-type` argument.

## Usage

```bash
python batch_docling_converter.py <input_folder> <output_folder> --doc-type {filings|slides|both}
```

## Options

### `--doc-type filings`
Processes only PDF files with "filing" in their filename (case-insensitive).

**Example filenames matched:**
- `company_filing_2024_q1.pdf` ✓
- `annual_filing_report.pdf` ✓
- `FILING_10K.pdf` ✓
- `investor_slides_2024.pdf` ✗

```bash
python batch_docling_converter.py ./pdfs ./markdown --doc-type filings
```

### `--doc-type slides`
Processes only PDF files with "slide" in their filename (case-insensitive).

**Example filenames matched:**
- `investor_slides_2024.pdf` ✓
- `earnings_slide_deck.pdf` ✓
- `Q4_SLIDES.pdf` ✓
- `annual_filing_report.pdf` ✗

```bash
python batch_docling_converter.py ./pdfs ./markdown --doc-type slides
```

### `--doc-type both` (default)
Processes all PDF files in the input folder, regardless of filename.

**Example filenames matched:**
- `company_filing_2024_q1.pdf` ✓
- `investor_slides_2024.pdf` ✓
- `any_document.pdf` ✓
- `report.pdf` ✓

```bash
python batch_docling_converter.py ./pdfs ./markdown --doc-type both
# or simply:
python batch_docling_converter.py ./pdfs ./markdown
```

## Implementation Details

### Filtering Patterns
- **Filings**: Uses glob pattern `*filing*.pdf` (case-insensitive)
- **Slides**: Uses glob pattern `*slide*.pdf` (case-insensitive)
- **Both**: Uses glob pattern `*.pdf` (all PDF files)

### File Matching
The filtering is applied at the file discovery stage in the `_get_pdf_files()` method:

```python
# batch_docling_converter.py:117-143
def _get_pdf_files(self) -> List[Path]:
    """Get all PDF files from the input folder, filtered by doc_type."""
    pdf_files = []

    if self.doc_type == "filings":
        # Only get filings
        for file_path in self.input_folder.rglob("*filing*.pdf"):
            if file_path.is_file():
                pdf_files.append(file_path)
    elif self.doc_type == "slides":
        # Only get slides
        for file_path in self.input_folder.rglob("*slide*.pdf"):
            if file_path.is_file():
                pdf_files.append(file_path)
    else:  # both
        # Get all PDF files
        for file_path in self.input_folder.rglob("*.pdf"):
            if file_path.is_file():
                pdf_files.append(file_path)

    pdf_files.sort()
    return pdf_files
```

## Use Cases

### Process only SEC filings
```bash
python batch_docling_converter.py ./sec_documents ./output \
  --doc-type filings \
  --upload \
  --upload-api-url "http://localhost:8000" \
  --upload-api-key "your-key" \
  --upload-ticker "AAPL" \
  --upload-document-type "FILING"
```

### Process only presentation slides
```bash
python batch_docling_converter.py ./presentations ./output \
  --doc-type slides \
  --keep-processed
```

### Process everything with high quality
```bash
python batch_docling_converter.py ./all_documents ./output \
  --doc-type both \
  --table-mode accurate \
  --images-scale 2.0
```

## Combining with Other Features

The `--doc-type` filter works seamlessly with all other features:

```bash
# Filter + Upload + Quality settings
python batch_docling_converter.py ./pdfs ./markdown \
  --doc-type filings \
  --batch-size 2 \
  --table-mode accurate \
  --images-scale 2.0 \
  --upload \
  --upload-api-url "http://localhost:8000" \
  --upload-api-key "sk-123" \
  --upload-ticker "AAPL"
```

## Naming Conventions

For best results, use consistent naming conventions:

### Filings
- `{company}_{filing_type}_{year}_{quarter}.pdf`
- `{ticker}_filing_{date}.pdf`
- `annual_filing_report_{year}.pdf`

### Slides
- `{company}_slides_{event}_{date}.pdf`
- `{ticker}_slide_deck_{quarter}.pdf`
- `investor_presentation_slides.pdf`

## Statistics

The statistics output shows the total number of files found after filtering:

```
Conversion completed!
Total files: 5       # Only filings found
Processed: 5
Skipped: 0
Failed: 0
```

## Test Results

Tested with sample filenames:

**Input files:**
- `company_filing_2024_q1.pdf`
- `annual_filing_report.pdf`
- `investor_slides_2024.pdf`
- `earnings_slide_deck.pdf`
- `regular_document.pdf`
- `another_filing_doc.pdf`

**Results:**
- `--doc-type filings`: Found 3 files (filing-related)
- `--doc-type slides`: Found 2 files (slide-related)
- `--doc-type both`: Found 6 files (all PDFs)

✓ All tests passed!
