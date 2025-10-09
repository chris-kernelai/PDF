# Docling PDF Converter

This directory contains two standalone Python files for converting PDF files to Markdown using the Docling library.

## Files

### 1. `docling_converter.py`
A standalone PDF to Markdown converter class that provides the core conversion functionality.

**Features:**
- Simple interface for converting individual PDF files
- Memory-efficient settings optimized for batch processing
- Optional page number insertion
- Error handling and cleanup
- Command-line interface

**Usage:**
```python
from docling_converter import DoclingConverter, convert_pdf_to_markdown

# Simple conversion
markdown = convert_pdf_to_markdown("document.pdf")

# With custom settings
converter = DoclingConverter(add_page_numbers=True)
markdown, document = converter.convert_pdf("document.pdf")
```

**Command line:**
```bash
python docling_converter.py document.pdf
```

### 2. `batch_docling_converter.py`
A batch processor for converting multiple PDF files asynchronously.

**Features:**
- Asynchronous batch processing with configurable batch size
- Recursive folder processing
- Progress logging and statistics
- Skip existing output files
- Automatic cleanup of temporary files
- Command-line interface

**Usage:**
```python
from batch_docling_converter import BatchDoclingConverter, convert_folder

# Convert all PDFs in a folder
stats = await convert_folder("input_folder", "output_folder", batch_size=2)

# Using the class directly
converter = BatchDoclingConverter("input_folder", "output_folder", batch_size=1)
stats = await converter.convert_all()
```

**Command line:**
```bash
python batch_docling_converter.py input_folder output_folder --batch-size 2 --add-page-numbers
```

## Dependencies

These files require the following Python packages:
- `docling`
- `docling-core`
- `asyncio` (built-in)
- `pathlib` (built-in)
- `logging` (built-in)
- `tempfile` (built-in)

Install with:
```bash
pip install docling docling-core
```

## Configuration

### DoclingConverter Options
- `artifacts_path`: Path for temporary artifacts (default: system temp directory)
- `add_page_numbers`: Whether to add page numbers to output (default: False)

### BatchDoclingConverter Options
- `input_folder`: Source folder containing PDF files
- `output_folder`: Destination folder for Markdown files
- `batch_size`: Number of files to process concurrently (default: 1)
- `artifacts_path`: Path for temporary artifacts (default: auto-generated temp directory)
- `add_page_numbers`: Whether to add page numbers to output (default: False)
- `log_level`: Logging level (default: INFO)

## Output

- Input: `document.pdf`
- Output: `document.md` (in the specified output folder)
- The converter preserves the folder structure from input to output

## Error Handling

Both converters include comprehensive error handling:
- File not found errors
- Invalid PDF format errors
- Conversion failures
- Memory issues
- I/O errors

Failed conversions are logged with detailed error messages and don't stop the batch processing.

## Performance

The converters are optimized for memory efficiency:
- Uses CPU-only processing to reduce memory usage
- Fast table extraction mode
- Minimal OCR processing
- Automatic cleanup of temporary files

For large batch processing, adjust the `batch_size` parameter based on your system's memory and CPU capabilities.
