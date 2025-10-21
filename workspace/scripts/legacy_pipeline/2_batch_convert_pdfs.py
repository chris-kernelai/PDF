"""
CLI wrapper for Docling batch conversion.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from src.pipeline.docling_batch_converter import convert_folder
from src.pipeline.paths import DATA_DIR, LOGS_DIR, STATE_DIR

load_dotenv()

_shutdown_requested = False

for directory in [
    DATA_DIR / "to_process",
    DATA_DIR / "images",
    DATA_DIR / "processed",
    DATA_DIR / "processed_raw",
    DATA_DIR / "processed_images",
    DATA_DIR / "processed_images_raw",
    LOGS_DIR,
    STATE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def _signal_handler(signum, frame):
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        logging.warning("\n\nShutdown requested. Finishing current files and cleaning up...")
        logging.warning("Press Ctrl+C again to force quit (not recommended - may leave GPU memory allocated)")
    else:
        logging.error("Force quit requested. Exiting immediately...")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert PDF files to Markdown using Docling",
    )
    parser.add_argument("input_folder", help="Path to the folder containing PDF files")
    parser.add_argument(
        "output_folder", help="Path to the folder where Markdown files will be saved"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of files to process concurrently (default: 1)",
    )
    parser.add_argument(
        "--add-page-numbers",
        action="store_true",
        help="Add page numbers to the markdown output",
    )
    parser.add_argument(
        "--keep-processed",
        action="store_true",
        help="Keep successfully processed PDF files (default: remove them)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (default: use GPU if available)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--table-mode",
        choices=["accurate", "fast"],
        default="accurate",
        help="Table structure recognition mode (default: accurate for highest quality)",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=3.0,
        help="Image scaling factor for processing (default: 3.0, higher = better quality)",
    )
    parser.add_argument(
        "--no-cell-matching",
        action="store_true",
        help="Disable precise cell matching in tables (default: enabled for best quality)",
    )
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=0.05,
        help="OCR confidence threshold, 0-1 (default: 0.05, lower = more text captured)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Enable uploading converted PDFs to API endpoint",
    )
    parser.add_argument(
        "--upload-api-url",
        type=str,
        help="API base URL for uploads (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--upload-api-key",
        type=str,
        help="API key for authentication",
    )
    parser.add_argument(
        "--upload-ticker",
        type=str,
        help="Ticker symbol for uploaded documents",
    )
    parser.add_argument(
        "--upload-document-type",
        choices=["FILING", "CALL_TRANSCRIPT"],
        default="FILING",
        help="Document type for uploads (default: FILING)",
    )
    parser.add_argument(
        "--doc-type",
        choices=["filings", "slides", "both"],
        default="both",
        help="Filter PDFs by document type",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract and save images from PDFs to data/processed_images/ folder",
    )
    parser.add_argument(
        "--chunk-page-limit",
        type=int,
        default=30,
        help="Chunk PDFs to this many pages before conversion (default: 30)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of PDFs to process in this run (default: process all available PDFs)",
    )
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = parse_args()
    log_level = getattr(logging, args.log_level.upper())

    stats = asyncio.run(
        convert_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            batch_size=args.batch_size,
            add_page_numbers=args.add_page_numbers,
            remove_processed=not args.keep_processed,
            use_gpu=not args.no_gpu,
            log_level=log_level,
            table_mode=args.table_mode,
            images_scale=args.images_scale,
            do_cell_matching=not args.no_cell_matching,
            ocr_confidence_threshold=args.ocr_confidence,
            upload_enabled=args.upload,
            upload_api_url=args.upload_api_url,
            upload_api_key=args.upload_api_key,
            upload_ticker=args.upload_ticker,
            upload_document_type=args.upload_document_type,
            doc_type=args.doc_type,
            extract_images=args.extract_images,
            chunk_page_limit=args.chunk_page_limit,
            max_docs=args.max_docs,
        )
    )

    print("\nConversion completed!")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed_files']}")
    print(f"Skipped (has both reps in Supabase): {stats['skipped_has_both_reps']}")
    print(f"Skipped (output exists): {stats['skipped_files']}")
    print(f"Failed: {stats['failed_files']}")
    print(f"Total images extracted: {stats['total_images_extracted']}")
    print(f"Images saved to: data/images/")
    if args.upload:
        print(f"Uploaded: {stats['uploaded_files']}")
        print(f"Upload failed: {stats['upload_failed_files']}")
    if not args.keep_processed:
        print(f"Deleted: {stats['removed_files']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
