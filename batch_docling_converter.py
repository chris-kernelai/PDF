"""
Batch Docling PDF Converter

This module provides batch processing capabilities for converting multiple PDF files
to Markdown using Docling. It processes files asynchronously in configurable batches.
"""

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

import httpx

from docling_converter import DoclingConverter
from processing_logger import ProcessingLogger


def clean_repeating_characters(text: str) -> str:
    """
    Clean excessive repeating characters in markdown text.

    Rules:
    - Limit consecutive spaces to max 9 (keep below 10)
    - Limit any other repeating character to max 49 (keep below 50)

    Args:
        text: Input markdown text

    Returns:
        Cleaned markdown text
    """
    # Replace 10 or more consecutive spaces with 9 spaces
    text = re.sub(r' {10,}', ' ' * 9, text)

    # Replace 50+ consecutive occurrences of any other character with 49 occurrences
    # Match any character (except space which we already handled) repeated 50+ times
    def replace_long_repeats(match):
        char = match.group(1)
        return char * 49

    text = re.sub(r'([^\s])\1{49,}', replace_long_repeats, text)

    return text


class BatchDoclingConverter:
    """
    Batch processor for converting multiple PDF files to Markdown using Docling.

    This class provides asynchronous batch processing capabilities for converting
    multiple PDF files in parallel with configurable batch sizes.
    """

    def __init__(
        self,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        batch_size: int = 1,
        artifacts_path: Optional[str] = None,
        add_page_numbers: bool = False,
        remove_processed: bool = True,
        use_gpu: bool = True,
        log_level: int = logging.INFO,
        table_mode: str = "accurate",
        images_scale: float = 3.0,
        do_cell_matching: bool = True,
        ocr_confidence_threshold: float = 0.05,
        upload_enabled: bool = False,
        upload_api_url: Optional[str] = None,
        upload_api_key: Optional[str] = None,
        upload_ticker: Optional[str] = None,
        upload_document_type: str = "FILING",
        doc_type: str = "both",
        extract_images: bool = False,
    ):
        """
        Initialize the batch converter.

        Args:
            input_folder: Path to the folder containing PDF files to convert.
            output_folder: Path to the folder where converted Markdown files will be saved.
            batch_size: Number of files to process concurrently (default: 1).
            artifacts_path: Path for temporary artifacts. If None, uses system temp directory.
            add_page_numbers: Whether to add page numbers to the markdown output.
            remove_processed: Whether to remove successfully processed PDF files (default: True).
            use_gpu: Whether to use GPU acceleration if available (default: True).
            log_level: Logging level for the converter.
            table_mode: Table structure recognition mode - "accurate" (default, highest quality) or "fast".
            images_scale: Image scaling factor for processing (default: 3.0, higher = better quality).
            do_cell_matching: Enable precise cell matching in tables (default: True for best quality).
            ocr_confidence_threshold: OCR confidence threshold, 0-1 (default: 0.05, lower = more text captured).
            upload_enabled: Whether to upload converted PDFs to the API endpoint (default: False).
            upload_api_url: Base URL for the upload API (e.g., "http://localhost:8000").
            upload_api_key: API key for authentication.
            upload_ticker: Ticker symbol for the uploaded documents.
            upload_document_type: Type of document - "FILING" or "CALL_TRANSCRIPT" (default: "FILING").
            doc_type: Type of documents to process - "filings", "slides", or "both" (default: "both").
            extract_images: Whether to extract and save images from PDFs (default: False).
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.batch_size = batch_size
        # Don't use custom artifacts_path by default - let Docling manage models automatically
        self.artifacts_path = artifacts_path
        self.add_page_numbers = add_page_numbers
        self.remove_processed = remove_processed
        self.use_gpu = use_gpu
        self.table_mode = table_mode
        self.images_scale = images_scale
        self.do_cell_matching = do_cell_matching
        self.ocr_confidence_threshold = ocr_confidence_threshold

        # Upload configuration
        self.upload_enabled = upload_enabled
        self.upload_api_url = upload_api_url
        self.upload_api_key = upload_api_key
        self.upload_ticker = upload_ticker
        self.upload_document_type = upload_document_type

        # Document type filter
        self.doc_type = doc_type.lower()

        # Image extraction
        self.extract_images = extract_images

        # Processed documents tracker
        self.processed_tracker_file = Path("processed_documents.txt")
        self.processed_doc_ids = self._load_processed_documents()

        # Setup logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "removed_files": 0,
            "uploaded_files": 0,
            "upload_failed_files": 0,
            "skipped_already_processed": 0,
            "total_images_extracted": 0,
        }

        # Initialize processing logger
        self.proc_logger = ProcessingLogger()

    def _load_processed_documents(self) -> set:
        """
        Load the set of already processed document IDs from the tracker file.

        Returns:
            Set of document IDs (as strings) that have been processed.
        """
        if not self.processed_tracker_file.exists():
            return set()

        try:
            with open(self.processed_tracker_file, 'r') as f:
                # Read all lines, strip whitespace, filter out empty lines
                doc_ids = {line.strip() for line in f if line.strip()}
            return doc_ids
        except Exception as e:
            logging.warning(f"Could not load processed documents list: {e}")
            return set()

    def _extract_doc_id_from_filename(self, pdf_file: Path) -> Optional[str]:
        """
        Extract document ID from PDF filename.
        Assumes format: doc_12345.pdf -> returns "12345"

        Args:
            pdf_file: Path to PDF file

        Returns:
            Document ID as string, or None if cannot extract
        """
        filename = pdf_file.stem  # Get filename without extension
        if filename.startswith('doc_'):
            return filename[4:]  # Remove 'doc_' prefix
        return None

    def _is_already_processed(self, pdf_file: Path) -> bool:
        """
        Check if a document has already been processed.

        Args:
            pdf_file: Path to PDF file

        Returns:
            True if document has been processed before
        """
        doc_id = self._extract_doc_id_from_filename(pdf_file)
        if doc_id is None:
            return False
        return doc_id in self.processed_doc_ids

    def _mark_as_processed(self, pdf_file: Path):
        """
        Add document ID to the processed list and append to tracker file.

        Args:
            pdf_file: Path to PDF file that was successfully processed
        """
        doc_id = self._extract_doc_id_from_filename(pdf_file)
        if doc_id is None:
            return

        # Add to in-memory set
        self.processed_doc_ids.add(doc_id)

        # Append to file
        try:
            with open(self.processed_tracker_file, 'a') as f:
                f.write(f"{doc_id}\n")
        except Exception as e:
            self.logger.warning(f"Could not update processed documents list: {e}")

    def _get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the input folder, filtered by doc_type."""
        pdf_files = []

        # Define patterns for different document types
        filing_pattern = "*filing*.pdf"
        slide_pattern = "*slide*.pdf"

        if self.doc_type == "filings":
            # Only get filings
            for file_path in self.input_folder.rglob(filing_pattern):
                if file_path.is_file():
                    pdf_files.append(file_path)
        elif self.doc_type == "slides":
            # Only get slides
            for file_path in self.input_folder.rglob(slide_pattern):
                if file_path.is_file():
                    pdf_files.append(file_path)
        else:  # both
            # Get all PDF files
            for file_path in self.input_folder.rglob("*.pdf"):
                if file_path.is_file():
                    pdf_files.append(file_path)

        # Sort files for consistent processing order
        pdf_files.sort()
        return pdf_files

    def _get_output_path(self, input_file: Path) -> Path:
        """Get the output path for a given input file."""
        # Get relative path from input folder
        relative_path = input_file.relative_to(self.input_folder)
        # Change extension to .md
        output_file = relative_path.with_suffix(".md")
        # Create full output path
        return self.output_folder / output_file

    async def _upload_pdf(self, pdf_file: Path) -> Tuple[bool, str]:
        """
        Upload a PDF file to the API endpoint.

        Args:
            pdf_file: Path to the PDF file to upload.

        Returns:
            Tuple of (success, message)
        """
        if not self.upload_enabled:
            return False, "Upload not enabled"

        if not self.upload_api_url or not self.upload_api_key or not self.upload_ticker:
            return False, "Upload configuration incomplete"

        try:
            # TODO: Replace with actual upload endpoint
            # Placeholder URL - replace with your actual upload endpoint
            upload_url = f"{self.upload_api_url.rstrip('/')}/api/v1/upload-placeholder"

            # TODO: Update this payload structure to match your actual API requirements
            # This is a placeholder structure
            data = {
                "ticker": self.upload_ticker,
                "document_type": self.upload_document_type,
                "title": pdf_file.stem,
                "filing_date": datetime.now().isoformat() + "Z",
                "period_date": datetime.now().isoformat() + "Z",
            }

            self.logger.info(
                "[PLACEHOLDER] Would upload %s to %s with data: %s",
                pdf_file.name,
                upload_url,
                data
            )

            # TODO: Implement actual upload logic here
            # For now, this is a placeholder that simulates success
            return True, f"[PLACEHOLDER] Upload simulated for {pdf_file.name}"

            # Uncomment and modify this code when ready to implement actual upload:
            # with open(pdf_file, "rb") as f:
            #     files = {
            #         "file": (pdf_file.name, f, "application/pdf")
            #     }
            #
            #     headers = {
            #         "Authorization": f"Bearer {self.upload_api_key}"
            #     }
            #
            #     async with httpx.AsyncClient(timeout=300.0) as client:
            #         response = await client.post(
            #             upload_url,
            #             files=files,
            #             data=data,
            #             headers=headers
            #         )
            #
            #         if response.status_code == 200:
            #             result = response.json()
            #             document_id = result.get("id", "unknown")
            #             self.logger.info(
            #                 "Successfully uploaded %s (document_id: %s)",
            #                 pdf_file.name,
            #                 document_id
            #             )
            #             return True, f"Uploaded with document_id: {document_id}"
            #         else:
            #             error_msg = f"Upload failed with status {response.status_code}: {response.text}"
            #             self.logger.error("Failed to upload %s: %s", pdf_file.name, error_msg)
            #             return False, error_msg

        except Exception as e:
            error_msg = f"Upload exception: {str(e)}"
            self.logger.error("Failed to upload %s: %s", pdf_file.name, error_msg)
            return False, error_msg

    async def _convert_single_file(self, pdf_file: Path) -> Tuple[Path, Path, bool, str, int]:
        """
        Convert a single PDF file to Markdown.

        Args:
            pdf_file: Path to the PDF file to convert.

        Returns:
            Tuple of (input_path, output_path, success, error_message, image_count)
        """
        output_path = self._get_output_path(pdf_file)

        # Check if already processed
        if self._is_already_processed(pdf_file):
            self.logger.info("Skipping %s - already processed", pdf_file.name)
            return pdf_file, output_path, True, "Skipped - already processed", 0

        # Skip if output file already exists
        if output_path.exists():
            self.logger.info("Skipping %s - output already exists", pdf_file.name)
            return pdf_file, output_path, True, "Skipped - output already exists", 0

        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert the file
            converter = DoclingConverter(
                artifacts_path=self.artifacts_path,
                add_page_numbers=self.add_page_numbers,
                use_gpu=self.use_gpu,
                table_mode=self.table_mode,
                images_scale=self.images_scale,
                do_cell_matching=self.do_cell_matching,
                ocr_confidence_threshold=self.ocr_confidence_threshold,
            )

            try:
                import time
                start_time = time.time()

                markdown, document, page_count = converter.convert_pdf(pdf_file)

                processing_time = time.time() - start_time

                # Extract images (always enabled now)
                image_count = 0
                # Extract document ID from filename
                doc_id = self._extract_doc_id_from_filename(pdf_file)
                if doc_id is None:
                    doc_id = pdf_file.stem

                # Create images directory in easy-to-find location
                images_output_dir = Path('images')
                image_count = converter.extract_images(document, images_output_dir, doc_id)
                self.logger.info("  Extracted %d images to images/%s/", image_count, doc_id)

                # Add metadata header to markdown
                metadata_header = f"""---
**Document:** {pdf_file.name}
**Pages:** {page_count}
**Images Extracted:** {image_count}
**Images Location:** images/{doc_id}/
**Processing Time:** {processing_time:.2f} seconds
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

"""
                raw_markdown = metadata_header + markdown

                # Clean repeating characters
                cleaned_markdown = clean_repeating_characters(raw_markdown)

                # Save raw version to processed_raw/
                # Use proper path operations instead of string replace
                raw_output_path = output_path.parent.parent / 'processed_raw' / output_path.name
                raw_output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(raw_output_path, "w", encoding="utf-8") as f:
                    f.write(raw_markdown)

                # Save cleaned version to processed/ (default)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_markdown)

                self.logger.info(
                    "Successfully converted %s (%d pages, %d images in %.2f seconds)",
                    pdf_file.name,
                    page_count,
                    image_count,
                    processing_time
                )

                # Log conversion to processing log
                # Extract doc_id from filename for logging
                doc_id = pdf_file.stem  # Use full stem as doc_id
                self.proc_logger.log_conversion(
                    doc_id=doc_id,
                    pages=page_count,
                    duration_seconds=processing_time,
                    status="success"
                )

                return pdf_file, output_path, True, "", image_count

            finally:
                converter.cleanup()

        except (FileNotFoundError, ValueError, OSError) as e:
            error_msg = f"Failed to convert {pdf_file.name}: {str(e)}"
            self.logger.error("%s", error_msg)
            return pdf_file, output_path, False, error_msg, 0

    async def _process_batch(
        self, pdf_files: List[Path]
    ) -> List[Tuple[Path, Path, bool, str, int]]:
        """
        Process a batch of PDF files concurrently.

        Args:
            pdf_files: List of PDF files to process.

        Returns:
            List of tuples containing (input_path, output_path, success, error_message, image_count) for each file.
        """
        tasks = [self._convert_single_file(pdf_file) for pdf_file in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_results: List[Tuple[Path, Path, bool, str, int]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Task failed with exception: {str(result)}"
                self.logger.error(
                    "Batch processing error for %s: %s", pdf_files[i].name, error_msg
                )
                processed_results.append(
                    (pdf_files[i], self._get_output_path(pdf_files[i]), False, error_msg, 0)
                )
            elif isinstance(result, tuple) and len(result) == 5:
                processed_results.append(result)
            else:
                # Handle unexpected result type
                error_msg = f"Unexpected result type: {type(result)}"
                self.logger.error(
                    "Unexpected result for %s: %s", pdf_files[i].name, error_msg
                )
                processed_results.append(
                    (pdf_files[i], self._get_output_path(pdf_files[i]), False, error_msg, 0)
                )

        return processed_results

    async def convert_all(self) -> Dict[str, int]:
        """
        Convert all PDF files in the input folder to Markdown.

        Returns:
            Dictionary containing conversion statistics.
        """
        self.logger.info(
            "Starting batch conversion from %s to %s",
            self.input_folder,
            self.output_folder,
        )

        # Get all PDF files
        pdf_files = self._get_pdf_files()
        self.stats["total_files"] = len(pdf_files)

        if not pdf_files:
            self.logger.warning("No PDF files found in %s", self.input_folder)
            return self.stats

        self.logger.info("Found %d PDF files to convert", len(pdf_files))

        # Process files in batches
        for i in range(0, len(pdf_files), self.batch_size):
            batch = pdf_files[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(pdf_files) + self.batch_size - 1) // self.batch_size

            self.logger.info(
                "Processing batch %d/%d (%d files)",
                batch_num,
                total_batches,
                len(batch),
            )

            # Process the batch
            results = await self._process_batch(batch)

            # Update statistics, upload files, and remove successfully processed files
            for input_path, _output_path, success, error_msg, image_count in results:
                # Track image count
                if success and image_count > 0:
                    self.stats["total_images_extracted"] += image_count
                if success:
                    if "Skipped - already processed" in error_msg:
                        self.stats["skipped_already_processed"] += 1
                    elif "Skipped" in error_msg:
                        self.stats["skipped_files"] += 1
                    else:
                        self.stats["processed_files"] += 1

                        # Mark as processed
                        self._mark_as_processed(input_path)

                        # Upload the PDF if upload is enabled
                        if self.upload_enabled:
                            upload_success, upload_msg = await self._upload_pdf(input_path)
                            if upload_success:
                                self.stats["uploaded_files"] += 1
                            else:
                                self.stats["upload_failed_files"] += 1
                                self.logger.warning(
                                    "Upload failed for %s: %s",
                                    input_path.name,
                                    upload_msg
                                )

                        # Move the successfully processed PDF to pdfs_processed folder
                        # Only move if upload succeeded or upload is not enabled
                        should_move = self.remove_processed and (
                            not self.upload_enabled or upload_success
                        )
                        if should_move:
                            try:
                                # Create pdfs_processed directory
                                processed_pdf_dir = Path("pdfs_processed")
                                processed_pdf_dir.mkdir(exist_ok=True)

                                # Move the PDF
                                dest_path = processed_pdf_dir / input_path.name
                                input_path.rename(dest_path)
                                self.stats["removed_files"] += 1
                                self.logger.info("Moved processed file to: %s", dest_path)
                            except OSError as e:
                                self.logger.warning(
                                    "Failed to move processed file %s: %s",
                                    input_path.name,
                                    e
                                )
                else:
                    self.stats["failed_files"] += 1

        # Log final statistics
        self.logger.info("Conversion completed!")
        self.logger.info("Total files: %d", self.stats["total_files"])
        self.logger.info("Processed: %d", self.stats["processed_files"])
        self.logger.info("Skipped (already processed): %d", self.stats["skipped_already_processed"])
        self.logger.info("Skipped (output exists): %d", self.stats["skipped_files"])
        self.logger.info("Failed: %d", self.stats["failed_files"])
        self.logger.info("Total images extracted: %d", self.stats["total_images_extracted"])
        if self.upload_enabled:
            self.logger.info("Uploaded: %d", self.stats["uploaded_files"])
            self.logger.info("Upload failed: %d", self.stats["upload_failed_files"])
        if self.remove_processed:
            self.logger.info("Moved to pdfs_processed/: %d", self.stats["removed_files"])

        return self.stats

    def cleanup(self):
        """Clean up temporary artifacts."""
        if self.artifacts_path and os.path.exists(self.artifacts_path):
            try:
                import shutil

                shutil.rmtree(self.artifacts_path, ignore_errors=True)
                self.logger.info(
                    "Cleaned up artifacts directory: %s", self.artifacts_path
                )
            except (OSError, PermissionError) as e:
                self.logger.warning("Failed to clean up artifacts directory: %s", e)


async def convert_folder(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    batch_size: int = 1,
    add_page_numbers: bool = False,
    remove_processed: bool = True,
    use_gpu: bool = True,
    log_level: int = logging.INFO,
    table_mode: str = "accurate",
    images_scale: float = 3.0,
    do_cell_matching: bool = True,
    ocr_confidence_threshold: float = 0.05,
    upload_enabled: bool = False,
    upload_api_url: Optional[str] = None,
    upload_api_key: Optional[str] = None,
    upload_ticker: Optional[str] = None,
    upload_document_type: str = "FILING",
    doc_type: str = "both",
    extract_images: bool = False,
) -> Dict[str, int]:
    """
    Convert all PDF files in a folder to Markdown.

    Args:
        input_folder: Path to the folder containing PDF files.
        output_folder: Path to the folder where converted Markdown files will be saved.
        batch_size: Number of files to process concurrently (default: 1).
        add_page_numbers: Whether to add page numbers to the markdown output.
        remove_processed: Whether to remove successfully processed PDF files (default: True).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        log_level: Logging level for the converter.
        table_mode: Table structure recognition mode - "accurate" (default) or "fast".
        images_scale: Image scaling factor for processing (default: 3.0).
        do_cell_matching: Enable precise cell matching in tables (default: True).
        ocr_confidence_threshold: OCR confidence threshold, 0-1 (default: 0.05).
        upload_enabled: Whether to upload converted PDFs to the API endpoint (default: False).
        upload_api_url: Base URL for the upload API.
        upload_api_key: API key for authentication.
        upload_ticker: Ticker symbol for the uploaded documents.
        upload_document_type: Type of document - "FILING" or "CALL_TRANSCRIPT" (default: "FILING").
        doc_type: Type of documents to process - "filings", "slides", or "both" (default: "both").
        extract_images: Whether to extract and save images from PDFs (default: False).

    Returns:
        Dictionary containing conversion statistics.
    """
    converter = BatchDoclingConverter(
        input_folder=input_folder,
        output_folder=output_folder,
        batch_size=batch_size,
        add_page_numbers=add_page_numbers,
        remove_processed=remove_processed,
        use_gpu=use_gpu,
        log_level=log_level,
        table_mode=table_mode,
        images_scale=images_scale,
        do_cell_matching=do_cell_matching,
        ocr_confidence_threshold=ocr_confidence_threshold,
        upload_enabled=upload_enabled,
        upload_api_url=upload_api_url,
        upload_api_key=upload_api_key,
        upload_ticker=upload_ticker,
        upload_document_type=upload_document_type,
        doc_type=doc_type,
        extract_images=extract_images,
    )

    try:
        stats = await converter.convert_all()
        return stats
    finally:
        converter.cleanup()


def main():
    """Command-line interface for the batch converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch convert PDF files to Markdown using Docling"
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
        help="Type of documents to process: 'filings' (files with 'filing' in name), 'slides' (files with 'slide' in name), or 'both' (all PDFs) (default: both)",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Extract and save images from PDFs to processed_images/ folder",
    )

    args = parser.parse_args()

    # Convert log level string to int
    log_level = getattr(logging, args.log_level.upper())

    # Run the conversion
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
        )
    )

    print("\nConversion completed!")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed_files']}")
    print(f"Skipped: {stats['skipped_files']}")
    print(f"Failed: {stats['failed_files']}")
    print(f"Total images extracted: {stats['total_images_extracted']}")
    print(f"Images saved to: images/")
    if args.upload:
        print(f"Uploaded: {stats['uploaded_files']}")
        print(f"Upload failed: {stats['upload_failed_files']}")
    if not args.keep_processed:
        print(f"Moved to pdfs_processed/: {stats['removed_files']}")


if __name__ == "__main__":
    main()
