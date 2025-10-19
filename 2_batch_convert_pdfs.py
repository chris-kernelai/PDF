"""
Batch Docling PDF Converter

This module provides batch processing capabilities for converting multiple PDF files
to Markdown using Docling. It processes files in parallel using ProcessPoolExecutor
for true multi-core utilization.
"""

import asyncio
import os
import re
import signal
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

import httpx
import asyncpg
from dotenv import load_dotenv

from src.docling_converter import DoclingConverter
from src.processing_logger import ProcessingLogger

# Load environment variables from .env file
load_dotenv()

# Global flag for graceful shutdown
_shutdown_requested = False

os.makedirs("data/images", exist_ok=True)

os.makedirs("data/processed", exist_ok=True)

os.makedirs("data/processed_raw", exist_ok=True)

os.makedirs("data/processed_images", exist_ok=True)

os.makedirs("data/processed_images_raw", exist_ok=True)


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        logging.warning("\n\nShutdown requested. Finishing current files and cleaning up...")
        logging.warning("Press Ctrl+C again to force quit (not recommended - may leave GPU memory allocated)")
    else:
        logging.error("Force quit requested. Exiting immediately...")
        sys.exit(1)


def wrap_long_lines(text: str, max_line_length: int = 10000) -> str:
    """
    Wrap lines that exceed the maximum length by inserting newlines.

    Args:
        text: Input text
        max_line_length: Maximum allowed line length (default: 10000)

    Returns:
        Text with long lines wrapped
    """
    lines = text.split('\n')
    wrapped_lines = []

    for line in lines:
        if len(line) <= max_line_length:
            wrapped_lines.append(line)
        else:
            # Split long lines at word boundaries when possible
            while len(line) > max_line_length:
                # Try to find a space near the max length to break at
                break_point = max_line_length
                space_idx = line.rfind(' ', 0, max_line_length)

                # If we found a space within reasonable distance, break there
                if space_idx > max_line_length * 0.8:  # Within last 20% of max length
                    break_point = space_idx + 1

                wrapped_lines.append(line[:break_point])
                line = line[break_point:]

            # Add the remaining part
            if line:
                wrapped_lines.append(line)

    return '\n'.join(wrapped_lines)


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


def _process_single_pdf_worker(
    pdf_file_path: str,
    output_path: str,
    raw_output_path: str,
    images_output_dir: str,
    use_gpu: bool,
    table_mode: str,
    images_scale: float,
    do_cell_matching: bool,
    ocr_confidence_threshold: float,
    add_page_numbers: bool,
    chunk_page_limit: int,
    batch_size: int = 2,
) -> Tuple[str, str, bool, str, int, int, float]:
    """
    Worker function for converting a single PDF in a separate process.
    This function is picklable and can be used with ProcessPoolExecutor.

    Args:
        pdf_file_path: Path to PDF file (as string)
        output_path: Path for output markdown file
        raw_output_path: Path for raw markdown file
        images_output_dir: Directory for extracted images
        use_gpu: Whether to use GPU
        table_mode: Table extraction mode
        images_scale: Image scaling factor
        do_cell_matching: Enable cell matching
        ocr_confidence_threshold: OCR threshold
        add_page_numbers: Whether to add page numbers

    Returns:
        Tuple of (input_path, output_path, success, error_message, image_count, page_count, processing_time)
    """
    import time
    import os
    import subprocess
    from pathlib import Path

    # Set up GPU memory limits BEFORE importing PyTorch/Docling
    if use_gpu:
        from src.gpu_memory_manager import setup_gpu_memory_limits
        # Calculate memory fraction as 0.8/batch_size for dynamic allocation
        memory_fraction = 0.8 / batch_size
        setup_gpu_memory_limits(memory_fraction=memory_fraction, max_split_size_mb=128)
    else:
        # Force CPU if use_gpu is False by hiding GPUs from this process
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    pdf_file = Path(pdf_file_path)

    # Check if file is actually a PDF using the `file` command
    try:
        result = subprocess.run(
            ['file', '--mime-type', '-b', str(pdf_file)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        mime_type = result.stdout.strip()

        # Skip text files - they're not real PDFs
        if mime_type.startswith('text/'):
            error_msg = f"Skipping {pdf_file.name} - detected as text file, not a PDF"
            # DELETE THE FILE IMMEDIATELY - IT'S THE WRONG TYPE
            try:
                pdf_file.unlink()
                error_msg += " [DELETED]"
            except Exception as e:
                error_msg += f" [DELETE FAILED: {e}]"
            return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)

        # Skip non-PDF files
        if not mime_type.startswith('application/pdf'):
            error_msg = f"Skipping {pdf_file.name} - not a PDF (detected as {mime_type})"
            # DELETE THE FILE IMMEDIATELY - IT'S THE WRONG TYPE
            try:
                pdf_file.unlink()
                error_msg += " [DELETED]"
            except Exception as e:
                error_msg += f" [DELETE FAILED: {e}]"
            return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)

    except subprocess.TimeoutExpired:
        error_msg = f"Timeout detecting file type for {pdf_file.name}"
        return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)
    except Exception as e:
        # If file type detection fails, try to proceed as PDF anyway
        pass

    # Import GPU memory manager for retry logic
    from src.gpu_memory_manager import (
        set_process_gpu_memory_fraction,
        clear_gpu_cache,
        wait_for_gpu_memory,
        is_oom_error,
        log_gpu_memory_stats,
    )

    # Set per-process memory fraction after importing PyTorch (via DoclingConverter)
    max_retries = 3
    retry_count = 0

    try:
        # Create converter instance for this process
        converter = DoclingConverter(
            artifacts_path=None,
            add_page_numbers=add_page_numbers,
            use_gpu=use_gpu,
            table_mode=table_mode,
            images_scale=images_scale,
            do_cell_matching=do_cell_matching,
            ocr_confidence_threshold=ocr_confidence_threshold,
        )

        # Set GPU memory fraction after converter is initialized
        if use_gpu:
            memory_fraction = 0.8 / batch_size
            set_process_gpu_memory_fraction(memory_fraction)
            log_gpu_memory_stats()

        try:
            doc_id = pdf_file.stem
            if doc_id.startswith('doc_'):
                doc_id = doc_id[4:]

            # Prepare chunking if needed
            chunk_paths: List[Path] = [pdf_file]
            page_offsets: List[int] = [0]
            chunked = False
            try:
                parsed_chunk_limit = int(chunk_page_limit)
            except (TypeError, ValueError):
                parsed_chunk_limit = 0
            chunk_page_limit = max(0, parsed_chunk_limit)

            chunk_temp_dir: Optional[Path] = None

            if chunk_page_limit > 0:
                try:
                    from pypdf import PdfReader, PdfWriter

                    reader = PdfReader(str(pdf_file))
                    total_pages = len(reader.pages)
                    if total_pages > chunk_page_limit:
                        chunked = True
                        chunk_paths = []
                        page_offsets = []
                        chunk_temp_dir = Path(tempfile.mkdtemp(prefix="pdf_chunks_"))

                        for start in range(0, total_pages, chunk_page_limit):
                            writer = PdfWriter()
                            end = min(start + chunk_page_limit, total_pages)
                            for page_idx in range(start, end):
                                writer.add_page(reader.pages[page_idx])

                            chunk_file = chunk_temp_dir / f"{pdf_file.stem}_part_{len(chunk_paths) + 1}.pdf"
                            with open(chunk_file, "wb") as chunk_fp:
                                writer.write(chunk_fp)

                            chunk_paths.append(chunk_file)
                            page_offsets.append(start)
                except ImportError:
                    error_msg = f"pypdf not available; cannot chunk {pdf_file.name} (chunking required)"
                    return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)
                except Exception as chunk_error:
                    error_msg = f"Failed to chunk {pdf_file.name} ({chunk_error}); chunking required but failed"
                    return (pdf_file_path, "", False, error_msg, 0, 0, 0.0)

            combined_markdown_parts: List[str] = []
            total_page_count = 0
            total_image_count = 0
            total_processing_time = 0.0

            for idx, chunk_path in enumerate(chunk_paths):
                page_offset = page_offsets[idx]
                chunk_label = f"part {idx + 1}" if chunked else "full document"

                # Retry loop for OOM errors
                retry_count = 0
                while retry_count <= max_retries:
                    try:
                        # Wait for GPU memory to be available before processing
                        if use_gpu and retry_count > 0:
                            clear_gpu_cache()
                            if not wait_for_gpu_memory(required_mb=2000, timeout_seconds=180):
                                raise RuntimeError("Timeout waiting for GPU memory")

                        start_time = time.time()
                        markdown, document, page_count = converter.convert_pdf(
                            chunk_path,
                            page_offset=page_offset,
                        )
                        chunk_processing_time = time.time() - start_time

                        image_count = converter.extract_images(
                            document,
                            Path(images_output_dir),
                            doc_id,
                            page_offset=page_offset,
                        )

                        total_processing_time += chunk_processing_time
                        total_page_count += page_count
                        total_image_count += image_count

                        # Success - break retry loop
                        # Clear GPU cache after each chunk to free memory
                        if use_gpu:
                            clear_gpu_cache()

                        break

                    except Exception as chunk_error:
                        if is_oom_error(chunk_error):
                            retry_count += 1
                            if retry_count <= max_retries:
                                logging.warning(
                                    f"GPU OOM error processing {pdf_file.name} {chunk_label}, "
                                    f"retry {retry_count}/{max_retries}: {chunk_error}"
                                )
                                clear_gpu_cache()
                                time.sleep(5 * retry_count)  # Exponential backoff
                                continue
                            else:
                                logging.error(
                                    f"Failed after {max_retries} retries due to GPU OOM: {pdf_file.name}"
                                )
                                raise RuntimeError(
                                    f"GPU out of memory after {max_retries} retries"
                                ) from chunk_error
                        else:
                            # Non-OOM error, don't retry
                            raise

                if chunked:
                    combined_markdown_parts.append(
                        f"\n<!-- CHUNK START: {chunk_label} (pages {page_offset + 1}-{page_offset + page_count}) -->\n\n"
                    )

                combined_markdown_parts.append(markdown)

                # Explicitly delete large objects to free memory after chunk processing
                del markdown, document

                # Force garbage collection for chunked documents to reclaim memory
                if chunked:
                    import gc
                    gc.collect()
                    logging.debug(f"  Memory cleanup after {chunk_label}")

                if chunked:
                    combined_markdown_parts.append(
                        f"\n<!-- CHUNK END: {chunk_label} -->\n"
                    )

            combined_markdown = "".join(combined_markdown_parts)

            if chunk_temp_dir and chunk_temp_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(chunk_temp_dir, ignore_errors=True)
                except Exception:
                    pass

            # Add metadata header
            metadata_lines = [
                "---",
                f"**Document:** {pdf_file.name}",
                f"**Pages:** {total_page_count}",
                f"**Images Extracted:** {total_image_count}",
                f"**Images Location:** data/images/{doc_id}/",
                f"**Processing Time:** {total_processing_time:.2f} seconds",
            ]
            if chunked:
                metadata_lines.append(f"**Chunks:** {len(chunk_paths)} (limit {chunk_page_limit} pages)")
            metadata_lines.append(f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            metadata_lines.append("---\n\n")

            metadata_header = "\n".join(metadata_lines)
            raw_markdown = metadata_header + combined_markdown

            # Clean repeating characters
            cleaned_markdown = clean_repeating_characters(raw_markdown)

            # Wrap long lines (ensure no line exceeds 10,000 characters)
            cleaned_markdown = wrap_long_lines(cleaned_markdown, max_line_length=10000)

            # Save raw version
            Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(raw_markdown)

            # Save cleaned version
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_markdown)

            return (
                pdf_file_path,
                output_path,
                True,
                "",
                total_image_count,
                total_page_count,
                total_processing_time,
            )

        finally:
            converter.cleanup()
            # Clear GPU cache to ensure memory is freed
            if use_gpu:
                clear_gpu_cache()
                log_gpu_memory_stats()

    except Exception as e:
        error_msg = f"Failed to convert {pdf_file.name}: {str(e)}"
        # Clear GPU cache even on error
        if use_gpu:
            clear_gpu_cache()
        return (pdf_file_path, output_path, False, error_msg, 0, 0, 0.0)


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
        batch_size: int = 2,
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
        chunk_page_limit: int = 50,
        max_docs: Optional[int] = None,
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
        chunk_page_limit: Maximum pages per chunk before splitting large PDFs (default: 50; 0 disables chunking).
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
        self.chunk_page_limit = chunk_page_limit

        # Maximum number of docs to process
        self.max_docs = max_docs

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
            "skipped_has_both_reps": 0,
            "total_images_extracted": 0,
        }

        # Initialize processing logger
        self.proc_logger = ProcessingLogger()

        # Supabase database connection parameters for checking existing representations
        self.supabase_db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        # Cache for existing representations in Supabase
        self.existing_representations: Dict[int, Set[str]] = {}

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

    async def _fetch_existing_representations_from_supabase(
        self, document_ids: Optional[List[int]] = None
    ) -> Dict[int, Set[str]]:
        """
        Fetch existing document representations from Supabase with pagination.

        Args:
            document_ids: Optional list of document IDs to check. If None, checks all.

        Returns:
            Dict mapping document_id to set of representation types (e.g., {'DOCLING', 'DOCLING_IMG'})
        """
        self.logger.info("Checking Supabase for existing document representations...")

        try:
            # Create asyncpg connection pool
            pool = await asyncpg.create_pool(**self.supabase_db_config)

            try:
                page_size = 1000
                offset = 0
                existing = {}

                async with pool.acquire() as conn:
                    while True:
                        # Query with pagination
                        if document_ids:
                            query = """
                                SELECT kdocument_id, representation_type::text
                                FROM librarian.document_locations_v2
                                WHERE kdocument_id = ANY($1)
                                AND representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                                ORDER BY kdocument_id
                                LIMIT $2 OFFSET $3
                            """
                            rows = await conn.fetch(query, document_ids, page_size, offset)
                        else:
                            query = """
                                SELECT kdocument_id, representation_type::text
                                FROM librarian.document_locations_v2
                                WHERE representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                                ORDER BY kdocument_id
                                LIMIT $1 OFFSET $2
                            """
                            rows = await conn.fetch(query, page_size, offset)

                        # Break if no more results
                        if not rows:
                            break

                        # Process results
                        for row in rows:
                            doc_id = row["kdocument_id"]
                            rep_type = row["representation_type"]
                            if doc_id not in existing:
                                existing[doc_id] = set()
                            existing[doc_id].add(rep_type)

                        # Move to next page
                        offset += page_size
                        self.logger.debug(f"Fetched {len(rows)} representations (offset: {offset})")

                        # If we got fewer results than page_size, we're done
                        if len(rows) < page_size:
                            break

                self.logger.info(
                    f"Found {len(existing)} documents with existing representations in Supabase"
                )
                return existing

            finally:
                await pool.close()

        except Exception as e:
            self.logger.error(f"Failed to check Supabase for existing representations: {e}")
            # Return empty dict on error - we'll proceed without filtering
            return {}

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

    def _has_both_representations(self, pdf_file: Path) -> bool:
        """
        Check if a document already has both DOCLING and DOCLING_IMG representations in Supabase.

        Args:
            pdf_file: Path to PDF file

        Returns:
            True if document has both representations
        """
        doc_id_str = self._extract_doc_id_from_filename(pdf_file)
        if doc_id_str is None:
            return False

        try:
            doc_id = int(doc_id_str)
            if doc_id in self.existing_representations:
                reps = self.existing_representations[doc_id]
                return "DOCLING" in reps and "DOCLING_IMG" in reps
        except ValueError:
            pass

        return False

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

    def _convert_single_file_sync(self, pdf_file: Path) -> Tuple[Path, Path, bool, str, int, int, float]:
        """
        Convert a single PDF file to Markdown (synchronous version for process pool).

        Args:
            pdf_file: Path to the PDF file to convert.

        Returns:
            Tuple of (input_path, output_path, success, error_message, image_count, page_count, processing_time)
        """
        output_path = self._get_output_path(pdf_file)

        # Check if already processed
        if self._is_already_processed(pdf_file):
            self.logger.info("Skipping %s - already processed", pdf_file.name)
            return pdf_file, output_path, True, "Skipped - already processed", 0, 0, 0.0

        # Skip if output file already exists
        if output_path.exists():
            self.logger.info("Skipping %s - output already exists", pdf_file.name)
            return pdf_file, output_path, True, "Skipped - output already exists", 0, 0, 0.0

        # Prepare paths for worker
        raw_output_path = output_path.parent.parent / 'data' / 'processed_raw' / output_path.name
        images_output_dir = 'data/images'

        # Call worker function
        result = _process_single_pdf_worker(
            pdf_file_path=str(pdf_file),
            output_path=str(output_path),
            raw_output_path=str(raw_output_path),
            images_output_dir=images_output_dir,
            use_gpu=self.use_gpu,
            table_mode=self.table_mode,
            images_scale=self.images_scale,
            do_cell_matching=self.do_cell_matching,
            ocr_confidence_threshold=self.ocr_confidence_threshold,
            add_page_numbers=self.add_page_numbers,
            chunk_page_limit=self.chunk_page_limit,
            batch_size=self.batch_size,
        )

        # Unpack result
        input_path_str, output_path_str, success, error_msg, image_count, page_count, processing_time = result

        if success and not error_msg:
            # Log conversion to processing log
            doc_id = pdf_file.stem
            self.proc_logger.log_conversion(
                doc_id=doc_id,
                pages=page_count,
                duration_seconds=processing_time,
                status="success"
            )

            self.logger.info(
                "Successfully converted %s (%d pages, %d images in %.2f seconds)",
                pdf_file.name,
                page_count,
                image_count,
                processing_time
            )

        return Path(input_path_str), Path(output_path_str), success, error_msg, image_count, page_count, processing_time

    def _process_batch_sequential(
        self, pdf_files: List[Path]
    ) -> List[Tuple[Path, Path, bool, str, int, int, float]]:
        """
        Process PDF files sequentially (used when batch_size=1 to avoid pooling overhead).

        Args:
            pdf_files: List of PDF files to process.

        Returns:
            List of tuples containing (input_path, output_path, success, error_message, image_count, page_count, processing_time).
        """
        results = []

        for pdf_file in pdf_files:
            output_path = self._get_output_path(pdf_file)
            raw_output_path = output_path.parent.parent / 'data' / 'processed_raw' / output_path.name

            # Skip if already processed or exists
            if self._is_already_processed(pdf_file):
                self.logger.info("Skipping %s - already processed", pdf_file.name)
                results.append((pdf_file, output_path, True, "Skipped - already processed", 0, 0, 0.0))
                continue

            if output_path.exists():
                self.logger.info("Skipping %s - output already exists", pdf_file.name)
                # Delete the PDF since output already exists
                if self.remove_processed and pdf_file.exists():
                    try:
                        pdf_file.unlink()
                        self.stats["removed_files"] += 1
                        self.logger.info("Deleted skipped file: %s", pdf_file.name)
                    except OSError as e:
                        self.logger.warning("Failed to delete skipped file %s: %s", pdf_file.name, e)
                results.append((pdf_file, output_path, True, "Skipped - output already exists", 0, 0, 0.0))
                continue

            # Process directly without worker process overhead
            try:
                result = _process_single_pdf_worker(
                    str(pdf_file),
                    str(output_path),
                    str(raw_output_path),
                    'data/images',
                    self.use_gpu,
                    self.table_mode,
                    self.images_scale,
                    self.do_cell_matching,
                    self.ocr_confidence_threshold,
                    self.add_page_numbers,
                    self.chunk_page_limit,
                    self.batch_size,
                )

                input_path_str, output_path_str, success, error_msg, image_count, page_count, processing_time = result

                if success and not error_msg:
                    doc_id = pdf_file.stem
                    self.proc_logger.log_conversion(
                        doc_id=doc_id,
                        pages=page_count,
                        duration_seconds=processing_time,
                        status="success"
                    )
                    self.logger.info(
                        "Successfully converted %s (%d pages, %d images in %.2f seconds)",
                        pdf_file.name,
                        page_count,
                        image_count,
                        processing_time
                    )
                else:
                    self.logger.error("Failed to convert %s: %s", pdf_file.name, error_msg)

                results.append((
                    Path(input_path_str),
                    Path(output_path_str),
                    success,
                    error_msg,
                    image_count,
                    page_count,
                    processing_time
                ))

            except Exception as e:
                error_msg = f"Processing failed: {str(e)}"
                self.logger.error("Error processing %s: %s", pdf_file.name, error_msg)
                results.append((
                    pdf_file,
                    output_path,
                    False,
                    error_msg,
                    0,
                    0,
                    0.0
                ))

        return results

    def _process_batch_parallel(
        self, pdf_files: List[Path], max_workers: Optional[int] = None
    ) -> List[Tuple[Path, Path, bool, str, int, int, float]]:
        """
        Process a batch of PDF files in parallel using ProcessPoolExecutor.

        Args:
            pdf_files: List of PDF files to process.
            max_workers: Maximum number of worker processes (defaults to batch_size).

        Returns:
            List of tuples containing (input_path, output_path, success, error_message, image_count, page_count, processing_time).
        """
        global _shutdown_requested

        if max_workers is None:
            max_workers = self.batch_size

        results = []

        # Use ProcessPoolExecutor for true parallelism
        spawn_ctx = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=spawn_ctx,
        ) as executor:
            # Submit all tasks
            future_to_pdf = {}
            for pdf_file in pdf_files:
                # Check for shutdown before submitting new tasks
                if _shutdown_requested:
                    self.logger.warning("Shutdown requested - not submitting new tasks")
                    break

                output_path = self._get_output_path(pdf_file)
                raw_output_path = output_path.parent.parent / 'data' / 'processed_raw' / output_path.name

                # Skip if already processed or exists
                if self._is_already_processed(pdf_file):
                    self.logger.info("Skipping %s - already processed", pdf_file.name)
                    results.append((pdf_file, output_path, True, "Skipped - already processed", 0, 0, 0.0))
                    continue

                if output_path.exists():
                    self.logger.info("Skipping %s - output already exists", pdf_file.name)
                    # Delete the PDF since output already exists
                    if self.remove_processed and pdf_file.exists():
                        try:
                            pdf_file.unlink()
                            self.stats["removed_files"] += 1
                            self.logger.info("Deleted skipped file: %s", pdf_file.name)
                        except OSError as e:
                            self.logger.warning("Failed to delete skipped file %s: %s", pdf_file.name, e)
                    results.append((pdf_file, output_path, True, "Skipped - output already exists", 0, 0, 0.0))
                    continue

                # Submit job to executor
                future = executor.submit(
                    _process_single_pdf_worker,
                    str(pdf_file),
                    str(output_path),
                    str(raw_output_path),
                    'data/images',
                    self.use_gpu,
                    self.table_mode,
                    self.images_scale,
                    self.do_cell_matching,
                    self.ocr_confidence_threshold,
                    self.add_page_numbers,
                    self.chunk_page_limit,
                    self.batch_size,
                )
                future_to_pdf[future] = pdf_file

            # Collect results as they complete
            for future in as_completed(future_to_pdf):
                # Check for shutdown request
                if _shutdown_requested:
                    self.logger.warning("Shutdown requested - cancelling remaining tasks")
                    # Cancel any pending futures
                    for f in future_to_pdf:
                        if not f.done():
                            f.cancel()
                    break

                pdf_file = future_to_pdf[future]
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per file
                    input_path_str, output_path_str, success, error_msg, image_count, page_count, processing_time = result

                    if success and not error_msg:
                        doc_id = pdf_file.stem
                        self.proc_logger.log_conversion(
                            doc_id=doc_id,
                            pages=page_count,
                            duration_seconds=processing_time,
                            status="success"
                        )
                        self.logger.info(
                            "Successfully converted %s (%d pages, %d images in %.2f seconds)",
                            pdf_file.name,
                            page_count,
                            image_count,
                            processing_time
                        )
                    else:
                        self.logger.error("Failed to convert %s: %s", pdf_file.name, error_msg)

                    results.append((
                        Path(input_path_str),
                        Path(output_path_str),
                        success,
                        error_msg,
                        image_count,
                        page_count,
                        processing_time
                    ))

                except Exception as e:
                    error_msg = f"Worker process failed: {str(e)}"
                    self.logger.error("Error processing %s: %s", pdf_file.name, error_msg)
                    results.append((
                        pdf_file,
                        self._get_output_path(pdf_file),
                        False,
                        error_msg,
                        0,
                        0,
                        0.0
                    ))

        return results

    async def convert_all(self) -> Dict[str, int]:
        """
        Convert all PDF files in the input folder to Markdown using parallel processing.

        Returns:
            Dictionary containing conversion statistics.
        """
        self.logger.info(
            "Starting batch conversion from %s to %s (max workers: %d)",
            self.input_folder,
            self.output_folder,
            self.batch_size,
        )

        # Get all PDF files
        pdf_files = self._get_pdf_files()
        self.stats["total_files"] = len(pdf_files)

        if not pdf_files:
            self.logger.warning("No PDF files found in %s", self.input_folder)
            return self.stats

        self.logger.info("Found %d PDF files to convert", len(pdf_files))

        # Check Supabase for existing representations
        document_ids = []
        for pdf_file in pdf_files:
            doc_id_str = self._extract_doc_id_from_filename(pdf_file)
            if doc_id_str:
                try:
                    document_ids.append(int(doc_id_str))
                except ValueError:
                    pass

        if document_ids:
            self.existing_representations = await self._fetch_existing_representations_from_supabase(
                document_ids
            )

            # Filter out files that already have both representations
            original_count = len(pdf_files)
            pdf_files_filtered = []
            for pdf_file in pdf_files:
                if self._has_both_representations(pdf_file):
                    self.stats["skipped_has_both_reps"] += 1
                    self.logger.info(
                        "Skipping %s - both representations already exist in Supabase",
                        pdf_file.name
                    )
                else:
                    pdf_files_filtered.append(pdf_file)

            pdf_files = pdf_files_filtered
            if self.stats["skipped_has_both_reps"] > 0:
                self.logger.info(
                    "Skipped %d files with both representations already in Supabase",
                    self.stats["skipped_has_both_reps"]
                )
                self.logger.info("Processing %d remaining files", len(pdf_files))

        # Limit to max_docs if specified
        if self.max_docs is not None and len(pdf_files) > self.max_docs:
            self.logger.info(
                "Limiting to %d documents (total available: %d)",
                self.max_docs,
                len(pdf_files)
            )
            pdf_files = pdf_files[:self.max_docs]

        if not pdf_files:
            self.logger.info("All files already have both representations - nothing to do!")
            return self.stats

        # Fast path for batch_size=1: process sequentially without async/pooling overhead
        if self.batch_size == 1:
            self.logger.info("Processing files sequentially (batch_size=1 - no pooling overhead)...")
            results = self._process_batch_sequential(pdf_files)
        else:
            # Process all files in parallel using ProcessPoolExecutor
            self.logger.info("Processing files with %d parallel workers...", self.batch_size)
            results = self._process_batch_parallel(pdf_files, max_workers=self.batch_size)

        # Update statistics, upload files, and remove successfully processed files
        # Note: We need to run upload in async context if enabled
        upload_tasks = []

        oom_signatures = (
            "cuda out of memory",
            "cuda error: out of memory",
            "cublas error: an illegal memory access was encountered",
            "out of memory on device",
            "hip out of memory",
            "resource exhausted",
        )

        for input_path, _output_path, success, error_msg, image_count, _page_count, _processing_time in results:
            # Handle files that were rejected during file type detection (empty output_path)
            # These should be moved and marked as processed to avoid rechecking
            if not _output_path or str(_output_path) == "" or _output_path == Path(""):
                self.stats["failed_files"] += 1
                self.logger.warning("Rejected non-PDF file: %s - %s", input_path.name, error_msg)

                # Mark as processed so we don't check it again
                self._mark_as_processed(input_path)

                # Delete the rejected file to avoid rechecking (always delete invalid files)
                if input_path.exists():
                    try:
                        input_path.unlink()
                        self.stats["removed_files"] += 1
                        self.logger.info("Deleted rejected non-PDF file: %s", input_path.name)
                    except OSError as e:
                        self.logger.warning("Failed to delete rejected file %s: %s", input_path.name, e)
                continue

            # Track image count
            if success and image_count > 0:
                self.stats["total_images_extracted"] += image_count

            if success:
                if "Skipped - already processed" in error_msg:
                    self.stats["skipped_already_processed"] += 1
                elif "Skipped" in error_msg:
                    self.stats["skipped_files"] += 1
                else:
                    # Actually processed (not skipped)
                    self.stats["processed_files"] += 1

                    # Mark as processed
                    self._mark_as_processed(input_path)

                    # Handle upload and deletion
                    if self.upload_enabled:
                        # Add to upload queue - will be deleted after upload
                        upload_tasks.append((input_path, _output_path))
                    else:
                        # No upload - delete immediately if configured
                        if self.remove_processed and input_path.exists():
                            try:
                                input_path.unlink()
                                self.stats["removed_files"] += 1
                                self.logger.info("Deleted processed file: %s", input_path.name)
                            except OSError as e:
                                self.logger.warning(
                                    "Failed to delete processed file %s: %s",
                                    input_path.name,
                                    e
                                )
            else:
                # Conversion failed
                self.stats["failed_files"] += 1

                # If we ran out of GPU memory, leave the PDF in place and pause briefly
                lowered_error = (error_msg or "").lower()
                if any(signature in lowered_error for signature in oom_signatures):
                    self.logger.warning(
                        "GPU memory exhausted while processing %s; leaving PDF for retry and pausing 10 seconds",
                        input_path.name,
                    )
                    await asyncio.sleep(10)
                    continue


        # Handle uploads asynchronously if needed
        if upload_tasks:
            for input_path, _output_path in upload_tasks:
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

                # Delete after upload attempt (only if upload succeeded and removal is enabled)
                should_delete = self.remove_processed and upload_success
                if should_delete and input_path.exists():
                    try:
                        input_path.unlink()
                        self.stats["removed_files"] += 1
                        self.logger.info("Deleted processed file after upload: %s", input_path.name)
                    except OSError as e:
                        self.logger.warning(
                            "Failed to delete processed file %s: %s",
                            input_path.name,
                            e
                        )

        # Log final statistics
        self.logger.info("Conversion completed!")
        self.logger.info("Total files: %d", self.stats["total_files"])
        self.logger.info("Processed: %d", self.stats["processed_files"])
        self.logger.info("Skipped (has both reps in Supabase): %d", self.stats["skipped_has_both_reps"])
        self.logger.info("Skipped (already processed): %d", self.stats["skipped_already_processed"])
        self.logger.info("Skipped (output exists): %d", self.stats["skipped_files"])
        self.logger.info("Failed: %d", self.stats["failed_files"])
        self.logger.info("Total images extracted: %d", self.stats["total_images_extracted"])
        if self.upload_enabled:
            self.logger.info("Uploaded: %d", self.stats["uploaded_files"])
            self.logger.info("Upload failed: %d", self.stats["upload_failed_files"])
        if self.remove_processed:
            self.logger.info("Deleted: %d", self.stats["removed_files"])

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
    batch_size: int = 2,
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
    chunk_page_limit: int = 50,
    max_docs: Optional[int] = None,
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
        chunk_page_limit: Maximum pages processed per chunk (default: 50; 0 disables chunking).
        max_docs: Maximum number of PDFs to process in this run (default: None = process all).

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
        chunk_page_limit=chunk_page_limit,
        max_docs=max_docs,
    )

    try:
        stats = await converter.convert_all()
        return stats
    finally:
        converter.cleanup()


def main():
    """Command-line interface for the batch converter."""
    import argparse

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

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
        default=2,
        help="Number of files to process concurrently (default: 2)",
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
        help="Extract and save images from PDFs to data/processed_images/ folder",
    )
    parser.add_argument(
        "--chunk-page-limit",
        type=int,
        default=30,
        help="Split PDFs into chunks with at most this many pages before processing (0 disables chunking)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of PDFs to process in this run (default: process all available PDFs)",
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


if __name__ == "__main__":
    main()
