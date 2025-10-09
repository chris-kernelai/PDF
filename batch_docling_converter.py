"""
Batch Docling PDF Converter

This module provides batch processing capabilities for converting multiple PDF files
to Markdown using Docling. It processes files asynchronously in configurable batches.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from docling_converter import DoclingConverter


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
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.batch_size = batch_size
        # Don't use custom artifacts_path by default - let Docling manage models automatically
        self.artifacts_path = artifacts_path
        self.add_page_numbers = add_page_numbers
        self.remove_processed = remove_processed
        self.use_gpu = use_gpu

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
        }

    def _get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the input folder."""
        pdf_files = []
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

    async def _convert_single_file(self, pdf_file: Path) -> Tuple[Path, Path, bool, str]:
        """
        Convert a single PDF file to Markdown.

        Args:
            pdf_file: Path to the PDF file to convert.

        Returns:
            Tuple of (input_path, output_path, success, error_message)
        """
        output_path = self._get_output_path(pdf_file)

        # Skip if output file already exists
        if output_path.exists():
            self.logger.info("Skipping %s - output already exists", pdf_file.name)
            return pdf_file, output_path, True, "Skipped - output already exists"

        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert the file
            converter = DoclingConverter(
                artifacts_path=self.artifacts_path,
                add_page_numbers=self.add_page_numbers,
                use_gpu=self.use_gpu,
            )

            try:
                markdown = converter.convert_pdf_to_markdown(pdf_file)

                # Write the markdown to output file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(markdown)

                self.logger.info("Successfully converted %s", pdf_file.name)
                return pdf_file, output_path, True, ""

            finally:
                converter.cleanup()

        except (FileNotFoundError, ValueError, OSError) as e:
            error_msg = f"Failed to convert {pdf_file.name}: {str(e)}"
            self.logger.error("%s", error_msg)
            return pdf_file, output_path, False, error_msg

    async def _process_batch(
        self, pdf_files: List[Path]
    ) -> List[Tuple[Path, Path, bool, str]]:
        """
        Process a batch of PDF files concurrently.

        Args:
            pdf_files: List of PDF files to process.

        Returns:
            List of tuples containing (input_path, output_path, success, error_message) for each file.
        """
        tasks = [self._convert_single_file(pdf_file) for pdf_file in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_results: List[Tuple[Path, Path, bool, str]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Task failed with exception: {str(result)}"
                self.logger.error(
                    "Batch processing error for %s: %s", pdf_files[i].name, error_msg
                )
                processed_results.append(
                    (pdf_files[i], self._get_output_path(pdf_files[i]), False, error_msg)
                )
            elif isinstance(result, tuple) and len(result) == 4:
                processed_results.append(result)
            else:
                # Handle unexpected result type
                error_msg = f"Unexpected result type: {type(result)}"
                self.logger.error(
                    "Unexpected result for %s: %s", pdf_files[i].name, error_msg
                )
                processed_results.append(
                    (pdf_files[i], self._get_output_path(pdf_files[i]), False, error_msg)
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

            # Update statistics and remove successfully processed files
            for input_path, _output_path, success, error_msg in results:
                if success:
                    if "Skipped" in error_msg:
                        self.stats["skipped_files"] += 1
                    else:
                        self.stats["processed_files"] += 1

                        # Remove the successfully processed PDF if enabled
                        if self.remove_processed:
                            try:
                                input_path.unlink()
                                self.stats["removed_files"] += 1
                                self.logger.info("Removed processed file: %s", input_path.name)
                            except OSError as e:
                                self.logger.warning(
                                    "Failed to remove processed file %s: %s",
                                    input_path.name,
                                    e
                                )
                else:
                    self.stats["failed_files"] += 1

        # Log final statistics
        self.logger.info("Conversion completed!")
        self.logger.info("Total files: %d", self.stats["total_files"])
        self.logger.info("Processed: %d", self.stats["processed_files"])
        self.logger.info("Skipped: %d", self.stats["skipped_files"])
        self.logger.info("Failed: %d", self.stats["failed_files"])
        if self.remove_processed:
            self.logger.info("Removed: %d", self.stats["removed_files"])

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
        )
    )

    print("\nConversion completed!")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed_files']}")
    print(f"Skipped: {stats['skipped_files']}")
    print(f"Failed: {stats['failed_files']}")
    if not args.keep_processed:
        print(f"Removed: {stats['removed_files']}")


if __name__ == "__main__":
    main()
