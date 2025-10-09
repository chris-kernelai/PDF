"""
Full Pipeline Runner

Orchestrates the complete workflow: fetch documents -> convert to markdown.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict

from fetch_documents import DocumentFetcher
from batch_docling_converter import BatchDoclingConverter
from document_metadata import MetadataManager


class PipelineRunner:
    """Orchestrates the full document processing pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline runner.

        Args:
            config_path: Path to configuration file.
        """
        self.fetcher = DocumentFetcher(config_path)
        self.config = self.fetcher.config
        self.metadata = self.fetcher.metadata

        self.logger = logging.getLogger(__name__)

    async def run_full_pipeline(self):
        """
        Run the complete pipeline:
        1. Fetch documents from API
        2. Download PDFs to to_process/
        3. Convert PDFs to markdown
        4. Remove successfully processed PDFs
        5. Update metadata
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Full Pipeline")
        self.logger.info("=" * 60)

        # Step 1: Fetch and download documents
        self.logger.info("\n[Step 1/3] Fetching documents from API...")
        await self.fetcher.fetch_all(download_pdfs=True)

        # Step 2: Convert PDFs to markdown
        self.logger.info("\n[Step 2/3] Converting PDFs to markdown...")
        downloaded_docs = self.metadata.get_downloaded_documents()

        if not downloaded_docs:
            self.logger.warning("No documents to convert")
            return

        # Run batch converter
        converter = BatchDoclingConverter(
            input_folder=self.config["paths"]["input_folder"],
            output_folder=self.config["paths"]["output_folder"],
            batch_size=1,  # Default to 1 as requested
            remove_processed=True,  # Remove PDFs after successful conversion
            log_level=logging.INFO,
        )

        try:
            conversion_stats = await converter.convert_all()
        finally:
            converter.cleanup()

        # Step 3: Update metadata for converted documents
        self.logger.info("\n[Step 3/3] Updating metadata...")
        self._update_metadata_after_conversion(conversion_stats)

        # Final summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline Complete")
        self.logger.info("=" * 60)
        self._print_final_stats()

    def _update_metadata_after_conversion(self, conversion_stats: Dict):
        """
        Update metadata database with conversion results.

        Args:
            conversion_stats: Statistics from batch converter.
        """
        # Get all documents that should have been converted
        input_folder = Path(self.config["paths"]["input_folder"])
        output_folder = Path(self.config["paths"]["output_folder"])

        downloaded_docs = self.metadata.get_downloaded_documents()

        for doc in downloaded_docs:
            pdf_filename = doc["pdf_filename"]
            pdf_path = input_folder / pdf_filename

            # If PDF was removed, it was successfully converted
            if not pdf_path.exists():
                # Find corresponding markdown file
                md_filename = Path(pdf_filename).with_suffix(".md").name
                md_path = output_folder / md_filename

                if md_path.exists():
                    self.metadata.mark_converted(doc["document_id"], md_filename)
                    self.logger.info(
                        f"Marked document {doc['document_id']} as converted"
                    )
                else:
                    # PDF removed but no markdown found - something went wrong
                    self.metadata.mark_failed(
                        doc["document_id"],
                        "PDF removed but markdown not found",
                        "conversion_failed",
                    )
            else:
                # PDF still exists - conversion likely failed
                self.metadata.mark_failed(
                    doc["document_id"],
                    "PDF still exists after conversion attempt",
                    "conversion_failed",
                )

    def _print_final_stats(self):
        """Print final pipeline statistics."""
        stats = self.metadata.get_statistics()

        self.logger.info("\nFinal Statistics:")
        self.logger.info(f"  Total documents: {stats.get('total', 0)}")
        self.logger.info(f"  Pending: {stats.get('pending', 0)}")
        self.logger.info(f"  Downloaded: {stats.get('downloaded', 0)}")
        self.logger.info(f"  Converted: {stats.get('converted', 0)}")
        self.logger.info(
            f"  Failed: {stats.get('download_failed', 0) + stats.get('conversion_failed', 0) + stats.get('failed', 0)}"
        )

    async def run_fetch_only(self):
        """Only fetch documents without conversion."""
        self.logger.info("Running fetch-only mode...")
        await self.fetcher.fetch_all(download_pdfs=True)

    async def run_convert_only(self):
        """Only convert existing PDFs without fetching new ones."""
        self.logger.info("Running convert-only mode...")

        converter = BatchDoclingConverter(
            input_folder=self.config["paths"]["input_folder"],
            output_folder=self.config["paths"]["output_folder"],
            batch_size=1,
            remove_processed=True,
            log_level=logging.INFO,
        )

        try:
            conversion_stats = await converter.convert_all()
            self._update_metadata_after_conversion(conversion_stats)
        finally:
            converter.cleanup()

        self._print_final_stats()

    async def retry_failed(self):
        """Retry all failed documents."""
        self.logger.info("Retrying failed documents...")

        # Reset failed documents to pending
        self.metadata.reset_failed_documents()

        # Download pending
        await self.fetcher.download_pending()

        # Convert
        await self.run_convert_only()


async def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full document processing pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fetch-only", action="store_true", help="Only fetch documents, don't convert"
    )
    mode_group.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert existing PDFs, don't fetch new ones",
    )
    mode_group.add_argument(
        "--retry-failed", action="store_true", help="Retry all failed documents"
    )
    mode_group.add_argument(
        "--stats", action="store_true", help="Show statistics only"
    )

    args = parser.parse_args()

    pipeline = PipelineRunner(args.config)

    if args.stats:
        stats = pipeline.metadata.get_statistics()
        print("\nDocument Statistics:")
        for status, count in stats.items():
            print(f"  {status}: {count}")
        return

    if args.fetch_only:
        await pipeline.run_fetch_only()
    elif args.convert_only:
        await pipeline.run_convert_only()
    elif args.retry_failed:
        await pipeline.retry_failed()
    else:
        # Run full pipeline
        await pipeline.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
