"""
Test script to download and process 10 slides documents with maximum quality.
"""

import asyncio
import os
from pathlib import Path
from document_metadata import MetadataManager
from fetch_documents import DocumentFetcher
from batch_docling_converter import BatchDoclingConverter
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_10_slides():
    """Download and process 10 slides documents."""

    logger.info("=" * 60)
    logger.info("Testing 10 slides documents with MAXIMUM QUALITY")
    logger.info("=" * 60)

    # Initialize fetcher
    fetcher = DocumentFetcher("config.yaml")
    metadata = fetcher.metadata

    # Step 1: Get pending slides documents
    logger.info("\n[Step 1] Looking for slides documents in database...")
    pending = metadata.get_pending_downloads()

    # Filter for slides only
    slides_docs = [doc for doc in pending if doc.get("document_type") == "slides"]

    if len(slides_docs) < 10:
        logger.warning(f"Only found {len(slides_docs)} slides documents in pending")
        logger.info("Fetching more documents from API...")
        await fetcher.fetch_all(download_pdfs=False)

        # Try again after fetching
        pending = metadata.get_pending_downloads()
        slides_docs = [doc for doc in pending if doc.get("document_type") == "slides"]

    if not slides_docs:
        logger.error("No slides documents found")
        return

    # Get first 10 slides
    slides_to_process = slides_docs[:10]
    logger.info(f"\nFound {len(slides_docs)} total slides documents")
    logger.info(f"Will process {len(slides_to_process)} slides documents\n")

    # Show what we'll process
    for i, doc in enumerate(slides_to_process, 1):
        logger.info(f"{i}. {doc['company_name']} ({doc['ticker']})")
        logger.info(f"   Title: {doc['title'] or 'N/A'}")
        logger.info(f"   ID: {doc['document_id']}")

    # Step 2: Download the slides
    logger.info(f"\n[Step 2] Downloading {len(slides_to_process)} PDFs...")
    import aiohttp

    async with aiohttp.ClientSession() as session:
        await fetcher._download_documents_batch(session, slides_to_process)

    # Check download status
    downloaded_count = 0
    failed_count = 0
    for doc in slides_to_process:
        downloaded = metadata.get_document_by_id(doc['document_id'])
        if downloaded['download_status'] == 'downloaded':
            downloaded_count += 1
        else:
            failed_count += 1
            logger.warning(f"Failed to download doc {doc['document_id']}: {downloaded.get('error_message', 'Unknown')}")

    logger.info(f"\nDownload Summary:")
    logger.info(f"  Successfully downloaded: {downloaded_count}")
    logger.info(f"  Failed: {failed_count}")

    if downloaded_count == 0:
        logger.error("No documents downloaded successfully")
        return

    # Step 3: Convert PDFs to markdown with MAXIMUM QUALITY
    logger.info(f"\n[Step 3] Converting {downloaded_count} PDFs to Markdown...")
    logger.info("Quality Settings:")
    logger.info("  - Table mode: accurate (highest quality)")
    logger.info("  - Image scale: 3.0 (maximum quality)")
    logger.info("  - Cell matching: enabled (precise tables)")
    logger.info("  - OCR confidence: 0.05 (capture more text)")
    logger.info("  - GPU: disabled (using CPU)")
    logger.info("")

    converter = BatchDoclingConverter(
        input_folder="to_process",
        output_folder="processed",
        batch_size=1,  # Process one at a time for stability
        remove_processed=False,  # Keep files for inspection
        use_gpu=False,  # CPU mode
        log_level=logging.INFO,
        doc_type="both",  # Process all PDFs (they're all slides we just downloaded)
        # Maximum quality defaults will be used automatically:
        # table_mode="accurate", images_scale=3.0, do_cell_matching=True, ocr_confidence_threshold=0.05
    )

    try:
        stats = await converter.convert_all()

        logger.info("\n" + "=" * 60)
        logger.info("CONVERSION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Processed: {stats['processed_files']}")
        logger.info(f"Skipped (already existed): {stats['skipped_files']}")
        logger.info(f"Failed: {stats['failed_files']}")

        if stats['processed_files'] > 0:
            logger.info("\n✓ SUCCESS! Documents processed with maximum quality")
            logger.info(f"\nOutput location: processed/")

            # List converted files
            logger.info("\nConverted markdown files:")
            processed_folder = Path("processed")
            md_files = sorted(processed_folder.glob("doc_*.md"))
            for md_file in md_files[-stats['processed_files']:]:
                size_kb = md_file.stat().st_size / 1024
                logger.info(f"  - {md_file.name} ({size_kb:.1f} KB)")

    finally:
        converter.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_10_slides())
