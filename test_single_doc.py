"""
Test script to download and process a single document from the API.
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


async def test_single_document():
    """Download and process a single slides document."""

    logger.info("=" * 60)
    logger.info("Testing single document download and processing")
    logger.info("=" * 60)

    # Initialize fetcher
    fetcher = DocumentFetcher("config.yaml")

    # Step 1: Fetch documents (just metadata, no download yet)
    logger.info("\n[Step 1] Fetching document metadata from API...")
    await fetcher.fetch_all(download_pdfs=False)

    # Step 2: Find a single slides document to test
    logger.info("\n[Step 2] Looking for a slides document to test...")
    pending = fetcher.metadata.get_pending_downloads()

    # Filter for slides only
    slides_docs = [doc for doc in pending if doc.get("document_type") == "slides"]

    if not slides_docs:
        logger.error("No slides documents found in pending downloads")
        return

    # Get the first slides document
    test_doc = slides_docs[0]
    logger.info(f"Found test document:")
    logger.info(f"  ID: {test_doc['document_id']}")
    logger.info(f"  Title: {test_doc['title']}")
    logger.info(f"  Company: {test_doc['company_name']} ({test_doc['ticker']})")
    logger.info(f"  Type: {test_doc['document_type']}")
    logger.info(f"  Filing Date: {test_doc['filing_date']}")

    # Step 3: Download just this one document
    logger.info("\n[Step 3] Downloading PDF...")
    import aiohttp

    async with aiohttp.ClientSession() as session:
        await fetcher._download_documents_batch(session, [test_doc])

    # Check if download succeeded
    downloaded = fetcher.metadata.get_document_by_id(test_doc['document_id'])
    if downloaded['download_status'] != 'downloaded':
        logger.error(f"Download failed: {downloaded.get('error_message', 'Unknown error')}")
        return

    logger.info(f"Successfully downloaded: {downloaded['pdf_filename']}")

    # Step 4: Convert the PDF to markdown
    logger.info("\n[Step 4] Converting PDF to Markdown...")
    converter = BatchDoclingConverter(
        input_folder="to_process",
        output_folder="processed",
        batch_size=1,
        remove_processed=False,  # Keep the file for inspection
        use_gpu=False,  # Use CPU for single document test
        log_level=logging.INFO,
    )

    try:
        stats = await converter.convert_all()
        logger.info("\nConversion Results:")
        logger.info(f"  Total files: {stats['total_files']}")
        logger.info(f"  Processed: {stats['processed_files']}")
        logger.info(f"  Failed: {stats['failed_files']}")

        if stats['processed_files'] > 0:
            logger.info("\n✓ SUCCESS! Document downloaded and converted")
            logger.info(f"  PDF: to_process/{downloaded['pdf_filename']}")

            # Find the markdown file
            pdf_path = Path(downloaded['pdf_filename'])
            md_filename = pdf_path.with_suffix('.md').name
            logger.info(f"  Markdown: processed/{md_filename}")

            # Show first few lines of markdown
            md_path = Path("processed") / md_filename
            if md_path.exists():
                logger.info("\nFirst 20 lines of converted markdown:")
                logger.info("-" * 60)
                with open(md_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:20]
                    for line in lines:
                        print(line.rstrip())
                logger.info("-" * 60)
        else:
            logger.error("Conversion failed")

    finally:
        converter.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_single_document())
