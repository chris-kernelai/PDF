"""
Test script to reprocess a document with maximum quality settings.
"""

import asyncio
import os
from pathlib import Path
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


async def reprocess_with_max_quality():
    """Reprocess the downloaded document with maximum quality settings."""

    logger.info("=" * 60)
    logger.info("Reprocessing with MAXIMUM QUALITY settings")
    logger.info("=" * 60)

    # First, delete the existing markdown if it exists
    md_path = Path("processed/doc_37350.md")
    if md_path.exists():
        md_path.unlink()
        logger.info("Deleted existing markdown file for fresh conversion")

    # Convert with maximum quality settings
    logger.info("\n[Processing] Converting PDF with maximum quality settings...")
    logger.info("Settings:")
    logger.info("  - Table mode: accurate (highest quality)")
    logger.info("  - Image scale: 3.0 (maximum quality)")
    logger.info("  - Cell matching: enabled (precise tables)")
    logger.info("  - OCR confidence: 0.05 (capture more text)")
    logger.info("  - GPU: disabled (using CPU)")

    converter = BatchDoclingConverter(
        input_folder="to_process",
        output_folder="processed",
        batch_size=1,
        remove_processed=False,  # Keep the file for inspection
        use_gpu=False,  # CPU mode
        log_level=logging.INFO,
        # MAXIMUM QUALITY SETTINGS
        table_mode="accurate",  # Highest quality table recognition
        images_scale=3.0,  # Maximum image quality (higher = better)
        do_cell_matching=True,  # Precise cell matching in tables
        ocr_confidence_threshold=0.05,  # Lower threshold = capture more text
    )

    try:
        stats = await converter.convert_all()
        logger.info("\nConversion Results:")
        logger.info(f"  Total files: {stats['total_files']}")
        logger.info(f"  Processed: {stats['processed_files']}")
        logger.info(f"  Failed: {stats['failed_files']}")

        if stats['processed_files'] > 0:
            logger.info("\n✓ SUCCESS! Document reprocessed with maximum quality")
            logger.info(f"  PDF: to_process/doc_37350.pdf")
            logger.info(f"  Markdown: processed/doc_37350.md")

            # Show first few lines of markdown
            if md_path.exists():
                file_size = md_path.stat().st_size / 1024  # KB
                logger.info(f"  File size: {file_size:.1f} KB")

                logger.info("\nFirst 30 lines of converted markdown:")
                logger.info("-" * 60)
                with open(md_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:30]
                    for line in lines:
                        print(line.rstrip())
                logger.info("-" * 60)
                logger.info(f"\nTotal lines in markdown: {len(open(md_path).readlines())}")
        else:
            logger.error("Conversion failed")

    finally:
        converter.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("Maximum quality processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(reprocess_with_max_quality())
