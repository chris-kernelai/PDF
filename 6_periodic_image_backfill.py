#!/usr/bin/env python3
"""
periodic_image_backfill.py

Periodically check for documents that have been converted but don't have
image descriptions integrated yet. Runs every N minutes in a loop.

This is designed to run in the background with 2 free cores while the main
pipeline runs with 2 workers doing Docling conversion and image extraction.

Usage:
    python 6_periodic_image_backfill.py                    # Run every 20 minutes (default)
    python 6_periodic_image_backfill.py --interval 10      # Run every 10 minutes
    python 6_periodic_image_backfill.py --once             # Run once and exit
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_integration_backfill(
    input_dir: Path = Path("processed"),
    output_dir: Path = Path("processed_images"),
    descriptions_dir: Optional[Path] = None,
    batch_prefix: str = "image_description_batches",
    unfiltered: bool = False,
    format_style: str = "detailed",
) -> bool:
    """
    Run the integration process for new documents only.

    Args:
        input_dir: Directory containing original markdown files
        output_dir: Directory to save enhanced markdown files
        descriptions_dir: Directory containing description JSON files
        batch_prefix: Batch prefix to locate description files
        unfiltered: Use unfiltered descriptions
        format_style: Format for image descriptions

    Returns:
        True if successful
    """
    try:
        # Import here to avoid circular dependencies
        from image_description_integrator import ImageDescriptionIntegrator

        # Determine descriptions directory
        if descriptions_dir is None:
            if unfiltered:
                descriptions_dir = Path(f".generated/{batch_prefix}_outputs")
            else:
                descriptions_dir = Path(f".generated/{batch_prefix}_outputs_filtered")

        # Check directories exist
        if not input_dir.exists():
            logger.warning(f"Input directory not found: {input_dir}")
            return False

        if not descriptions_dir.exists():
            logger.debug(f"Descriptions directory not found: {descriptions_dir}")
            return False

        logger.info("=" * 80)
        logger.info("🔄 Running periodic image description backfill")
        logger.info(f"📁 Input: {input_dir}")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"📁 Descriptions: {descriptions_dir}")
        logger.info("=" * 80)

        # Initialize integrator
        integrator = ImageDescriptionIntegrator(
            markdown_dir=input_dir,
            output_dir=output_dir,
            descriptions_dir=descriptions_dir,
            image_format=format_style,
            overwrite=False,  # Never overwrite in backfill mode
        )

        # Load descriptions
        descriptions_by_doc = integrator.load_all_descriptions()

        if not descriptions_by_doc:
            logger.debug("No descriptions available yet")
            return False

        # Process only new files (those with descriptions but no output yet)
        success = integrator.process_all(descriptions_by_doc, only_new=True)

        # Print summary if we processed anything
        if integrator.stats["processed_files"] > 0:
            integrator.print_summary()
            logger.info(f"✅ Processed {integrator.stats['processed_files']} new files")
        else:
            logger.info("No new files to process")

        return success

    except Exception as e:
        logger.error(f"Failed to run integration backfill: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def run_periodic_loop(
    interval_minutes: int = 20,
    input_dir: Path = Path("processed"),
    output_dir: Path = Path("processed_images"),
    descriptions_dir: Optional[Path] = None,
    batch_prefix: str = "image_description_batches",
    unfiltered: bool = False,
    format_style: str = "detailed",
) -> None:
    """
    Run the backfill process in a periodic loop.

    Args:
        interval_minutes: How often to run (in minutes)
        input_dir: Directory containing original markdown files
        output_dir: Directory to save enhanced markdown files
        descriptions_dir: Directory containing description JSON files
        batch_prefix: Batch prefix to locate description files
        unfiltered: Use unfiltered descriptions
        format_style: Format for image descriptions
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Periodic Image Description Backfill Service")
    logger.info("=" * 80)
    logger.info(f"⏰ Interval: {interval_minutes} minutes")
    logger.info(f"📁 Input directory: {input_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🎨 Format: {format_style}")
    logger.info("=" * 80)
    logger.info("Press Ctrl+C to stop\n")

    iteration = 0
    while True:
        try:
            iteration += 1
            logger.info(f"\n{'=' * 80}")
            logger.info(f"🔄 Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'=' * 80}")

            # Run the backfill
            run_integration_backfill(
                input_dir=input_dir,
                output_dir=output_dir,
                descriptions_dir=descriptions_dir,
                batch_prefix=batch_prefix,
                unfiltered=unfiltered,
                format_style=format_style,
            )

            # Wait for next iteration
            logger.info(f"\n⏸️  Sleeping for {interval_minutes} minutes...")
            logger.info(f"   Next run at: {datetime.fromtimestamp(time.time() + interval_minutes * 60).strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("\n\n👋 Received interrupt signal, shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Error in periodic loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info(f"⏸️  Waiting {interval_minutes} minutes before retry...")
            time.sleep(interval_minutes * 60)

    logger.info("=" * 80)
    logger.info("✅ Periodic backfill service stopped")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Periodically integrate image descriptions for newly processed documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run every 20 minutes (default)
  python 6_periodic_image_backfill.py

  # Run every 10 minutes
  python 6_periodic_image_backfill.py --interval 10

  # Run once and exit
  python 6_periodic_image_backfill.py --once

  # Custom directories
  python 6_periodic_image_backfill.py --input-dir processed --output-dir processed_enhanced
        """
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="How often to run (in minutes, default: 20)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't loop)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("processed"),
        help="Directory containing original markdown files (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_images"),
        help="Directory to save enhanced markdown files (default: processed_images)",
    )
    parser.add_argument(
        "--descriptions-dir",
        type=Path,
        default=None,
        help="Directory containing description JSON files (default: .generated/{batch_prefix}_outputs_filtered)",
    )
    parser.add_argument(
        "--batch-prefix",
        default="image_description_batches",
        help="Batch prefix to locate description files (default: image_description_batches)",
    )
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="Use unfiltered descriptions instead of filtered ones",
    )
    parser.add_argument(
        "--format",
        choices=["detailed", "inline", "section"],
        default="detailed",
        help="Format for image descriptions (default: detailed)",
    )

    args = parser.parse_args()

    if args.once:
        # Run once and exit
        logger.info("Running once (--once mode)")
        success = run_integration_backfill(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            descriptions_dir=args.descriptions_dir,
            batch_prefix=args.batch_prefix,
            unfiltered=args.unfiltered,
            format_style=args.format,
        )
        return 0 if success else 1
    else:
        # Run in periodic loop
        run_periodic_loop(
            interval_minutes=args.interval,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            descriptions_dir=args.descriptions_dir,
            batch_prefix=args.batch_prefix,
            unfiltered=args.unfiltered,
            format_style=args.format,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
