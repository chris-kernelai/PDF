#!/usr/bin/env python3
"""Fetch documents that still need Docling or Docling_IMG representations."""

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.document_fetcher import DocumentFetcher

load_dotenv()


def configure_logging(config_path: Path) -> None:
    import yaml

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    log_config = config.get("logging", {})
    level_name = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = log_config.get("file", "fetch_documents.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch documents for Docling processing")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to pipeline config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--randomize", action="store_true", help="Randomize selection before applying limit")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used when randomizing")
    parser.add_argument("--min-doc-id", type=int, default=None, help="Minimum document ID")
    parser.add_argument("--max-doc-id", type=int, default=None, help="Maximum document ID")
    parser.add_argument("--run-all-images", action="store_true", help="Fetch documents that already have DOCLING but are missing DOCLING_IMG")
    parser.add_argument("--no-download", action="store_true", help="List documents without downloading PDFs")
    args = parser.parse_args()

    configure_logging(args.config)
    logging.getLogger(__name__).info("Starting document fetch")

    fetcher = DocumentFetcher(
        config_path=args.config,
        limit=args.limit,
        randomize=args.randomize,
        random_seed=args.random_seed,
        min_doc_id=args.min_doc_id,
        max_doc_id=args.max_doc_id,
        run_all_images=args.run_all_images,
        download_pdfs=not args.no_download,
    )

    stats = await fetcher.run()

    logging.info("=" * 60)
    logging.info("Fetch Summary")
    logging.info("Documents considered: %s", stats.documents_considered)
    logging.info("Documents selected: %s", stats.documents_selected)
    logging.info("Documents skipped (existing reps): %s", stats.documents_skipped_existing)
    if stats.documents_skipped_completed:
        logging.info("Documents skipped (completed log): %s", stats.documents_skipped_completed)
    logging.info("Documents downloaded: %s", stats.documents_downloaded)
    if stats.documents_rejected_not_pdf:
        logging.info("Rejected (not PDF): %s", stats.documents_rejected_not_pdf)
    if stats.documents_failed:
        logging.info("Failed downloads: %s", stats.documents_failed)
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
