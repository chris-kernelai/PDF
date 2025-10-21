#!/usr/bin/env python3
"""Unified pipeline runner with subcommands for common workflows."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR / "workspace"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

SRC_PATH = WORKSPACE_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(WORKSPACE_ROOT))
else:
    print(f"Warning: workspace/src not found at {SRC_PATH}", file=sys.stderr)

from src.pipeline import DocumentFetcher, ImageDescriptionWorkflow
from src.pipeline.docling_batch_converter import convert_folder
from src.pipeline.image_extraction import (
    extract_images_from_directory,
    extract_images_from_pdf,
)
from src.pipeline.paths import CONFIGS_DIR, DATA_DIR, LOGS_DIR, STATE_DIR, WORKSPACE_ROOT as PATH_WORKSPACE_ROOT

load_dotenv()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _ensure_data_dirs() -> None:
    directories = [
        DATA_DIR / "images",
        DATA_DIR / "to_process",
        DATA_DIR / "processed",
        DATA_DIR / "processed_raw",
        DATA_DIR / "processed_images",
        DATA_DIR / "processed_images_raw",
        LOGS_DIR,
        STATE_DIR,
        PATH_WORKSPACE_ROOT / ".generated",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


async def run_full(args: argparse.Namespace) -> None:
    configure_logging()
    logger = logging.getLogger("pipeline.full")

    _ensure_data_dirs()

    fetcher = DocumentFetcher(
        config_path=args.config,
        limit=args.limit,
        randomize=args.randomize,
        random_seed=args.random_seed,
        min_doc_id=args.min_doc_id,
        max_doc_id=args.max_doc_id,
        run_all_images=args.run_all_images,
        download_pdfs=not args.fetch_only,
    )

    logger.info("Fetching documents (min=%s, max=%s)", args.min_doc_id, args.max_doc_id)
    fetch_stats = await fetcher.run()
    logger.info(
        "Fetched %s documents (selected=%s, downloaded=%s)",
        fetch_stats.documents_considered,
        fetch_stats.documents_selected,
        fetch_stats.documents_downloaded,
    )

    if args.fetch_only:
        logger.info("Fetch-only flag set; skipping conversion stage")
        return

    output_stats = await convert_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
        extract_images=not args.skip_images,
        chunk_page_limit=args.chunk_page_limit,
        max_docs=args.max_docs,
    )
    logger.info("Markdown conversion complete: %s", output_stats)

    if args.skip_images:
        logger.info("Image workflow skipped by flag")
        return

    workflow = ImageDescriptionWorkflow(
        images_dir=args.images_dir,
        processed_markdown_dir=args.output_folder,
        enhanced_markdown_dir=args.enhanced_dir,
        generated_root=args.generated_root,
        gemini_model=args.gemini_model,
        gcs_input_prefix=args.gcs_input_prefix,
        gcs_output_prefix=args.gcs_output_prefix,
        batch_prefix=args.batch_prefix,
        image_format=args.image_format,
    )

    upload_summary = await workflow.run(
        session_id=args.session_id,
        batch_size=args.image_batch_size,
        system_instruction=args.image_system_instruction,
        wait_seconds=args.image_wait_seconds,
        max_retries=args.image_max_retries,
        upload=not args.skip_image_upload,
    )
    logger.info(
        "Image upload summary: uploaded=%s skipped=%s failed=%s",
        upload_summary.uploaded,
        upload_summary.skipped,
        upload_summary.failed,
    )


async def run_markdown_only(args: argparse.Namespace) -> None:
    configure_logging()
    logger = logging.getLogger("pipeline.markdown")

    _ensure_data_dirs()

    stats = await convert_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
        extract_images=args.extract_images,
        chunk_page_limit=args.chunk_page_limit,
        max_docs=args.max_docs,
    )
    logger.info("Markdown conversion complete: %s", stats)


async def run_images_only(args: argparse.Namespace) -> None:
    configure_logging()
    logger = logging.getLogger("pipeline.images")

    _ensure_data_dirs()

    if args.pdf:
        total = 0
        for pdf_path in args.pdf:
            count = await asyncio.to_thread(
                extract_images_from_pdf,
                pdf_path,
                args.output_dir / pdf_path.stem,
            )
            total += count
        logger.info("Extracted %s images from %s PDFs", total, len(args.pdf))
    else:
        count = await asyncio.to_thread(
            extract_images_from_directory,
            args.input_dir,
            args.output_dir,
        )
        logger.info("Extracted %s images from %s", count, args.input_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kernel PDF pipeline runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_parser = subparsers.add_parser("full", help="Run fetch + markdown conversion pipeline")
    full_parser.add_argument("min_doc_id", type=int)
    full_parser.add_argument("max_doc_id", type=int)
    full_parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "config.yaml")
    full_parser.add_argument("--limit", type=int, default=None)
    full_parser.add_argument("--randomize", action="store_true")
    full_parser.add_argument("--random-seed", type=int, default=42)
    full_parser.add_argument("--run-all-images", action="store_true")
    full_parser.add_argument("--batch-size", type=int, default=1)
    full_parser.add_argument("--chunk-page-limit", type=int, default=50)
    full_parser.add_argument("--input-folder", type=Path, default=DATA_DIR / "to_process")
    full_parser.add_argument("--output-folder", type=Path, default=DATA_DIR / "processed")
    full_parser.add_argument("--images-dir", type=Path, default=DATA_DIR / "images")
    full_parser.add_argument("--enhanced-dir", type=Path, default=DATA_DIR / "processed_images")
    full_parser.add_argument("--generated-root", type=Path, default=WORKSPACE_ROOT / ".generated")
    full_parser.add_argument("--gemini-model", type=str, default="gemini-2.0-flash-001")
    full_parser.add_argument("--gcs-input-prefix", type=str, default="gemini_batches/input")
    full_parser.add_argument("--gcs-output-prefix", type=str, default="gemini_batches/output")
    full_parser.add_argument("--batch-prefix", type=str, default="image_description_batches")
    full_parser.add_argument("--image-format", type=str, default="detailed")
    full_parser.add_argument("--image-batch-size", type=int, default=100)
    full_parser.add_argument("--image-wait-seconds", type=int, default=120)
    full_parser.add_argument("--image-max-retries", type=int, default=60)
    full_parser.add_argument("--image-system-instruction", type=str, default=None)
    full_parser.add_argument("--session-id", type=str, default=None)
    full_parser.add_argument("--max-docs", type=int, default=None)
    full_parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    full_parser.add_argument("--fetch-only", action="store_true", help="Skip conversion stage")
    full_parser.add_argument("--skip-images", action="store_true", help="Skip Gemini image workflow")
    full_parser.add_argument("--skip-image-upload", action="store_true", help="Skip uploading Supabase representations")
