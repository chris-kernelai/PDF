#!/usr/bin/env python3
"""Unified pipeline runner with subcommands for common workflows."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR / "workspace"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

SRC_PATH = WORKSPACE_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(WORKSPACE_ROOT))
else:
    print(f"Warning: workspace/src not found at {SRC_PATH}", file=sys.stderr)

from src.pipeline import (
    DoclingMarkdownDownloader,
    DocumentFetcher,
    ImageDescriptionWorkflow,
    SupabaseConfig,
    UploadSummary,
    fetch_existing_representations,
)
from src.pipeline.docling_batch_converter import convert_folder
try:
    from src.pipeline.image_extraction import extract_images_from_pdf
except ImportError:  # pragma: no cover - optional dependency
    extract_images_from_pdf = None
from src.pipeline.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    LOGS_DIR,
    STATE_DIR,
    WORKSPACE_ROOT as PATH_WORKSPACE_ROOT,
)
from src.standalone_upload_representations import DocumentRepresentationUploader

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
    print("DEBUG: run_full() called", file=sys.stderr)
    print(f"DEBUG: args.min_doc_id={args.min_doc_id}, args.max_doc_id={args.max_doc_id}, args.limit={args.limit}", file=sys.stderr)

    configure_logging()
    logger = logging.getLogger("pipeline.full")
    logger.info("=" * 80)
    logger.info("Starting full pipeline")
    logger.info("Arguments: %s", args)
    logger.info("=" * 80)

    print("DEBUG: Ensuring data directories", file=sys.stderr)
    _ensure_data_dirs()
    logger.info("Data directories created/verified")

    print("DEBUG: Creating DocumentFetcher", file=sys.stderr)
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
    logger.info("DocumentFetcher created with config: %s", args.config)

    print("DEBUG: Starting document fetch", file=sys.stderr)
    logger.info("Fetching documents (min=%s, max=%s)", args.min_doc_id, args.max_doc_id)
    fetch_stats = await fetcher.run()
    logger.info(
        "Fetched %s documents (selected=%s, downloaded=%s)",
        fetch_stats.documents_considered,
        fetch_stats.documents_selected,
        fetch_stats.documents_downloaded,
    )
    print(f"DEBUG: Fetch complete - considered={fetch_stats.documents_considered}, selected={fetch_stats.documents_selected}, downloaded={fetch_stats.documents_downloaded}", file=sys.stderr)

    if args.fetch_only:
        logger.info("Fetch-only flag set; skipping conversion stage")
        return

    print("DEBUG: Starting markdown conversion", file=sys.stderr)
    output_stats = await convert_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
        extract_images=not args.skip_images,
        chunk_page_limit=args.chunk_page_limit,
        max_docs=args.max_docs,
        min_doc_id=args.min_doc_id,
        max_doc_id=args.max_doc_id,
    )
    logger.info("Markdown conversion complete: %s", output_stats)
    print(f"DEBUG: Markdown conversion stats: {output_stats}", file=sys.stderr)

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
        aws_profile=getattr(args, "aws_profile", None),
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
    print("DEBUG: run_markdown_only() called", file=sys.stderr)
    print(f"DEBUG: args.min_doc_id={args.min_doc_id}, args.max_doc_id={args.max_doc_id}, args.limit={args.limit}", file=sys.stderr)

    configure_logging()
    logger = logging.getLogger("pipeline.markdown")
    logger.info("=" * 80)
    logger.info("Starting markdown-only pipeline")
    logger.info("Arguments: %s", args)
    logger.info("=" * 80)

    _ensure_data_dirs()

    stats = await convert_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
        extract_images=args.extract_images,
        chunk_page_limit=args.chunk_page_limit,
        max_docs=args.max_docs,
        min_doc_id=args.min_doc_id,
        max_doc_id=args.max_doc_id,
    )
    logger.info("Markdown conversion complete: %s", stats)
    print(f"DEBUG: Markdown conversion stats: {stats}", file=sys.stderr)

    if getattr(args, "upload_docling", False):
        doc_ids = list(stats.get("doc_ids_processed", []))
        if not doc_ids:
            doc_ids = _discover_existing_markdown_doc_ids(
                args.output_folder,
                min_doc_id=args.min_doc_id,
                max_doc_id=args.max_doc_id,
                limit=args.limit,
            )
            if doc_ids:
                logger.info(
                    "Found %s existing markdown files to upload from %s",
                    len(doc_ids),
                    args.output_folder,
                )
        upload_summary = await upload_docling_markdown(
            doc_ids,
            args.output_folder,
            aws_profile=getattr(args, "aws_profile", None),
        )
        logger.info(
            "Docling upload summary: uploaded=%s skipped=%s failed=%s",
            upload_summary.uploaded,
            upload_summary.skipped,
            upload_summary.failed,
        )
        for error in upload_summary.errors:
            logger.error("Docling upload error: %s", error)


async def upload_docling_markdown(
    doc_ids: Iterable[int],
    output_folder: Path,
    *,
    aws_profile: str | None = None,
) -> UploadSummary:
    logger = logging.getLogger("pipeline.markdown")
    doc_ids_set = {
        int(doc_id) for doc_id in doc_ids if isinstance(doc_id, int)
    }

    summary = UploadSummary()

    if not doc_ids_set:
        logger.info("No document IDs available for Docling upload")
        return summary

    uploader = DocumentRepresentationUploader(aws_profile=aws_profile)
    await uploader.initialize()

    try:
        existing = await uploader.get_existing_representations(list(doc_ids_set))

        for doc_id in sorted(doc_ids_set):
            docling_path = output_folder / f"doc_{doc_id}.md"
            if not docling_path.exists():
                docling_path = output_folder / f"{doc_id}.md"

            if not docling_path.exists():
                logger.warning(
                    "Docling markdown missing for doc_%s; skipping upload",
                    doc_id,
                )
                summary.skipped += 1
                continue

            reps = existing.get(doc_id, set())
            if "DOCLING" in reps:
                summary.skipped += 1
                continue

            result = await uploader.upload_representations(
                document_id=doc_id,
                docling_file=str(docling_path),
                docling_img_file=None,
                docling_filename=f"doc_{doc_id}.txt",
                docling_img_filename=None,
            )

            if result["errors"]:
                summary.failed += 1
                summary.errors.extend(result["errors"])
            else:
                summary.uploaded += 1

    finally:
        await uploader.close()

    return summary


def _discover_existing_markdown_doc_ids(
    output_folder: Path,
    *,
    min_doc_id: int | None,
    max_doc_id: int | None,
    limit: int | None,
) -> list[int]:
    if not output_folder.exists():
        return []

    doc_ids: set[int] = set()
    for md_path in sorted(output_folder.glob("*.md")):
        stem = md_path.stem
        if stem.startswith("doc_"):
            candidate = stem[4:]
        else:
            candidate = stem
        try:
            doc_id = int(candidate)
        except ValueError:
            continue

        if min_doc_id is not None and doc_id < min_doc_id:
            continue
        if max_doc_id is not None and doc_id > max_doc_id:
            continue

        doc_ids.add(doc_id)

    ordered = sorted(doc_ids)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def _parse_doc_id_from_path(pdf_path: Path) -> int | None:
    stem = pdf_path.stem
    if stem.startswith("doc_"):
        stem = stem[4:]
    try:
        return int(stem)
    except ValueError:
        return None


async def run_images_only(args: argparse.Namespace) -> None:
    configure_logging()
    logger = logging.getLogger("pipeline.images_only")

    _ensure_data_dirs()

    output_dir = args.output_folder
    images_dir = args.images_dir
    enhanced_dir = args.enhanced_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    if not args.pdf:
        logger.error("Images-only mode requires at least one --pdf argument")
        return

    pdf_map: dict[int, Path] = {}
    for pdf in args.pdf:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            logger.warning("Skipping missing PDF: %s", pdf_path)
            continue

        doc_id = _parse_doc_id_from_path(pdf_path)
        if doc_id is None:
            logger.warning("Could not determine document ID from %s", pdf_path.name)
            continue

        pdf_map[doc_id] = pdf_path

    if not pdf_map:
        logger.error("No valid PDF inputs detected; aborting images-only pipeline")
        return

    doc_ids = sorted(pdf_map.keys())
    logger.info("Evaluating %s documents for images-only processing", len(doc_ids))

    supabase_config = SupabaseConfig.from_env()
    existing = await fetch_existing_representations(supabase_config)

    eligible_docs: list[int] = []
    skipped_missing_docling: list[int] = []
    skipped_has_docling_img: list[int] = []

    for doc_id in doc_ids:
        reps = existing.get(doc_id, set())
        if "DOCLING" not in reps:
            skipped_missing_docling.append(doc_id)
            continue
        if "DOCLING_IMG" in reps:
            skipped_has_docling_img.append(doc_id)
            continue
        eligible_docs.append(doc_id)

    if skipped_missing_docling:
        logger.info(
            "Skipping %s documents with no DOCLING representation: %s",
            len(skipped_missing_docling),
            skipped_missing_docling,
        )
    if skipped_has_docling_img:
        logger.info(
            "Skipping %s documents that already have DOCLING_IMG: %s",
            len(skipped_has_docling_img),
            skipped_has_docling_img,
        )

    if args.max_docs is not None:
        eligible_docs = eligible_docs[: args.max_docs]

    if not eligible_docs:
        logger.info("No documents require images-only processing; exiting")
        return

    downloader = DoclingMarkdownDownloader(
        output_dir=output_dir,
        supabase_config=supabase_config,
        aws_profile=getattr(args, "aws_profile", None),
    )
    download_summary = await downloader.download(eligible_docs)
    logger.info(
        "Markdown download summary: requested=%s downloaded=%s skipped=%s missing=%s failed=%s",
        download_summary.requested,
        download_summary.downloaded,
        download_summary.skipped_existing,
        len(download_summary.missing),
        len(download_summary.failed),
    )
    if download_summary.failed:
        for doc_id, error_msg in download_summary.failed.items():
            logger.error("Failed to download markdown for doc_%s: %s", doc_id, error_msg)

    final_docs: list[int] = []
    for doc_id in eligible_docs:
        md_path = output_dir / f"doc_{doc_id}.md"
        if md_path.exists():
            final_docs.append(doc_id)
        else:
            logger.warning(
                "Markdown not available locally for doc_%s; skipping image extraction",
                doc_id,
            )

    if not final_docs:
        logger.error("No documents remain after markdown download; aborting")
        return

    total_images = 0
    processed_docs: list[int] = []

    for doc_id in final_docs:
        pdf_path = pdf_map[doc_id]
        doc_images_dir = images_dir / f"doc_{doc_id}"
        if doc_images_dir.exists():
            shutil.rmtree(doc_images_dir, ignore_errors=True)

        try:
            count = await asyncio.to_thread(
                extract_images_from_pdf,
                pdf_path,
                doc_images_dir,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Image extraction failed for doc_%s: %s", doc_id, exc)
            continue

        if count > 0:
            total_images += count
            processed_docs.append(doc_id)
        else:
            logger.warning("No images found in doc_%s", doc_id)

    if not processed_docs:
        logger.warning("No images extracted; skipping Gemini workflow")
        return

    logger.info(
        "Extracted %s images across %s documents", total_images, len(processed_docs)
    )

    workflow = ImageDescriptionWorkflow(
        images_dir=images_dir,
        processed_markdown_dir=output_dir,
        enhanced_markdown_dir=enhanced_dir,
        generated_root=args.generated_root,
        gemini_model=args.gemini_model,
        gcs_input_prefix=args.gcs_input_prefix,
        gcs_output_prefix=args.gcs_output_prefix,
        batch_prefix=args.batch_prefix,
        image_format=args.image_format,
        aws_profile=getattr(args, "aws_profile", None),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kernel PDF pipeline runner")
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile name to use for S3 interactions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_parser = subparsers.add_parser("full", help="Run fetch + markdown conversion pipeline")
    full_parser.add_argument("min_doc_id", type=int, help="Minimum document ID to process")
    full_parser.add_argument("max_doc_id", type=int, help="Maximum document ID to process")
    full_parser.add_argument("limit", type=int, nargs="?", default=None, help="Maximum number of documents to process (optional)")
    full_parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "config.yaml")
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

    markdown_parser = subparsers.add_parser("markdown", help="Run markdown conversion only")
    markdown_parser.add_argument("min_doc_id", type=int, nargs="?", default=None, help="Minimum document ID to process (optional)")
    markdown_parser.add_argument("max_doc_id", type=int, nargs="?", default=None, help="Maximum document ID to process (optional)")
    markdown_parser.add_argument("limit", type=int, nargs="?", default=None, help="Maximum number of documents to process (optional)")
    markdown_parser.add_argument("--input-folder", type=Path, default=DATA_DIR / "to_process")
    markdown_parser.add_argument("--output-folder", type=Path, default=DATA_DIR / "processed")
    markdown_parser.add_argument("--batch-size", type=int, default=1)
    markdown_parser.add_argument("--chunk-page-limit", type=int, default=50)
    markdown_parser.add_argument("--max-docs", type=int, default=None)
    markdown_parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    markdown_parser.add_argument("--extract-images", action="store_true", help="Extract images during conversion")
    markdown_parser.add_argument(
        "--upload-docling",
        action="store_true",
        help="Upload Docling markdown representations after conversion",
    )

    images_parser = subparsers.add_parser(
        "images",
        help="Run images-only workflow for documents with existing Docling markdown",
    )
    images_parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        required=True,
        help="PDF file to process (repeatable)",
    )
    images_parser.add_argument(
        "--output-folder",
        type=Path,
        default=DATA_DIR / "processed",
        help="Directory containing Docling markdown files",
    )
    images_parser.add_argument(
        "--images-dir",
        type=Path,
        default=DATA_DIR / "images_only",
        help="Directory to store extracted images",
    )
    images_parser.add_argument(
        "--enhanced-dir",
        type=Path,
        default=DATA_DIR / "processed_images",
        help="Directory for markdown merged with image descriptions",
    )
    images_parser.add_argument(
        "--generated-root",
        type=Path,
        default=WORKSPACE_ROOT / ".generated",
        help="Directory for intermediate artifacts",
    )
    images_parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash-001",
        help="Gemini model name",
    )
    images_parser.add_argument(
        "--gcs-input-prefix",
        type=str,
        default="gemini_batches/input",
        help="GCS prefix for batch job inputs",
    )
    images_parser.add_argument(
        "--gcs-output-prefix",
        type=str,
        default="gemini_batches/output",
        help="GCS prefix for batch job outputs",
    )
    images_parser.add_argument(
        "--batch-prefix",
        type=str,
        default="image_description_batches",
        help="Prefix for generated batch files",
    )
    images_parser.add_argument(
        "--image-format",
        type=str,
        default="detailed",
        help="Image description formatting mode",
    )
    images_parser.add_argument(
        "--image-batch-size",
        type=int,
        default=100,
        help="Number of images per Gemini batch job",
    )
    images_parser.add_argument(
        "--image-wait-seconds",
        type=int,
        default=120,
        help="Seconds to wait between Gemini batch status checks",
    )
    images_parser.add_argument(
        "--image-max-retries",
        type=int,
        default=60,
        help="Maximum Gemini batch status polls",
    )
    images_parser.add_argument(
        "--image-system-instruction",
        type=str,
        default=None,
        help="Optional system instruction for Gemini",
    )
    images_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional session identifier for naming artifacts",
    )
    images_parser.add_argument(
        "--skip-image-upload",
        action="store_true",
        help="Skip uploading merged markdown to Supabase",
    )
    images_parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of documents processed",
    )

    return parser


async def async_main() -> int:
    """Main async entry point with comprehensive debug logging."""
    print("=" * 80, file=sys.stderr)
    print("DEBUG: Starting run_pipeline.py", file=sys.stderr)
    print(f"DEBUG: Command line args: {sys.argv}", file=sys.stderr)
    print(f"DEBUG: Python version: {sys.version}", file=sys.stderr)
    print(f"DEBUG: Working directory: {Path.cwd()}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    try:
        parser = build_parser()
        print("DEBUG: Parser created successfully", file=sys.stderr)

        args = parser.parse_args()
        print(f"DEBUG: Parsed args: {args}", file=sys.stderr)
        print(f"DEBUG: Command: {args.command}", file=sys.stderr)

        if getattr(args, "aws_profile", None):
            os.environ["AWS_PROFILE"] = args.aws_profile
            print(
                f"DEBUG: AWS_PROFILE set to {args.aws_profile}",
                file=sys.stderr,
            )

        command_map: dict[str, Callable[[argparse.Namespace], Awaitable[None]]] = {
            "full": run_full,
            "markdown": run_markdown_only,
            "images": run_images_only,
        }

        if args.command not in command_map:
            print(f"ERROR: Unknown command: {args.command}", file=sys.stderr)
            parser.print_help()
            return 1

        print(f"DEBUG: Executing command: {args.command}", file=sys.stderr)
        await command_map[args.command](args)
        print("DEBUG: Command completed successfully", file=sys.stderr)
        return 0

    except Exception as e:
        print(f"ERROR: Fatal exception occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    print("DEBUG: main() called", file=sys.stderr)
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nINFO: Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: Unhandled exception in main: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    print("DEBUG: __name__ == '__main__' block executing", file=sys.stderr)
    sys.exit(main())
