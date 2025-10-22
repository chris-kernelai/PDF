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
from uuid import uuid4

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
)
from src.pipeline.docling_batch_converter import convert_folder
from src.pipeline.image_extraction import extract_images_from_pdf
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
    print(f"DEBUG: args.min_doc_id={args.min_doc_id}, args.max_doc_id={args.max_doc_id}, args.doc_batch_size={args.doc_batch_size}", file=sys.stderr)

    configure_logging()
    logger = logging.getLogger("pipeline.full")
    logger.info("=" * 80)
    logger.info("Starting full pipeline with batching")
    logger.info("Range: %s-%s, Batch size: %s", args.min_doc_id, args.max_doc_id, args.doc_batch_size)
    logger.info("=" * 80)

    print("DEBUG: Ensuring data directories", file=sys.stderr)
    _ensure_data_dirs()
    logger.info("Data directories created/verified")

    batch_num = 0
    total_processed = 0
    total_uploaded = 0
    processed_doc_ids_all: set[int] = set()
    current_min_doc_id = args.min_doc_id

    while True:
        batch_num += 1
        logger.info("\n\n\n" + "=" * 80)
        logger.info("BATCH %s STARTING", batch_num)
        logger.info("Requesting up to %s documents from range %s-%s", args.doc_batch_size, current_min_doc_id, args.max_doc_id)
        logger.info("=" * 80)

        print(f"DEBUG: Creating DocumentFetcher for batch {batch_num}", file=sys.stderr)
        fetcher = DocumentFetcher(
            config_path=args.config,
            limit=args.doc_batch_size,
            randomize=args.randomize,
            random_seed=args.random_seed,
            min_doc_id=current_min_doc_id,
            max_doc_id=args.max_doc_id,
            run_all_images=args.run_all_images,
            download_pdfs=not args.fetch_only,
        )

        print(f"DEBUG: Starting document fetch for batch {batch_num}", file=sys.stderr)
        fetch_stats = await fetcher.run()
        logger.info(
            "Batch %s: Fetched %s documents (selected=%s, downloaded=%s)",
            batch_num,
            fetch_stats.documents_considered,
            fetch_stats.documents_selected,
            fetch_stats.documents_downloaded,
        )
        print(f"DEBUG: Batch {batch_num} fetch complete - selected={fetch_stats.documents_selected}, downloaded={fetch_stats.documents_downloaded}", file=sys.stderr)

        if args.fetch_only:
            logger.info("Fetch-only flag set; skipping conversion stage")
            if fetch_stats.documents_selected == 0:
                break
            continue

        if fetch_stats.documents_selected == 0:
            logger.info("Batch %s: No more documents to process in range", batch_num)
            break

        # Get the doc IDs from this batch
        batch_doc_ids = []
        if hasattr(fetcher, "last_selected_docs"):
            batch_doc_ids = sorted([doc["id"] for doc in fetcher.last_selected_docs])
        else:
            # Fallback: infer from PDFs in input folder
            for pdf_path in args.input_folder.glob("doc_*.pdf"):
                doc_id = _doc_id_from_filename(pdf_path)
                if doc_id is not None:
                    batch_doc_ids.append(doc_id)
            batch_doc_ids = sorted(batch_doc_ids)

        if not batch_doc_ids:
            logger.warning("Batch %s: No document IDs found; skipping", batch_num)
            continue

        logger.info(
            "Batch %s: Fetched %s documents (doc IDs: %s-%s)",
            batch_num,
            len(batch_doc_ids),
            min(batch_doc_ids),
            max(batch_doc_ids),
        )

        # PHASE 1: Markdown conversion
        logger.info("\n\n\n" + "=" * 80)
        logger.info("Batch %s: PHASE 1 - MARKDOWN CONVERSION", batch_num)
        logger.info("=" * 80)

        # Check which docs already have markdown locally
        docs_needing_conversion = []
        docs_already_converted = []

        logger.info("Batch %s: Checking local markdown status for %s documents", batch_num, len(batch_doc_ids))
        for idx, doc_id in enumerate(batch_doc_ids, 1):
            md_path = args.output_folder / f"doc_{doc_id}.md"
            if md_path.exists():
                docs_already_converted.append(doc_id)
                logger.info(
                    "Batch %s: [%s/%s] doc_%s - markdown exists locally, skipping conversion",
                    batch_num,
                    idx,
                    len(batch_doc_ids),
                    doc_id,
                )
            else:
                # Check if PDF exists for conversion
                pdf_path = args.input_folder / f"doc_{doc_id}.pdf"
                if pdf_path.exists():
                    docs_needing_conversion.append(doc_id)
                    logger.info(
                        "Batch %s: [%s/%s] doc_%s - needs conversion",
                        batch_num,
                        idx,
                        len(batch_doc_ids),
                        doc_id,
                    )

        # Print comprehensive batch statistics
        batch_range_size = args.max_doc_id - current_min_doc_id + 1
        representation_type = "DOCLING_IMG" if args.run_all_images else "DOCLING"
        logger.info("\n" + "=" * 80)
        logger.info("Batch %s: DOCUMENT PROCESSING SUMMARY", batch_num)
        logger.info("=" * 80)
        logger.info("")
        logger.info("Batch Range: doc_%s to doc_%s (%s potential documents)", current_min_doc_id, args.max_doc_id, batch_range_size)
        logger.info("")
        logger.info("Pipeline Stage Breakdown:")
        logger.info("─" * 80)
        logger.info("1. Fetched from kdocuments table in Supabase:           %s documents", fetch_stats.documents_considered)
        logger.info("   ↓")
        logger.info("2. Filtered out (already have %s in DB):      -%s documents", representation_type, fetch_stats.documents_skipped_existing)
        logger.info("   ↓")
        logger.info("3. Selected for processing:                             %s documents", fetch_stats.documents_selected)
        logger.info("   ↓")
        logger.info("4. PDF Download Results:")
        logger.info("   • Downloaded successfully:                            %s documents", fetch_stats.documents_downloaded)
        logger.info("   • Rejected (RAW file is not a valid PDF):            %s documents", fetch_stats.documents_rejected_not_pdf)
        if fetch_stats.documents_failed > 0:
            logger.info("   • Failed (other errors):                              %s documents", fetch_stats.documents_failed)
            logger.info("     Errors:")
            for error in fetch_stats.failed_errors[:10]:  # Show first 10 errors
                logger.info("       - %s", error)
            if len(fetch_stats.failed_errors) > 10:
                logger.info("       ... and %s more", len(fetch_stats.failed_errors) - 10)
        else:
            logger.info("   • Failed (other errors):                              0 documents")
        logger.info("   ↓")
        logger.info("5. Local Markdown Check:")
        logger.info("   • Already have markdown locally (skip conversion):    %s documents", len(docs_already_converted))
        logger.info("   • Need markdown conversion:                           %s documents", len(docs_needing_conversion))
        logger.info("   ↓")
        logger.info("6. Documents proceeding to next phase:                   %s documents", len(docs_already_converted) + len(docs_needing_conversion))
        logger.info("=" * 80)
        logger.info("")

        # Only convert docs that need it
        output_stats = {"processed_files": 0, "uploaded_files": 0, "doc_ids_processed": []}

        if docs_needing_conversion:
            if fetch_stats.documents_selected > 0 and fetch_stats.documents_downloaded == 0:
                logger.warning("Batch %s: Documents were selected but none downloaded successfully", batch_num)
                # Check if any PDFs exist locally
                pdf_count = len(list(args.input_folder.glob("*.pdf")))
                if pdf_count == 0:
                    logger.error("Batch %s: No PDFs available; skipping conversion", batch_num)
                else:
                    logger.info("Batch %s: Converting %s documents", batch_num, len(docs_needing_conversion))
                    print(f"DEBUG: Batch {batch_num} starting markdown conversion", file=sys.stderr)
                    output_stats = await convert_folder(
                        input_folder=args.input_folder,
                        output_folder=args.output_folder,
                        batch_size=args.batch_size,
                        use_gpu=not args.cpu,
                        extract_images=not args.skip_images,
                        chunk_page_limit=args.chunk_page_limit,
                        max_docs=args.max_docs,
                        min_doc_id=min(docs_needing_conversion),
                        max_doc_id=max(docs_needing_conversion),
                    )
                    logger.info("Batch %s: Markdown conversion complete: %s", batch_num, output_stats)
                    print(f"DEBUG: Batch {batch_num} conversion stats: {output_stats}", file=sys.stderr)
            else:
                logger.info("Batch %s: Converting %s documents", batch_num, len(docs_needing_conversion))
                print(f"DEBUG: Batch {batch_num} starting markdown conversion", file=sys.stderr)
                output_stats = await convert_folder(
                    input_folder=args.input_folder,
                    output_folder=args.output_folder,
                    batch_size=args.batch_size,
                    use_gpu=not args.cpu,
                    extract_images=not args.skip_images,
                    chunk_page_limit=args.chunk_page_limit,
                    max_docs=args.max_docs,
                    min_doc_id=min(docs_needing_conversion),
                    max_doc_id=max(docs_needing_conversion),
                )
                logger.info("Batch %s: Markdown conversion complete: %s", batch_num, output_stats)
                print(f"DEBUG: Batch {batch_num} conversion stats: {output_stats}", file=sys.stderr)

        # Track all docs in this batch (both converted and already-converted)
        processed_doc_ids = output_stats.get("doc_ids_processed", [])
        all_batch_docs = list(set(processed_doc_ids + docs_already_converted))

        total_processed += len(all_batch_docs)
        total_uploaded += output_stats.get("uploaded_files", 0)

        logger.info(
            "Batch %s: Markdown phase complete - %s docs in batch (%s newly converted, %s already had markdown)",
            batch_num,
            len(all_batch_docs),
            len(processed_doc_ids),
            len(docs_already_converted),
        )

        # PHASE 2: Process images for this batch
        upload_summary = None
        if not args.skip_images and all_batch_docs:
            logger.info("\n\n\n" + "=" * 80)
            logger.info("Batch %s: PHASE 2 - IMAGE PROCESSING", batch_num)
            logger.info("=" * 80)
            logger.info("Batch %s: Processing images for %s documents (doc IDs: %s-%s)", batch_num, len(all_batch_docs), min(all_batch_docs), max(all_batch_docs))
            batch_session_id = f"{uuid4().hex[:8]}-{batch_num:03d}"
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

            # Extract images, run Gemini batch processing, integrate descriptions
            upload_summary = await workflow.run(
                session_id=batch_session_id,
                batch_size=args.image_batch_size,
                system_instruction=args.image_system_instruction,
                wait_seconds=args.image_wait_seconds,
                max_retries=args.image_max_retries,
                upload=not args.skip_image_upload,
                allowed_doc_ids=all_batch_docs,
            )

            # PHASE 3: Upload complete
            logger.info("\n\n\n" + "=" * 80)
            logger.info("Batch %s: PHASE 3 - UPLOAD COMPLETE", batch_num)
            logger.info("=" * 80)
            logger.info(
                "Batch %s: Uploaded %s documents to Supabase (skipped=%s, failed=%s)",
                batch_num,
                upload_summary.uploaded,
                upload_summary.skipped,
                upload_summary.failed,
            )

            # Clean up batch files for this session
            logger.info("Batch %s: Cleaning up Gemini batch files for session %s", batch_num, batch_session_id)
            cleanup_count_batch = {"batch_files": 0, "description_files": 0}

            # Clean up batch input files
            batch_dir = args.generated_root / args.batch_prefix
            if batch_dir.exists():
                for batch_file in batch_dir.glob(f"*{batch_session_id}*.jsonl"):
                    try:
                        batch_file.unlink()
                        cleanup_count_batch["batch_files"] += 1
                        logger.debug("Deleted batch file: %s", batch_file.name)
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", batch_file.name, exc)

            # Clean up description result files
            descriptions_dir = args.generated_root / "image_descriptions"
            if descriptions_dir.exists():
                for desc_file in descriptions_dir.glob(f"image_descriptions_{batch_session_id}_*.json"):
                    try:
                        desc_file.unlink()
                        cleanup_count_batch["description_files"] += 1
                        logger.debug("Deleted description file: %s", desc_file.name)
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", desc_file.name, exc)

            logger.info(
                "Batch %s: Cleaned up %s batch files, %s description files",
                batch_num,
                cleanup_count_batch["batch_files"],
                cleanup_count_batch["description_files"],
            )
        elif not all_batch_docs:
            logger.info("\n\n\n" + "=" * 80)
            logger.info("Batch %s: SKIPPING PHASE 2 & 3", batch_num)
            logger.info("=" * 80)
            logger.info("Batch %s: No documents to process - skipping image processing and upload", batch_num)
            logger.info("Reason: All documents either failed PDF download or already have complete processing")

        # Clean up processed documents (PDFs, markdown, images) after image workflow completes
        if all_batch_docs:
            logger.info("Batch %s: Cleaning up %s processed documents", batch_num, len(all_batch_docs))
            cleanup_count = {"pdfs": 0, "md": 0, "images": 0}

            for doc_id in all_batch_docs:
                # Remove PDFs
                pdf_path = args.input_folder / f"doc_{doc_id}.pdf"
                if pdf_path.exists():
                    try:
                        pdf_path.unlink()
                        cleanup_count["pdfs"] += 1
                        logger.debug("Deleted PDF: %s", pdf_path.name)
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", pdf_path.name, exc)

                # Remove markdown files (only if not skipping images, otherwise keep them)
                if not args.skip_images:
                    md_files = [
                        args.output_folder / f"doc_{doc_id}.md",
                        args.output_folder / f"{doc_id}.md",
                    ]
                    for md_file in md_files:
                        if md_file.exists():
                            try:
                                md_file.unlink()
                                cleanup_count["md"] += 1
                                logger.debug("Deleted markdown: %s", md_file.name)
                            except Exception as exc:
                                logger.warning("Failed to delete %s: %s", md_file.name, exc)

                # Remove image directories
                img_dirs = [
                    args.images_dir / f"doc_{doc_id}",
                    args.images_dir / str(doc_id),
                ]
                for img_dir in img_dirs:
                    if img_dir.exists() and img_dir.is_dir():
                        try:
                            shutil.rmtree(img_dir)
                            cleanup_count["images"] += 1
                            logger.debug("Deleted image directory: %s", img_dir.name)
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", img_dir.name, exc)

            logger.info(
                "Batch %s: Cleanup complete - deleted %s PDFs, %s markdown files, %s image directories",
                batch_num,
                cleanup_count["pdfs"],
                cleanup_count["md"],
                cleanup_count["images"],
            )

        # Batch complete summary
        logger.info("\n\n\n" + "=" * 80)
        logger.info("BATCH %s COMPLETE", batch_num)
        logger.info("=" * 80)
        logger.info("Documents processed: %s (converted: %s, already had markdown: %s)", len(all_batch_docs), len(processed_doc_ids), len(docs_already_converted))
        if not args.skip_images and upload_summary:
            logger.info("Images uploaded: %s, skipped: %s, failed: %s", upload_summary.uploaded, upload_summary.skipped, upload_summary.failed)
        logger.info("Total documents processed so far: %s", len(processed_doc_ids_all) + len(all_batch_docs))
        logger.info("=" * 80)

        # Update tracking for next batch
        # IMPORTANT: Update min_doc_id based on what was FETCHED, not just what was processed
        # This prevents infinite loops when documents fail to process
        if batch_doc_ids:
            if all_batch_docs:
                processed_doc_ids_all.update(all_batch_docs)
            # Move to next doc_id after the highest one we ATTEMPTED (fetched)
            current_min_doc_id = max(batch_doc_ids) + 1
            logger.info("\n→ Moving to next batch (starting from doc_id %s)", current_min_doc_id)

            # Check if we've exceeded the max range
            if current_min_doc_id > args.max_doc_id:
                logger.info("Reached end of document range (max: %s)", args.max_doc_id)
                break
        else:
            # No documents fetched at all, we're done
            logger.info("No documents fetched, ending pipeline")
            break

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("Total batches: %s", batch_num)
    logger.info("Total documents processed: %s", total_processed)
    logger.info("Total documents uploaded: %s", total_uploaded)
    logger.info("=" * 80)


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


def _directory_has_images(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        if any(path.glob(pattern)):
            return True
    return False


def _doc_id_from_filename(path: Path) -> int | None:
    stem = path.stem
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
    images_base_dir = args.images_dir
    enhanced_dir = args.enhanced_dir
    pdf_dir = Path(args.input_folder)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_base_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    supabase_config = SupabaseConfig.from_env()

    logger.info("=" * 80)
    logger.info("Starting images-only pipeline with batching")
    logger.info("Range: %s-%s, Batch size: %s", args.min_doc_id, args.max_doc_id, args.doc_batch_size)
    logger.info("=" * 80)

    overall_summary = UploadSummary()
    overall_total_images = 0
    overall_processed_docs = 0
    base_session_id = args.session_id or uuid4().hex[:8]
    batch_num = 0
    current_min_doc_id = args.min_doc_id

    workflow = ImageDescriptionWorkflow(
        images_dir=images_base_dir,
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

    while True:
        batch_num += 1
        logger.info("\n\n\n" + "=" * 80)
        logger.info("BATCH %s: Requesting up to %s documents (range: %s-%s)", batch_num, args.doc_batch_size, current_min_doc_id, args.max_doc_id)
        logger.info("=" * 80)

        fetcher = DocumentFetcher(
            config_path=args.config,
            limit=args.doc_batch_size,  # Fetch batch_size docs at a time
            min_doc_id=current_min_doc_id,
            max_doc_id=args.max_doc_id,
            run_all_images=True,
            download_pdfs=not args.skip_pdf_download,
        )
        fetch_stats = await fetcher.run()
        logger.info(
            "Batch %s: Fetched (selected=%s, downloaded=%s)",
            batch_num,
            fetch_stats.documents_selected,
            fetch_stats.documents_downloaded,
        )

        if fetch_stats.documents_selected == 0:
            logger.info("Batch %s: No more documents to process in range", batch_num)
            break

        # Get doc IDs from this batch
        doc_ids: list[int]
        if hasattr(fetcher, "last_selected_docs"):
            last_docs = getattr(fetcher, "last_selected_docs", [])
            doc_ids = sorted({doc["id"] for doc in last_docs})
            logger.info(
                "Batch %s: Document IDs to process: %s-%s (%s docs)",
                batch_num,
                min(doc_ids) if doc_ids else "N/A",
                max(doc_ids) if doc_ids else "N/A",
                len(doc_ids),
            )
        else:
            logger.warning(
                "DocumentFetcher missing last_selected_docs; inferring doc IDs from %s",
                pdf_dir,
            )
            inferred: set[int] = set()
            for pdf_path in pdf_dir.glob("doc_*.pdf"):
                doc_id = _doc_id_from_filename(pdf_path)
                if doc_id is None:
                    continue
                if doc_id < args.min_doc_id or doc_id > args.max_doc_id:
                    continue
                inferred.add(doc_id)
            doc_ids = sorted(inferred)

        if not doc_ids:
            logger.warning("Batch %s: No candidate documents; continuing", batch_num)
            continue

        logger.info("Batch %s: Processing %s documents", batch_num, len(doc_ids))

        pdf_map: dict[int, Path] = {}
        missing_pdfs: list[int] = []
        for doc_id in doc_ids:
            pdf_path = pdf_dir / f"doc_{doc_id}.pdf"
            if pdf_path.exists():
                pdf_map[doc_id] = pdf_path
            else:
                missing_pdfs.append(doc_id)

        if missing_pdfs:
            logger.warning(
                "Batch %s missing PDFs for %s documents: %s",
                batch_num,
                len(missing_pdfs),
                missing_pdfs,
            )

        if not pdf_map:
            logger.warning("No PDFs available for batch %s; skipping", batch_num)
            continue

        downloader = DoclingMarkdownDownloader(
            output_dir=output_dir,
            supabase_config=supabase_config,
            aws_profile=getattr(args, "aws_profile", None),
        )
        download_summary = await downloader.download(pdf_map.keys())
        logger.info(
            "Batch %s markdown download: requested=%s downloaded=%s skipped=%s missing=%s failed=%s",
            batch_num,
            download_summary.requested,
            download_summary.downloaded,
            download_summary.skipped_existing,
            len(download_summary.missing),
            len(download_summary.failed),
        )
        if download_summary.failed:
            for doc_id, error_msg in download_summary.failed.items():
                logger.error("Batch %s failed markdown download doc_%s: %s", batch_num, doc_id, error_msg)

        final_docs: list[int] = []
        for doc_id in sorted(pdf_map.keys()):
            md_path = output_dir / f"doc_{doc_id}.md"
            if md_path.exists():
                final_docs.append(doc_id)
            else:
                logger.warning(
                    "Batch %s markdown missing for doc_%s; skipping",
                    batch_num,
                    doc_id,
                )

        if not final_docs:
            logger.warning("Batch %s: no documents have markdown; skipping", batch_num)
            continue

        # First, check which docs already have enhanced markdown and can skip processing
        force_reintegrate = getattr(args, "force_reintegrate", False)
        docs_needing_integration = []
        docs_already_integrated = []
        docs_to_process = []

        logger.info("\n" + "Batch %s: Checking processing status for %s documents", batch_num, len(final_docs))
        for idx, doc_id in enumerate(final_docs, 1):
            enhanced_path = enhanced_dir / f"doc_{doc_id}.md"
            if enhanced_path.exists() and not force_reintegrate:
                docs_already_integrated.append(doc_id)
                logger.info(
                    "Batch %s: [%s/%s] doc_%s - enhanced markdown exists, will upload only",
                    batch_num,
                    idx,
                    len(final_docs),
                    doc_id,
                )
            else:
                docs_to_process.append(doc_id)
                logger.info(
                    "Batch %s: [%s/%s] doc_%s - needs image processing",
                    batch_num,
                    idx,
                    len(final_docs),
                    doc_id,
                )

        if docs_already_integrated:
            logger.info(
                "Batch %s: Summary - %s/%s docs already have enhanced markdown (will skip processing)",
                batch_num,
                len(docs_already_integrated),
                len(final_docs),
            )

        # Extract images only for docs that need processing
        docs_with_images: list[int] = []
        batch_total_images = 0

        if docs_to_process:
            logger.info("\n" + "Batch %s: Extracting images from %s documents", batch_num, len(docs_to_process))
            for idx, doc_id in enumerate(docs_to_process, 1):
                pdf_path = pdf_map[doc_id]
                base_doc_dir = images_base_dir / f"doc_{doc_id}"

                # Check if we already processed this document (directory exists = processed)
                if base_doc_dir.exists() and not args.force_reextract:
                    has_images = _directory_has_images(base_doc_dir)
                    if has_images:
                        logger.info(
                            "Batch %s: [%s/%s] doc_%s - reusing cached images (%s files)",
                            batch_num,
                            idx,
                            len(docs_to_process),
                            doc_id,
                            len(list(base_doc_dir.glob("*.png"))) + len(list(base_doc_dir.glob("*.jpg"))),
                        )
                    else:
                        logger.info(
                            "Batch %s: [%s/%s] doc_%s - already processed (0 images found previously)",
                            batch_num,
                            idx,
                            len(docs_to_process),
                            doc_id,
                        )
                else:
                    # Extract images - either directory doesn't exist or force_reextract is True
                    if base_doc_dir.exists():
                        shutil.rmtree(base_doc_dir, ignore_errors=True)
                    base_doc_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        logger.info(
                            "Batch %s: [%s/%s] doc_%s - extracting images from PDF",
                            batch_num,
                            idx,
                            len(docs_to_process),
                            doc_id,
                        )
                        count = await asyncio.to_thread(
                            extract_images_from_pdf,
                            pdf_path,
                            base_doc_dir,
                        )
                        logger.info(
                            "Batch %s: [%s/%s] doc_%s - extracted %s images",
                            batch_num,
                            idx,
                            len(docs_to_process),
                            doc_id,
                            count,
                        )
                        if count == 0:
                            logger.info(
                                "Batch %s: [%s/%s] doc_%s - no figures found, will create enhanced markdown without images",
                                batch_num,
                                idx,
                                len(docs_to_process),
                                doc_id,
                            )
                        batch_total_images += count
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.exception(
                            "Batch %s: [%s/%s] doc_%s - image extraction failed: %s",
                            batch_num,
                            idx,
                            len(docs_to_process),
                            doc_id,
                            exc,
                        )
                        continue

                # Include doc even if it has no images - it still needs enhanced markdown created
                docs_with_images.append(doc_id)

            # Docs with images become docs needing integration
            docs_needing_integration = docs_with_images

        if not docs_needing_integration and not docs_already_integrated:
            logger.warning("Batch %s: no documents to process or upload; skipping", batch_num)
            continue

        # If all docs already integrated, skip Gemini and go straight to upload
        if not docs_needing_integration:
            logger.info("Batch %s: all docs already integrated; skipping Gemini, proceeding to upload", batch_num)
            if not args.skip_image_upload:
                upload_summary = await workflow._upload_documents(docs_already_integrated)
                overall_summary.uploaded += upload_summary.uploaded
                overall_summary.skipped += upload_summary.skipped
                overall_summary.failed += upload_summary.failed
                overall_summary.errors.extend(upload_summary.errors)
            overall_processed_docs += len(docs_already_integrated)

            # Clean up batch files before continuing
            logger.info("Batch %s: Cleaning up %s processed documents", batch_num, len(docs_already_integrated))
            cleanup_count = {"pdfs": 0, "md": 0, "images": 0}

            for doc_id in docs_already_integrated:
                # Remove PDFs
                pdf_path = pdf_dir / f"doc_{doc_id}.pdf"
                if pdf_path.exists():
                    try:
                        pdf_path.unlink()
                        cleanup_count["pdfs"] += 1
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", pdf_path.name, exc)

                # Remove markdown files
                md_files = [
                    output_dir / f"doc_{doc_id}.md",
                    output_dir / f"{doc_id}.md",
                ]
                for md_file in md_files:
                    if md_file.exists():
                        try:
                            md_file.unlink()
                            cleanup_count["md"] += 1
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", md_file.name, exc)

                # Remove image directories
                img_dirs = [
                    images_base_dir / f"doc_{doc_id}",
                    images_base_dir / str(doc_id),
                ]
                for img_dir in img_dirs:
                    if img_dir.exists() and img_dir.is_dir():
                        try:
                            shutil.rmtree(img_dir)
                            cleanup_count["images"] += 1
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", img_dir.name, exc)

            logger.info(
                "Batch %s: Cleanup complete - deleted %s PDFs, %s markdown files, %s image directories",
                batch_num,
                cleanup_count["pdfs"],
                cleanup_count["md"],
                cleanup_count["images"],
            )
            continue

        # Run Gemini workflow only for docs needing integration
        logger.info("Batch %s: running Gemini for %s docs", batch_num, len(docs_needing_integration))
        batch_session_id = f"{base_session_id}-{batch_num:03d}"
        upload_summary = await workflow.run(
            session_id=batch_session_id,
            batch_size=args.image_batch_size,
            system_instruction=args.image_system_instruction,
            wait_seconds=args.image_wait_seconds,
            max_retries=args.image_max_retries,
            upload=not args.skip_image_upload,
            allowed_doc_ids=docs_needing_integration,
            force_reintegrate=force_reintegrate,
        )

        # Upload docs that were already integrated
        if docs_already_integrated and not args.skip_image_upload:
            logger.info("Batch %s: uploading %s already-integrated docs", batch_num, len(docs_already_integrated))
            already_integrated_summary = await workflow._upload_documents(docs_already_integrated)
            upload_summary.uploaded += already_integrated_summary.uploaded
            upload_summary.skipped += already_integrated_summary.skipped
            upload_summary.failed += already_integrated_summary.failed
            upload_summary.errors.extend(already_integrated_summary.errors)

        logger.info(
            "Batch %s upload summary: uploaded=%s skipped=%s failed=%s",
            batch_num,
            upload_summary.uploaded,
            upload_summary.skipped,
            upload_summary.failed,
        )

        # Aggregate results
        overall_summary.uploaded += upload_summary.uploaded
        overall_summary.skipped += upload_summary.skipped
        overall_summary.failed += upload_summary.failed
        overall_summary.errors.extend(upload_summary.errors)
        overall_total_images += batch_total_images

        # All docs processed in this batch (both newly integrated and already integrated)
        all_batch_docs = docs_needing_integration + docs_already_integrated
        overall_processed_docs += len(all_batch_docs)

        # Clean up batch files
        if all_batch_docs:
            logger.info("Batch %s: Cleaning up %s processed documents", batch_num, len(all_batch_docs))
            cleanup_count = {"pdfs": 0, "md": 0, "images": 0}

            for doc_id in all_batch_docs:
                # Remove PDFs
                pdf_path = pdf_dir / f"doc_{doc_id}.pdf"
                if pdf_path.exists():
                    try:
                        pdf_path.unlink()
                        cleanup_count["pdfs"] += 1
                        logger.debug("Deleted PDF: %s", pdf_path.name)
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", pdf_path.name, exc)

                # Remove markdown files
                md_files = [
                    output_dir / f"doc_{doc_id}.md",
                    output_dir / f"{doc_id}.md",
                ]
                for md_file in md_files:
                    if md_file.exists():
                        try:
                            md_file.unlink()
                            cleanup_count["md"] += 1
                            logger.debug("Deleted markdown: %s", md_file.name)
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", md_file.name, exc)

                # Remove image directories
                img_dirs = [
                    images_base_dir / f"doc_{doc_id}",
                    images_base_dir / str(doc_id),
                ]
                for img_dir in img_dirs:
                    if img_dir.exists() and img_dir.is_dir():
                        try:
                            shutil.rmtree(img_dir)
                            cleanup_count["images"] += 1
                            logger.debug("Deleted image directory: %s", img_dir.name)
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", img_dir.name, exc)

            logger.info(
                "Batch %s: Cleanup complete - deleted %s PDFs, %s markdown files, %s image directories",
                batch_num,
                cleanup_count["pdfs"],
                cleanup_count["md"],
                cleanup_count["images"],
            )

        # Clean up Gemini batch files for this session
        if docs_needing_integration:
            logger.info("Batch %s: Cleaning up Gemini batch files for session %s", batch_num, batch_session_id)
            cleanup_count_batch = {"batch_files": 0, "description_files": 0}

            # Clean up batch input files
            batch_dir = args.generated_root / args.batch_prefix
            if batch_dir.exists():
                for batch_file in batch_dir.glob(f"*{batch_session_id}*.jsonl"):
                    try:
                        batch_file.unlink()
                        cleanup_count_batch["batch_files"] += 1
                        logger.debug("Deleted batch file: %s", batch_file.name)
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", batch_file.name, exc)

            # Clean up description result files
            descriptions_dir = args.generated_root / "image_descriptions"
            if descriptions_dir.exists():
                for desc_file in descriptions_dir.glob(f"image_descriptions_{batch_session_id}_*.json"):
                    try:
                        desc_file.unlink()
                        cleanup_count_batch["description_files"] += 1
                        logger.debug("Deleted description file: %s", desc_file.name)
                    except Exception as exc:
                        logger.warning("Failed to delete %s: %s", desc_file.name, exc)

            logger.info(
                "Batch %s: Cleaned up %s batch files, %s description files",
                batch_num,
                cleanup_count_batch["batch_files"],
                cleanup_count_batch["description_files"],
            )

        # Update current_min_doc_id for next batch to avoid infinite loops
        if doc_ids:
            current_min_doc_id = max(doc_ids) + 1
            logger.info("Batch %s: Next batch will start from doc_id %s", batch_num, current_min_doc_id)

            # Check if we've exceeded the max range
            if current_min_doc_id > args.max_doc_id:
                logger.info("Batch %s: Reached end of document range (max: %s)", batch_num, args.max_doc_id)
                break

    logger.info("\n" + "=" * 80)
    logger.info("IMAGES-ONLY PIPELINE COMPLETE")
    logger.info("Total batches: %s", batch_num)
    logger.info("Total documents processed: %s", overall_processed_docs)
    logger.info("Total new images extracted: %s", overall_total_images)
    logger.info(
        "Overall upload summary: uploaded=%s skipped=%s failed=%s",
        overall_summary.uploaded,
        overall_summary.skipped,
        overall_summary.failed,
    )
    logger.info("=" * 80)


async def run_integrate_images(args: argparse.Namespace) -> None:
    """Integrate existing image descriptions and upload to Supabase."""
    configure_logging()
    logger = logging.getLogger("pipeline.integrate_images")

    _ensure_data_dirs()

    if not args.session_id:
        logger.error("--session-id is required for integrate-images command")
        return

    descriptions_dir = args.generated_root / "image_descriptions"
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    # Download from GCS if requested
    if args.download:
        logger.info("Downloading results from GCS for session %s", args.session_id)

        # Load tracking file to get job information
        tracking_file = args.generated_root / args.batch_prefix / "batch_jobs_tracking.json"
        if not tracking_file.exists():
            logger.error("Tracking file not found: %s", tracking_file)
            logger.error("Cannot download without job tracking information")
            return

        import json
        with open(tracking_file, "r") as fh:
            tracking_data = json.load(fh)

        # Filter jobs for this session
        all_jobs = tracking_data.get("jobs", [])
        session_jobs = [
            job for job in all_jobs
            if args.session_id in str(job.get("batch_file", ""))
        ]

        if not session_jobs:
            logger.error("No jobs found for session %s in tracking file", args.session_id)
            return

        logger.info("Found %s batch jobs for session %s", len(session_jobs), args.session_id)

        # Initialize clients
        from src.pipeline.gemini import init_client, validate_environment
        from google.cloud import storage

        validate_environment()
        client = init_client()
        gcs_client = storage.Client()

        # Create UploadJob objects for download
        from dataclasses import dataclass
        @dataclass
        class UploadJob:
            batch_file: Path
            job_name: str
            timestamp: str

        jobs = [
            UploadJob(
                batch_file=Path(job["batch_file"]),
                job_name=job["job_name"],
                timestamp=job["timestamp"]
            )
            for job in session_jobs
        ]

        # Download results
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

        download_result = await asyncio.to_thread(
            workflow._download_results,
            client,
            gcs_client,
            args.session_id,
            jobs,
        )

        logger.info(
            "Downloaded %s description files (%s total descriptions)",
            len(download_result.description_files),
            download_result.total_descriptions,
        )

    # Find description files for this session
    description_files = list(descriptions_dir.glob(f"image_descriptions_{args.session_id}_*.json"))
    if not description_files:
        logger.error("No description files found for session %s", args.session_id)
        if args.download:
            logger.error("Download completed but no files were created - check GCS outputs")
        return

    logger.info("Found %s description files for session %s", len(description_files), args.session_id)

    # Determine which doc IDs to process
    doc_ids: set[int] = set()
    for desc_file in description_files:
        import json
        with open(desc_file, "r") as fh:
            records = json.load(fh)
        for record in records:
            key = record.get("key", "")
            if key.startswith("doc_"):
                doc_name = key.split("_page_")[0]
                try:
                    doc_id = int(doc_name.replace("doc_", ""))
                    # Filter by range if specified
                    if args.min_doc_id is not None and doc_id < args.min_doc_id:
                        continue
                    if args.max_doc_id is not None and doc_id > args.max_doc_id:
                        continue
                    doc_ids.add(doc_id)
                except ValueError:
                    continue

    if not doc_ids:
        logger.error("No valid document IDs found in description files")
        return

    logger.info("Processing %s documents from session %s", len(doc_ids), args.session_id)

    # Run integration
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

    integration_result = await asyncio.to_thread(
        workflow._integrate_descriptions,
        descriptions_dir,
        doc_ids,
        getattr(args, "force_reintegrate", False),
        args.session_id,
    )

    logger.info(
        "Integration complete: %s files processed, %s descriptions added",
        integration_result.processed_files,
        integration_result.descriptions_added,
    )

    # Upload to Supabase if requested
    if not args.skip_upload:
        upload_summary = await workflow._upload_documents(doc_ids)
        logger.info(
            "Upload summary: uploaded=%s skipped=%s failed=%s",
            upload_summary.uploaded,
            upload_summary.skipped,
            upload_summary.failed,
        )
        for error in upload_summary.errors:
            logger.error("Upload error: %s", error)


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
    full_parser.add_argument("doc_batch_size", type=int, nargs="?", default=10, help="Number of documents to fetch and process per batch (default: 10)")
    full_parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "config.yaml")
    full_parser.add_argument("--randomize", action="store_true")
    full_parser.add_argument("--random-seed", type=int, default=42)
    full_parser.add_argument("--run-all-images", action="store_true")
    full_parser.add_argument("--batch-size", type=int, default=1)
    full_parser.add_argument("--chunk-page-limit", type=int, default=30)
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
    markdown_parser.add_argument("--chunk-page-limit", type=int, default=30)
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
        help="Run images-only workflow for documents missing DOCLING_IMG",
    )
    images_parser.add_argument("min_doc_id", type=int, help="Minimum document ID to consider")
    images_parser.add_argument("max_doc_id", type=int, help="Maximum document ID to consider")
    images_parser.add_argument(
        "--doc-batch-size",
        type=int,
        default=10,
        help="Number of documents to process per batch",
    )
    images_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum total number of documents to process",
    )
    images_parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "config.yaml",
        help="Pipeline configuration file for Supabase/API access",
    )
    images_parser.add_argument(
        "--input-folder",
        type=Path,
        default=DATA_DIR / "to_process",
        help="Directory where PDFs are stored/ downloaded",
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
        default=DATA_DIR / "images",
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
        "--skip-pdf-download",
        action="store_true",
        help="Assume PDFs already exist locally and skip Supabase download",
    )
    images_parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-extract images even if cached images already exist",
    )
    images_parser.add_argument(
        "--force-reintegrate",
        action="store_true",
        help="Re-integrate descriptions even if enhanced markdown already exists",
    )
    images_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (disable GPU even if available)",
    )
    images_parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of documents processed",
    )

    integrate_parser = subparsers.add_parser(
        "integrate-images",
        help="Integrate existing image descriptions and upload to Supabase (requires session-id)",
    )
    integrate_parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session identifier from previous Gemini batch run",
    )
    integrate_parser.add_argument(
        "--min-doc-id",
        type=int,
        default=None,
        help="Minimum document ID to process (optional filter)",
    )
    integrate_parser.add_argument(
        "--max-doc-id",
        type=int,
        default=None,
        help="Maximum document ID to process (optional filter)",
    )
    integrate_parser.add_argument(
        "--output-folder",
        type=Path,
        default=DATA_DIR / "processed",
        help="Directory containing Docling markdown files",
    )
    integrate_parser.add_argument(
        "--enhanced-dir",
        type=Path,
        default=DATA_DIR / "processed_images",
        help="Directory for markdown merged with image descriptions",
    )
    integrate_parser.add_argument(
        "--images-dir",
        type=Path,
        default=DATA_DIR / "images",
        help="Directory containing extracted images",
    )
    integrate_parser.add_argument(
        "--generated-root",
        type=Path,
        default=WORKSPACE_ROOT / ".generated",
        help="Directory containing image descriptions",
    )
    integrate_parser.add_argument(
        "--image-format",
        type=str,
        default="detailed",
        help="Image description formatting mode",
    )
    integrate_parser.add_argument(
        "--download",
        action="store_true",
        help="Download results from GCS before integrating",
    )
    integrate_parser.add_argument(
        "--force-reintegrate",
        action="store_true",
        help="Re-integrate descriptions even if enhanced markdown already exists",
    )
    integrate_parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading merged markdown to Supabase",
    )
    integrate_parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash-001",
        help="Gemini model name (for workflow initialization)",
    )
    integrate_parser.add_argument(
        "--gcs-input-prefix",
        type=str,
        default="gemini_batches/input",
        help="GCS prefix (for workflow initialization)",
    )
    integrate_parser.add_argument(
        "--gcs-output-prefix",
        type=str,
        default="gemini_batches/output",
        help="GCS prefix (for workflow initialization)",
    )
    integrate_parser.add_argument(
        "--batch-prefix",
        type=str,
        default="image_description_batches",
        help="Batch prefix (for workflow initialization)",
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
            "integrate-images": run_integrate_images,
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
