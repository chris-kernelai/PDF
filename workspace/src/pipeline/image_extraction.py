"""Utilities for running Docling image-extraction-only pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from docling.pipeline import Pipeline
from docling.models import ImageExtractionConfig

logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> int:
    """Extract images from a single PDF using Docling's image extraction pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(
        steps=[
            (
                "extract_images",
                ImageExtractionConfig(output_dir=str(output_dir)),
            )
        ]
    )

    pipeline.run(str(pdf_path))
    count = sum(1 for child in output_dir.glob("*.*") if child.is_file())
    logger.info("Extracted %s images from %s", count, pdf_path.name)
    return count


def extract_images_from_directory(pdf_dir: Path, output_dir: Path) -> int:
    """Extract images for all PDFs in the given directory."""
    total = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_output = output_dir / pdf_path.stem
        doc_output.mkdir(parents=True, exist_ok=True)
        total += extract_images_from_pdf(pdf_path, doc_output)

    return total


__all__ = ["extract_images_from_pdf", "extract_images_from_directory"]
