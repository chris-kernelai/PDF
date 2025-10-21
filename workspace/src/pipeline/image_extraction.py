"""Docling-based utilities for extracting images from PDFs without markdown."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption

logger = logging.getLogger(__name__)


def _create_converter(artifacts_path: Path) -> DocumentConverter:
    fmt_option = PdfFormatOption().set_optional_field_default()
    pipeline_options = fmt_option.pipeline_options

    pipeline_options.keep_images = True
    pipeline_options.keep_text = False
    pipeline_options.keep_tables = False
    pipeline_options.keep_annotations = False
    pipeline_options.artifacts_path = artifacts_path

    return DocumentConverter(format_options={InputFormat.PDF: fmt_option})


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = _create_converter(output_dir)
    result = converter.convert(str(pdf_path))
    document = result.document

    image_count = 0
    for page in getattr(document, "pages", []):
        page_number = getattr(page, "page_number", image_count + 1)
        for picture in getattr(page, "images", []):
            src_path = getattr(picture, "path", None)
            if not src_path:
                continue
            src = Path(src_path)
            if not src.exists():
                continue
            image_count += 1
            target_name = f"page_{page_number:03d}_img_{image_count:02d}{src.suffix}"
            shutil.copy2(src, output_dir / target_name)

    logger.info("Extracted %s images from %s", image_count, pdf_path.name)
    return image_count


def extract_images_from_directory(pdf_dir: Path, output_dir: Path) -> int:
    total = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_output = output_dir / pdf_path.stem
        if doc_output.exists():
            shutil.rmtree(doc_output)
        doc_output.mkdir(parents=True, exist_ok=True)
        total += extract_images_from_pdf(pdf_path, doc_output)

    return total


__all__ = ["extract_images_from_pdf", "extract_images_from_directory"]
