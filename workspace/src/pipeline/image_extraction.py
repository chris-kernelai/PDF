"""Image extraction powered by Docling's StandardPdfPipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import PictureItem

logger = logging.getLogger(__name__)

_CONVERTER: Optional[DocumentConverter] = None


def _get_converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False
    pipeline_opts.generate_page_images = False
    pipeline_opts.generate_picture_images = True
    pipeline_opts.images_scale = 2.0
    pipeline_opts.do_table_structure = False

    fmt_option = PdfFormatOption(
        pipeline_options=pipeline_opts,
        pipeline_cls=StandardPdfPipeline,
    )

    _CONVERTER = DocumentConverter(
        format_options={InputFormat.PDF: fmt_option}
    )
    return _CONVERTER


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> int:
    """Extract page snapshots and cropped figures from a PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = _get_converter()

    result = converter.convert(str(pdf_path))
    document = result.document
    stem = pdf_path.stem

    figure_index = 0
    for element, _ in document.iterate_items():
        if not isinstance(element, PictureItem):
            continue

        pil_image = element.get_image(document)
        if pil_image is None:
            continue

        page_num = 0
        try:
            if element.prov:
                page_num = int(element.prov[0].page_no)
        except Exception:  # pragma: no cover - defensive
            page_num = 0

        figure_index += 1
        filename = output_dir / f"{stem}_page_{page_num:03d}_img_{figure_index:02d}.png"
        pil_image.save(filename, "PNG")

    logger.info("Extracted %s figure images from %s", figure_index, pdf_path.name)
    return figure_index


def extract_images_from_directory(pdf_dir: Path, output_dir: Path) -> int:
    total = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_output = output_dir / pdf_path.stem
        if doc_output.exists():
            for existing in doc_output.glob("*"):
                if existing.is_file():
                    existing.unlink()
        else:
            doc_output.mkdir(parents=True, exist_ok=True)
        total += extract_images_from_pdf(pdf_path, doc_output)

    return total


__all__ = ["extract_images_from_pdf", "extract_images_from_directory"]
