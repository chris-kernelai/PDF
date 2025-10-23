"""Image extraction powered by Docling's StandardPdfPipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import PictureItem

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# Chunk size for processing large PDFs (pages)
CHUNK_SIZE = int(os.environ.get("PDF_CHUNK_SIZE", "30"))
# GPU memory limit (fraction, 0-1)
GPU_MEMORY_LIMIT = float(os.environ.get("GPU_MEMORY_LIMIT", "0.8"))

_CONVERTER: Optional[DocumentConverter] = None


def _set_gpu_memory_limit() -> None:
    """Set GPU memory limit to avoid OOM errors."""
    if not HAS_TORCH:
        return
    
    if torch.cuda.is_available():
        try:
            # Set memory fraction for all GPUs
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT, device=i)
            logger.info(f"Set GPU memory limit to {GPU_MEMORY_LIMIT*100:.0f}% on {torch.cuda.device_count()} device(s)")
        except Exception as e:
            logger.warning(f"Could not set GPU memory limit: {e}")


def _get_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF."""
    if not HAS_PYPDF:
        logger.warning("pypdf not available, cannot chunk PDFs")
        return -1
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        logger.warning(f"Could not get page count for {pdf_path}: {e}")
        return -1


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


def _extract_images_from_chunk(pdf_path: Path, output_dir: Path, start_page: int, end_page: int, figure_index_offset: int = 0) -> int:
    """Extract images from a specific page range of a PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary chunked PDF
    if not HAS_PYPDF:
        # Fallback: process entire PDF if pypdf not available
        logger.warning("pypdf not available, processing entire PDF")
        return _extract_images_full(pdf_path, output_dir)
    
    temp_pdf = output_dir / f"_temp_chunk_{start_page}_{end_page}.pdf"
    try:
        # Extract page range to temporary PDF
        with open(pdf_path, 'rb') as input_file:
            reader = pypdf.PdfReader(input_file)
            writer = pypdf.PdfWriter()
            
            # Add pages in range (0-indexed)
            for page_idx in range(start_page, min(end_page, len(reader.pages))):
                writer.add_page(reader.pages[page_idx])
            
            # Write temporary PDF
            with open(temp_pdf, 'wb') as output_file:
                writer.write(output_file)
        
        # Process the chunk
        converter = _get_converter()
        result = converter.convert(str(temp_pdf))
        document = result.document
        stem = pdf_path.stem
        
        figure_index = figure_index_offset
        for element, _ in document.iterate_items():
            if not isinstance(element, PictureItem):
                continue
            
            pil_image = element.get_image(document)
            if pil_image is None:
                continue
            
            # Get page number from chunk and adjust for actual document position
            chunk_page_num = 0
            try:
                if element.prov:
                    chunk_page_num = int(element.prov[0].page_no)
            except Exception:
                chunk_page_num = 0
            
            # Adjust page number to actual position in original PDF
            actual_page_num = start_page + chunk_page_num
            
            figure_index += 1
            filename = output_dir / f"{stem}_page_{actual_page_num:03d}_img_{figure_index:02d}.png"
            pil_image.save(filename, "PNG")
        
        return figure_index - figure_index_offset
    
    finally:
        # Clean up temporary file
        if temp_pdf.exists():
            temp_pdf.unlink()


def _extract_images_full(pdf_path: Path, output_dir: Path) -> int:
    """Extract images from entire PDF (no chunking)."""
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
        except Exception:
            page_num = 0
        
        figure_index += 1
        filename = output_dir / f"{stem}_page_{page_num:03d}_img_{figure_index:02d}.png"
        pil_image.save(filename, "PNG")
    
    return figure_index


def extract_images_from_pdf(pdf_path: Path, output_dir: Path, chunk_size: int = CHUNK_SIZE) -> int:
    """Extract page snapshots and cropped figures from a PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save extracted images
        chunk_size: Number of pages to process at once (0 = no chunking)
    
    Returns:
        Total number of images extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set GPU memory limit
    _set_gpu_memory_limit()
    
    # Get page count
    page_count = _get_page_count(pdf_path)
    
    # If chunking disabled or can't get page count, process full PDF
    if chunk_size <= 0 or page_count <= 0 or page_count <= chunk_size:
        logger.info(f"Processing {pdf_path.name} as single document ({page_count} pages)")
        total_images = _extract_images_full(pdf_path, output_dir)
        logger.info(f"Extracted {total_images} figure images from {pdf_path.name}")
        return total_images
    
    # Process in chunks
    logger.info(f"Processing {pdf_path.name} in chunks ({page_count} pages, chunk_size={chunk_size})")
    total_images = 0
    figure_offset = 0
    
    for start_page in range(0, page_count, chunk_size):
        end_page = min(start_page + chunk_size, page_count)
        logger.info(f"  Chunk {start_page//chunk_size + 1}: pages {start_page+1}-{end_page}")
        
        chunk_images = _extract_images_from_chunk(
            pdf_path,
            output_dir,
            start_page,
            end_page,
            figure_offset
        )
        total_images += chunk_images
        figure_offset += chunk_images
        
        logger.info(f"  Extracted {chunk_images} images from chunk (total: {total_images})")
    
    logger.info(f"Extracted {total_images} figure images from {pdf_path.name} (chunked)")
    return total_images


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
