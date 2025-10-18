"""
Standalone Docling PDF Converter

This module provides a standalone PDF to Markdown converter using Docling.
It extracts the core conversion logic from the existing codebase for use in other projects.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


class DoclingConverter:
    """
    Standalone PDF to Markdown converter using Docling.

    This class provides a simple interface for converting PDF files to Markdown
    using the Docling library with optimized settings for memory efficiency.
    """

    def __init__(
        self,
        artifacts_path: Optional[str] = None,
        add_page_numbers: bool = False,
        use_gpu: bool = True,
        table_mode: str = "accurate",
        images_scale: float = 2.0,
        do_cell_matching: bool = True,
        ocr_confidence_threshold: float = 0.1,
    ):
        """
        Initialize the Docling converter.

        Args:
            artifacts_path: Path for temporary artifacts. If None, uses Docling's default.
            add_page_numbers: Whether to add page numbers to the markdown output.
            use_gpu: Whether to use GPU acceleration if available (default: True).
            table_mode: Table structure recognition mode - "accurate" (default, highest quality) or "fast".
            images_scale: Image scaling factor for processing (default: 2.0, higher = better quality).
            do_cell_matching: Enable precise cell matching in tables (default: True for best quality).
            ocr_confidence_threshold: OCR confidence threshold, 0-1 (default: 0.1, lower = more text captured).
        """
        # Let Docling handle artifacts_path by default for automatic model downloads
        self.artifacts_path = artifacts_path
        self.add_page_numbers = add_page_numbers
        self.use_gpu = use_gpu
        self.table_mode = table_mode
        self.images_scale = images_scale
        self.do_cell_matching = do_cell_matching
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self._converter = None

    def _detect_device(self) -> AcceleratorDevice:
        """Detect if GPU is available and return appropriate device."""
        if not self.use_gpu:
            return AcceleratorDevice.CPU

        try:
            import torch
            if torch.cuda.is_available():
                return AcceleratorDevice.CUDA
            elif torch.backends.mps.is_available():
                return AcceleratorDevice.MPS  # Apple Silicon GPU
        except ImportError:
            pass

        return AcceleratorDevice.CPU

    def _get_converter(self) -> DocumentConverter:
        """Get or create the document converter with GPU and quality settings."""
        if self._converter is None:
            device = self._detect_device() if self.use_gpu else AcceleratorDevice.CPU

            # Configure accelerator options
            accelerator_options = AcceleratorOptions(
                num_threads=4 if device == AcceleratorDevice.CPU else 1,
                device=device
            )

            # Configure table structure options for highest quality
            table_structure_options = TableStructureOptions(
                do_cell_matching=self.do_cell_matching,
                mode=TableFormerMode.ACCURATE if self.table_mode.lower() == "accurate" else TableFormerMode.FAST
            )

            # Configure OCR options for highest quality
            ocr_options = EasyOcrOptions(
                confidence_threshold=self.ocr_confidence_threshold,
                force_full_page_ocr=False,  # Use smart OCR detection
            )

            # Build pipeline options with quality settings
            pipeline_kwargs = {
                "accelerator_options": accelerator_options,
                "table_structure_options": table_structure_options,
                "ocr_options": ocr_options,
                "images_scale": self.images_scale,
                "do_table_structure": True,  # Enable table structure recognition
                "do_ocr": True,  # Enable OCR
                "generate_picture_images": True,  # Enable picture image extraction
            }

            # Only set artifacts_path if explicitly provided
            if self.artifacts_path:
                pipeline_kwargs["artifacts_path"] = self.artifacts_path

            pipeline_options = PdfPipelineOptions(**pipeline_kwargs)

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                }
            )

        return self._converter

    def extract_images(
        self,
        document: DoclingDocument,
        output_dir: Path,
        doc_id: str
    ) -> int:
        """
        Extract images from a DoclingDocument and save them to disk.

        Args:
            document: The DoclingDocument to extract images from.
            output_dir: Directory to save extracted images.
            doc_id: Document ID for organizing images.

        Returns:
            Number of images extracted.
        """
        # Create output directory for this document
        doc_output_dir = output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0

        # Iterate through all pictures in the document
        pictures = list(document.pictures)

        for picture in pictures:
            try:
                # Get page number from provenance
                if hasattr(picture, 'prov') and picture.prov:
                    page_num = picture.prov[0].page_no
                else:
                    continue

                # Get image data using get_image method
                pil_image = picture.get_image(document)

                if pil_image is None:
                    continue

                # Image index for this document
                img_idx = image_count + 1

                # Save image
                image_filename = f"page_{page_num:03d}_img_{img_idx:02d}.png"
                image_path = doc_output_dir / image_filename
                pil_image.save(image_path, "PNG")

                image_count += 1

            except Exception:
                continue

        return image_count

    def convert_pdf(self, pdf_source: Union[str, Path]) -> Tuple[str, DoclingDocument, int]:
        """
        Convert a PDF file to Markdown.

        Args:
            pdf_source: Path to the PDF file to convert.

        Returns:
            Tuple containing the markdown content, the DoclingDocument object, and page count.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF or conversion fails.
        """
        import subprocess

        pdf_path = Path(pdf_source)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        # Validate actual file type using the `file` command
        try:
            result = subprocess.run(
                ['file', '--mime-type', '-b', str(pdf_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            mime_type = result.stdout.strip()

            if not mime_type.startswith('application/pdf'):
                raise ValueError(f"File is not a valid PDF (detected as {mime_type}): {pdf_path}")
        except subprocess.TimeoutExpired:
            raise ValueError(f"Timeout while validating file type: {pdf_path}")
        except subprocess.CalledProcessError as e:
            # If file command fails, log but continue (might be OS-specific issue)
            pass

        try:
            converter = self._get_converter()
            doc = converter.convert(pdf_path)
            document = doc.document

            pages = list(document.pages)
            page_count = len(pages)

            # Always add page markers
            result_markdown = ""
            for i in range(page_count):
                if self.add_page_numbers:
                    result_markdown += f"Page {i + 1}\n"
                result_markdown += document.export_to_markdown(page_no=i)
                if i < page_count - 1:  # Don't add marker after last page
                    result_markdown += f"\n\n<!-- PAGE {i + 1} -->\n\n"

            return result_markdown, document, page_count

        except (FileNotFoundError, ValueError) as e:
            raise e
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Failed to convert PDF {pdf_path}: {str(e)}") from e

    def convert_pdf_to_markdown(self, pdf_source: Union[str, Path]) -> str:
        """
        Convert a PDF file to Markdown string.

        Args:
            pdf_source: Path to the PDF file to convert.

        Returns:
            The markdown content as a string.
        """
        result_markdown, _, _ = self.convert_pdf(pdf_source)
        return result_markdown

    def cleanup(self):
        """Clean up resources and temporary files."""
        if self.artifacts_path and os.path.exists(self.artifacts_path):
            try:
                import shutil

                shutil.rmtree(self.artifacts_path, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors


# Convenience function for simple conversions
def convert_pdf_to_markdown(
    pdf_path: Union[str, Path],
    artifacts_path: Optional[str] = None,
    add_page_numbers: bool = False,
    table_mode: str = "accurate",
    images_scale: float = 2.0,
    do_cell_matching: bool = True,
    ocr_confidence_threshold: float = 0.1,
) -> str:
    """
    Convert a PDF file to Markdown using Docling.

    Args:
        pdf_path: Path to the PDF file to convert.
        artifacts_path: Path for temporary artifacts. If None, uses system temp directory.
        add_page_numbers: Whether to add page numbers to the markdown output.
        table_mode: Table structure recognition mode - "accurate" (default) or "fast".
        images_scale: Image scaling factor for processing (default: 2.0).
        do_cell_matching: Enable precise cell matching in tables (default: True).
        ocr_confidence_threshold: OCR confidence threshold, 0-1 (default: 0.1).

    Returns:
        The markdown content as a string.
    """
    converter = DoclingConverter(
        artifacts_path=artifacts_path,
        add_page_numbers=add_page_numbers,
        table_mode=table_mode,
        images_scale=images_scale,
        do_cell_matching=do_cell_matching,
        ocr_confidence_threshold=ocr_confidence_threshold,
    )
    try:
        return converter.convert_pdf_to_markdown(pdf_path)
    finally:
        converter.cleanup()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python docling_converter.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    try:
        markdown = convert_pdf_to_markdown(pdf_file)
        print(markdown)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
