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
    ):
        """
        Initialize the Docling converter.

        Args:
            artifacts_path: Path for temporary artifacts. If None, uses Docling's default.
            add_page_numbers: Whether to add page numbers to the markdown output.
            use_gpu: Whether to use GPU acceleration if available (default: True).
        """
        # Let Docling handle artifacts_path by default for automatic model downloads
        self.artifacts_path = artifacts_path
        self.add_page_numbers = add_page_numbers
        self.use_gpu = use_gpu
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
        """Get or create the document converter with GPU settings."""
        if self._converter is None:
            # Only configure GPU settings, let Docling handle everything else automatically
            if self.use_gpu:
                device = self._detect_device()

                # Only configure if we actually have GPU or want specific CPU settings
                if device != AcceleratorDevice.CPU or self.artifacts_path:
                    accelerator_options = AcceleratorOptions(
                        num_threads=4 if device == AcceleratorDevice.CPU else 1,
                        device=device
                    )

                    # Build minimal pipeline options
                    pipeline_kwargs = {
                        "accelerator_options": accelerator_options,
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
                else:
                    # Use completely default configuration for CPU
                    self._converter = DocumentConverter()
            else:
                # CPU-only mode with explicit configuration
                accelerator_options = AcceleratorOptions(
                    num_threads=4,
                    device=AcceleratorDevice.CPU
                )

                pipeline_kwargs = {"accelerator_options": accelerator_options}
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

    def convert_pdf(self, pdf_source: Union[str, Path]) -> Tuple[str, DoclingDocument]:
        """
        Convert a PDF file to Markdown.

        Args:
            pdf_source: Path to the PDF file to convert.

        Returns:
            Tuple containing the markdown content and the DoclingDocument object.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF or conversion fails.
        """
        pdf_path = Path(pdf_source)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        try:
            converter = self._get_converter()
            doc = converter.convert(pdf_path)
            document = doc.document

            if self.add_page_numbers:
                result_markdown = ""
                pages = list(document.pages)
                for i in range(len(pages)):
                    result_markdown += f"Page {i + 1}\n"
                    result_markdown += document.export_to_markdown(page_no=i)
                    result_markdown += f"\n\n<----PAGE {i + 1}---->\n\n"
            else:
                result_markdown = document.export_to_markdown()

            return result_markdown, document

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
        result_markdown, _ = self.convert_pdf(pdf_source)
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
) -> str:
    """
    Convert a PDF file to Markdown using Docling.

    Args:
        pdf_path: Path to the PDF file to convert.
        artifacts_path: Path for temporary artifacts. If None, uses system temp directory.
        add_page_numbers: Whether to add page numbers to the markdown output.

    Returns:
        The markdown content as a string.
    """
    converter = DoclingConverter(
        artifacts_path=artifacts_path, add_page_numbers=add_page_numbers
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
