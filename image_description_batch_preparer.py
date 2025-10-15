#!/usr/bin/env python3
"""
image_description_batch_preparer.py

Step 1: Extract images from PDFs and prepare batch requests for Gemini Flash
- Extracts images from PDFs using Docling
- Creates JSONL batch files for Gemini API
- Supports both Developer API and Vertex AI formats
- Groups images into batches of configurable size

Usage:
    python image_description_batch_preparer.py developer
    python image_description_batch_preparer.py vertex
    python image_description_batch_preparer.py developer --batch-size 100 --images-per-pdf 10
"""

import argparse
import base64
import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling_core.types.doc.document import DoclingDocument

# Import processing logger
from processing_logger import ProcessingLogger

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Image Extraction
# -------------------------------------------------------------------


class ImageExtractor:
    """Extract images from PDFs using Docling"""

    def __init__(self, use_gpu: bool = False):
        """
        Initialize image extractor.

        Args:
            use_gpu: Whether to use GPU acceleration (default: False for faster image-only extraction)
        """
        self.use_gpu = use_gpu
        self._converter = None

    def _get_converter(self) -> DocumentConverter:
        """Get or create the document converter"""
        if self._converter is None:
            device = AcceleratorDevice.CPU
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = AcceleratorDevice.CUDA
                    elif torch.backends.mps.is_available():
                        device = AcceleratorDevice.MPS
                except ImportError:
                    pass

            accelerator_options = AcceleratorOptions(
                num_threads=4 if device == AcceleratorDevice.CPU else 1,
                device=device
            )

            # Pipeline configuration for image extraction
            pipeline_options = PdfPipelineOptions(
                accelerator_options=accelerator_options,
                do_table_structure=False,  # Don't need table processing
                do_ocr=False,  # Don't need OCR
                images_scale=2.0,  # Medium quality for extraction
                generate_page_images=True,  # Enable page image generation
                generate_picture_images=True,  # Enable picture image extraction
            )

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                }
            )

        return self._converter

    def extract_images_from_pdf(
        self, pdf_path: Path, output_dir: Path, max_images: Optional[int] = None
    ) -> Tuple[List[Dict[str, str]], int, float]:
        """
        Extract images from a PDF file.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            max_images: Maximum number of images to extract per PDF (None = all)

        Returns:
            Tuple of (image_list, page_count, duration_seconds)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting images from {pdf_path.name}...")
        start_time = datetime.now()

        try:
            converter = self._get_converter()
            doc = converter.convert(pdf_path)
            document = doc.document

            # Get page count
            page_count = len(document.pages) if hasattr(document, 'pages') else 0

            # Extract document ID from filename (assumes doc_12345.pdf format)
            doc_id = pdf_path.stem.replace("doc_", "") if pdf_path.stem.startswith("doc_") else pdf_path.stem

            # Create output directory for this document
            doc_output_dir = output_dir / doc_id
            doc_output_dir.mkdir(parents=True, exist_ok=True)

            extracted_images = []
            image_count = 0

            # Iterate through all pictures in the document
            pictures = list(document.pictures)

            for picture in pictures:
                if max_images and image_count >= max_images:
                    break

                try:
                    # Get page number from provenance
                    if hasattr(picture, 'prov') and picture.prov:
                        page_num = picture.prov[0].page_no
                    else:
                        logger.warning("Picture has no provenance info, skipping")
                        continue

                    # Get image data using get_image method
                    pil_image = picture.get_image(document)

                    if pil_image is None:
                        logger.warning(f"Got None image on page {page_num}")
                        continue

                    # Image index for this document
                    img_idx = image_count + 1

                    # Save image
                    image_filename = f"page_{page_num:03d}_img_{img_idx:02d}.png"
                    image_path = doc_output_dir / image_filename
                    pil_image.save(image_path, "PNG")

                    extracted_images.append({
                        'image_path': str(image_path),
                        'page_number': page_num,
                        'image_index': img_idx,
                        'document_id': doc_id,
                        'filename': image_filename,
                    })
                    image_count += 1
                    logger.debug(f"  Extracted: {image_filename}")

                except Exception as e:
                    logger.warning(f"Failed to extract picture: {e}")
                    continue

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"  Extracted {len(extracted_images)} images from {pdf_path.name}")
            return extracted_images, page_count, duration

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            return [], 0, duration


# -------------------------------------------------------------------
# Batch Request Preparation
# -------------------------------------------------------------------


def encode_image_base64(image_path: str, max_size: int = 1024, quality: int = 85) -> str:
    """
    Encode image file as base64 string with compression.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height) in pixels
        quality: JPEG quality (1-100, default 85)

    Returns:
        Base64 encoded string
    """
    try:
        # Open image
        img = Image.open(image_path)

        # Convert to RGB if necessary (for JPEG)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if too large
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Compress to JPEG in memory
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception as e:
        logger.error(f"Failed to compress {image_path}: {e}")
        # Fallback to original encoding
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def create_batch_request(
    image_info: Dict[str, str],
    mode: str,
    batch_uuid: str,
    system_instruction: Optional[str] = None
) -> Dict:
    """
    Create a single batch request for image description.

    Args:
        image_info: Image metadata dict
        mode: 'developer' or 'vertex'
        batch_uuid: UUID for this batch run
        system_instruction: Optional system instruction for the model

    Returns:
        Batch request dict in Gemini format
    """
    # Encode image
    image_base64 = encode_image_base64(image_info['image_path'])

    # Default prompt for financial document image description
    user_prompt = """Provide a terse, factual description of this image.

Report exactly what is shown:
- Content type (chart, table, diagram, etc.)
- All visible text, labels, and legends
- All numerical data, values, and units
- Axis labels and scales

Do not interpret or analyze. State only what is directly visible. Be concise."""

    # Build request key (unique identifier) - NOW WITH UUID!
    request_key = f"{batch_uuid}_{image_info['document_id']}_page_{image_info['page_number']:03d}_img_{image_info['image_index']:02d}"

    # Build contents
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_base64
                    }
                },
                {"text": user_prompt}
            ]
        }
    ]

    # Build request
    request = {
        "contents": contents,
    }

    # Add system instruction if provided
    if system_instruction:
        request["system_instruction"] = {"parts": [{"text": system_instruction}]}

    # Format for Developer vs Vertex mode
    if mode == "developer":
        batch_request = {
            "key": request_key,
            "request": request
        }
    else:  # vertex
        batch_request = {
            "custom_id": request_key,
            "request": {
                "contents": contents,
            }
        }
        if system_instruction:
            batch_request["request"]["system_instruction"] = {"parts": [{"text": system_instruction}]}

    return batch_request


def write_batch_file(
    batch_requests: List[Dict],
    output_path: Path,
    batch_num: int,
    total_docs: int
) -> None:
    """Write batch requests to JSONL file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    logger.info(f"✅ Created batch file: {output_path.name} ({len(batch_requests)} images from {total_docs} documents)")


# -------------------------------------------------------------------
# Main Processing
# -------------------------------------------------------------------


def process_pdfs(
    input_folder: Path,
    output_folder: Path,
    mode: str,
    batch_size: int = 100,
    images_per_pdf: Optional[int] = None,
    system_instruction: Optional[str] = None,
    use_gpu: bool = False,
) -> bool:
    """
    Process PDFs to extract images and create batch files.

    Args:
        input_folder: Folder containing PDF files
        output_folder: Output folder for batch files
        mode: 'developer' or 'vertex'
        batch_size: Number of images per batch file
        images_per_pdf: Max images to extract per PDF (None = all)
        system_instruction: Optional system instruction
        use_gpu: Whether to use GPU for extraction

    Returns:
        True if successful
    """
    # Generate UUID for this batch run
    batch_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars for brevity
    logger.info(f"🆔 Batch run UUID: {batch_uuid}")

    # Initialize processing logger
    proc_logger = ProcessingLogger()

    # Create output directories
    images_dir = output_folder / "extracted_images"
    batches_dir = output_folder / "image_description_batches"
    images_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Find PDF files
    pdf_files = sorted(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {input_folder}")
        return False

    logger.info(f"📄 Found {len(pdf_files)} PDF files")

    # Extract images from all PDFs
    extractor = ImageExtractor(use_gpu=use_gpu)
    all_images = []
    doc_stats = {}  # Track stats per document

    for pdf_file in pdf_files:
        images, page_count, duration = extractor.extract_images_from_pdf(
            pdf_file,
            images_dir,
            max_images=images_per_pdf
        )

        if images:
            doc_id = images[0]['document_id']
            doc_stats[doc_id] = {
                'pages': page_count,
                'images_extracted': len(images),
                'extraction_duration': duration
            }

            # Log extraction
            proc_logger.log_extraction(
                doc_id=doc_id,
                pages=page_count,
                images_extracted=len(images),
                duration_seconds=duration,
                status="success"
            )

        all_images.extend(images)

    if not all_images:
        logger.error("No images extracted from PDFs")
        return False

    logger.info(f"🖼️  Total images extracted: {len(all_images)}")

    # Create batch requests
    logger.info("📦 Creating batch requests...")
    batch_requests = []

    for image_info in all_images:
        try:
            request = create_batch_request(image_info, mode, batch_uuid, system_instruction)
            batch_requests.append(request)
        except Exception as e:
            logger.error(f"Failed to create request for {image_info['filename']}: {e}")
            continue

    if not batch_requests:
        logger.error("No batch requests created")
        return False

    # Split into batch files
    logger.info(f"✂️  Splitting into batches of {batch_size}...")
    batch_files_created = []
    batch_creation_start = datetime.now()

    for i in range(0, len(batch_requests), batch_size):
        batch_num = (i // batch_size) + 1
        batch_chunk = batch_requests[i:i + batch_size]

        # Count unique documents in this batch
        unique_docs = len(set(req.get('key', req.get('custom_id', '')).split('_')[0] for req in batch_chunk))

        # Create batch file
        batch_filename = f"image_description_batch_{batch_num:03d}_imgs_{len(batch_chunk):04d}.jsonl"
        batch_path = batches_dir / batch_filename

        write_batch_file(batch_chunk, batch_path, batch_num, unique_docs)
        batch_files_created.append(batch_path)

    batch_creation_duration = (datetime.now() - batch_creation_start).total_seconds()

    # Log batch creation per document
    images_by_doc = {}
    for img in all_images:
        doc_id = img['document_id']
        images_by_doc[doc_id] = images_by_doc.get(doc_id, 0) + 1

    for doc_id, img_count in images_by_doc.items():
        proc_logger.log_batch_creation(
            doc_id=doc_id,
            batch_uuid=batch_uuid,
            images_sent=img_count,
            duration_seconds=batch_creation_duration / len(images_by_doc),  # Approximate
            status="success"
        )

    # Collect document IDs
    doc_ids = list(set(img['document_id'] for img in all_images))

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "batch_uuid": batch_uuid,
        "mode": mode,
        "total_pdfs": len(pdf_files),
        "total_images": len(all_images),
        "total_batches": len(batch_files_created),
        "batch_size": batch_size,
        "images_per_pdf": images_per_pdf,
        "document_ids": doc_ids,
        "pdf_files": [str(p.name) for p in pdf_files],
        "batch_files": [str(b.name) for b in batch_files_created],
    }

    metadata_path = batches_dir / "batch_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"📝 Metadata saved to: {metadata_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("✅ BATCH PREPARATION COMPLETE")
    print("=" * 60)
    print(f"📄 PDFs processed: {len(pdf_files)}")
    print(f"🖼️  Images extracted: {len(all_images)}")
    print(f"📦 Batch files created: {len(batch_files_created)}")
    print(f"📁 Batches directory: {batches_dir}")
    print(f"📁 Images directory: {images_dir}")
    print("\nNext step:")
    print(f"  python NEW_GEMINI_batch_uploader.py {mode} --batch-prefix {batches_dir.name}")
    print("=" * 60)

    return True


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from PDFs and prepare batch requests for Gemini Flash"
    )
    parser.add_argument(
        "mode",
        choices=["developer", "vertex"],
        help="API mode: 'developer' for Gemini Developer API or 'vertex' for Vertex AI",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=Path("to_process"),
        help="Input folder containing PDFs (default: to_process)",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path(".generated"),
        help="Output folder for batches (default: .generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of images per batch file (default: 100)",
    )
    parser.add_argument(
        "--images-per-pdf",
        type=int,
        default=None,
        help="Max images to extract per PDF (default: all)",
    )
    parser.add_argument(
        "--system-instruction",
        type=str,
        default=None,
        help="Optional system instruction for the model",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for image extraction (slower but higher quality)",
    )

    args = parser.parse_args()

    print("🚀 Image Description Batch Preparer")
    print("=" * 60)
    print(f"📁 Input folder: {args.input_folder}")
    print(f"📁 Output folder: {args.output_folder}")
    print(f"📦 Batch size: {args.batch_size} images")
    if args.images_per_pdf:
        print(f"🖼️  Max images per PDF: {args.images_per_pdf}")
    print(f"🔧 Mode: {args.mode}")
    print(f"⚡ GPU: {'enabled' if args.use_gpu else 'disabled'}")
    print("=" * 60)

    success = process_pdfs(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        mode=args.mode,
        batch_size=args.batch_size,
        images_per_pdf=args.images_per_pdf,
        system_instruction=args.system_instruction,
        use_gpu=args.use_gpu,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
