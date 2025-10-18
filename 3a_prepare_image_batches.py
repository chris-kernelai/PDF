#!/usr/bin/env python3
"""
image_description_batch_from_extracted.py

Create batch requests for Gemini Flash from pre-extracted images
- Uses images already extracted by batch_docling_converter.py
- Creates JSONL batch files for Gemini API
- Supports both Developer API and Vertex AI formats

Usage:
    python image_description_batch_from_extracted.py developer
    python image_description_batch_from_extracted.py vertex
    python image_description_batch_from_extracted.py developer --batch-size 100
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Load environment
load_dotenv()


# -------------------------------------------------------------------
# Environment Validation
# -------------------------------------------------------------------

def validate_environment(mode: str) -> None:
    """
    Validate that required environment variables are set before proceeding.

    Args:
        mode: 'developer' or 'vertex'

    Raises:
        SystemExit: If validation fails
    """
    errors = []

    if mode == "developer":
        if not os.environ.get("GEMINI_API_KEY"):
            errors.append("❌ GEMINI_API_KEY not set (required for Developer mode)")
    elif mode == "vertex":
        # Check Google Cloud credentials
        if not os.environ.get("GCP_PROJECT"):
            errors.append("❌ GCP_PROJECT not set (required for Vertex AI mode)")
        if not os.environ.get("GCP_LOCATION"):
            errors.append("❌ GCP_LOCATION not set (required for Vertex AI mode)")
        if not os.environ.get("GCS_BUCKET"):
            errors.append("❌ GCS_BUCKET not set (required for Vertex AI mode)")

        # Check if authenticated (either gcloud or service account)
        has_gcloud_auth = False
        has_service_account = False

        # Check for service account key file
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            key_file = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            if os.path.exists(key_file):
                has_service_account = True

        # Check for gcloud authentication
        if not has_service_account:
            try:
                import subprocess
                result = subprocess.run(
                    ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.stdout.strip():
                    has_gcloud_auth = True
            except FileNotFoundError:
                errors.append("❌ gcloud CLI not found. Install Google Cloud SDK first")
            except Exception as e:
                errors.append(f"⚠️  Could not verify gcloud authentication: {e}")

        # Require either authentication method
        if not has_gcloud_auth and not has_service_account:
            errors.append("❌ No authentication found. Either:")
            errors.append("   1. Run 'gcloud auth login', OR")
            errors.append("   2. Set GOOGLE_APPLICATION_CREDENTIALS in .env")

    if errors:
        print("\n" + "="*60)
        print("❌ ENVIRONMENT VALIDATION FAILED")
        print("="*60)
        for error in errors:
            print(error)
        print("\nPlease set the required environment variables and ensure you're authenticated.")
        print("="*60 + "\n")
        sys.exit(1)

    print("✅ Environment validation passed\n")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Image Processing
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
    image_path: Path,
    doc_id: str,
    page_number: int,
    image_index: int,
    mode: str,
    system_instruction: Optional[str] = None
) -> Dict:
    """
    Create a single batch request for image description.

    Args:
        image_path: Path to the image file
        doc_id: Document ID
        page_number: Page number (from filename)
        image_index: Image index (from filename)
        mode: 'developer' or 'vertex'
        system_instruction: Optional system instruction for the model

    Returns:
        Batch request dict in Gemini format
    """
    # Encode image
    image_base64 = encode_image_base64(str(image_path))

    # Default prompt for financial document image description
    user_prompt = """Provide a terse, factual description of this image.

Report exactly what is shown:
- Content type (chart, table, diagram, etc.)
- All visible text, labels, and legends
- All numerical data, values, and units
- Axis labels and scales

Do not interpret or analyze. State only what is directly visible. Be concise."""

    # Build request key (unique identifier)
    request_key = f"{doc_id}_page_{page_number:03d}_img_{image_index:02d}"

    # Build contents
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {"text": user_prompt}
            ]
        }
    ]

    # Build request (same format for both developer and vertex)
    batch_request = {
        "key": request_key,
        "request": {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.0
            }
        }
    }

    # Add system instruction if provided (at request level, not inside contents)
    if system_instruction:
        batch_request["request"]["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    return batch_request


def parse_image_filename(filename: str) -> Optional[tuple]:
    """
    Parse image filename to extract page number and image index.
    Expected format: page_XXX_img_YY.png

    Returns:
        Tuple of (page_number, image_index) or None if parsing fails
    """
    try:
        parts = filename.replace('.png', '').split('_')
        if len(parts) == 4 and parts[0] == 'page' and parts[2] == 'img':
            page_num = int(parts[1])
            img_idx = int(parts[3])
            return (page_num, img_idx)
    except (ValueError, IndexError):
        pass
    return None


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


def process_extracted_images(
    images_folder: Path,
    output_folder: Path,
    mode: str,
    batch_size: int = 100,
    system_instruction: Optional[str] = None,
) -> bool:
    """
    Process pre-extracted images and create batch files.
    Only processes images for documents that have corresponding markdown files in data/processed/.

    Args:
        images_folder: Folder containing extracted images (data/images/)
        output_folder: Output folder for batch files
        mode: 'developer' or 'vertex'
        batch_size: Number of images per batch file
        system_instruction: Optional system instruction

    Returns:
        True if successful
    """
    # Create output directories
    batches_dir = output_folder / "image_description_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Always check against data/processed/ folder to ensure documents are fully converted
    processed_folder = Path("data/processed")

    # Find all document folders
    all_doc_folders = [d for d in images_folder.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not all_doc_folders:
        logger.error(f"No document folders found in {images_folder}")
        return False

    logger.info(f"📁 Found {len(all_doc_folders)} document folders in images folder")

    # Filter to only include folders with corresponding markdown files
    doc_folders = []
    skipped_folders = []

    for doc_folder in all_doc_folders:
        doc_id = doc_folder.name
        # Check for markdown file with doc_ prefix
        expected_md_path = processed_folder / f"doc_{doc_id}.md"

        if expected_md_path.exists():
            doc_folders.append(doc_folder)
        else:
            skipped_folders.append(doc_id)
            logger.debug(f"⏭️  Skipping {doc_id}: no markdown file found at {expected_md_path}")

    if skipped_folders:
        logger.info(f"⏭️  Skipped {len(skipped_folders)} folders without markdown files")

    if not doc_folders:
        logger.warning(f"No document folders have corresponding markdown files in {processed_folder}")
        logger.warning("Make sure documents have been fully processed before preparing image batches")
        return False

    logger.info(f"✅ Processing {len(doc_folders)} document folders with markdown files")

    # Collect all images
    all_image_data = []

    for doc_folder in sorted(doc_folders):
        doc_id = doc_folder.name
        image_files = sorted(doc_folder.glob("*.png"))

        logger.info(f"📄 Processing {doc_id}: {len(image_files)} images")

        for image_file in image_files:
            # Parse filename to get page number and image index
            parsed = parse_image_filename(image_file.name)
            if parsed is None:
                logger.warning(f"  Skipping invalid filename: {image_file.name}")
                continue

            page_num, img_idx = parsed
            all_image_data.append({
                'image_path': image_file,
                'doc_id': doc_id,
                'page_number': page_num,
                'image_index': img_idx,
            })

    if not all_image_data:
        logger.error("No valid images found")
        return False

    logger.info(f"🖼️  Total images to process: {len(all_image_data)}")

    # Create batch requests
    logger.info("📦 Creating batch requests...")
    batch_requests = []

    for image_data in all_image_data:
        try:
            request = create_batch_request(
                image_path=image_data['image_path'],
                doc_id=image_data['doc_id'],
                page_number=image_data['page_number'],
                image_index=image_data['image_index'],
                mode=mode,
                system_instruction=system_instruction
            )
            batch_requests.append(request)
        except Exception as e:
            logger.error(f"Failed to create request for {image_data['image_path'].name}: {e}")
            continue

    if not batch_requests:
        logger.error("No batch requests created")
        return False

    # Split into batch files
    logger.info(f"✂️  Splitting into batches of {batch_size}...")
    batch_files_created = []

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

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "mode": mode,
        "source": "pre_extracted_images",
        "images_folder": str(images_folder),
        "total_documents": len(doc_folders),
        "total_images": len(all_image_data),
        "total_batches": len(batch_files_created),
        "batch_size": batch_size,
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
    print(f"📁 Documents processed: {len(doc_folders)}")
    print(f"🖼️  Images processed: {len(all_image_data)}")
    print(f"📦 Batch files created: {len(batch_files_created)}")
    print(f"📁 Batches directory: {batches_dir}")
    print("\nNext step:")
    print(f"  python gemini_batch_uploader.py")
    print("=" * 60)

    return True


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Create batch requests from pre-extracted images"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="vertex",
        choices=["developer", "vertex"],
        help="API mode: 'vertex' (default) for Vertex AI or 'developer' for Gemini Developer API",
    )
    parser.add_argument(
        "--images-folder",
        type=Path,
        default=Path("data/images"),
        help="Folder containing extracted images (default: data/images)",
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
        "--system-instruction",
        type=str,
        default=None,
        help="Optional system instruction for the model",
    )

    args = parser.parse_args()

    # Validate environment before proceeding
    validate_environment(args.mode)

    if not args.images_folder.exists():
        print(f"❌ Images folder not found: {args.images_folder}")
        return 1

    print("🚀 Image Description Batch Preparer (Pre-extracted)")
    print("=" * 60)
    print(f"📁 Images folder: {args.images_folder}")
    print(f"📁 Output folder: {args.output_folder}")
    print(f"📦 Batch size: {args.batch_size} images")
    print(f"🔧 Mode: {args.mode}")
    print("=" * 60)

    success = process_extracted_images(
        images_folder=args.images_folder,
        output_folder=args.output_folder,
        mode=args.mode,
        batch_size=args.batch_size,
        system_instruction=args.system_instruction,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
