#!/usr/bin/env python3
"""
4a_prepare_filter_batches.py

Step 1: Prepare filter batch files for Gemini Batch API.
Groups descriptions into batches of 20 per API request for efficient filtering.

Usage:
    python 4a_prepare_filter_batches.py
    python 4a_prepare_filter_batches.py --batch-size 1000 --descriptions-per-request 20
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Filter prompt for batched evaluations (20 descriptions per request)
BATCH_FILTER_PROMPT = """You are analyzing image descriptions from financial documents (annual reports, earnings presentations, etc.).

Your task: Evaluate EACH of the {count} descriptions below and determine if it contains meaningful FINANCIAL information that would be valuable to include in a financial analysis document.

INCLUDE if the image contains:
- Financial data, numbers, metrics (revenue, profit, growth rates, margins, etc.)
- Charts/graphs showing financial trends or performance
- Tables with financial figures or operational metrics
- Geographic/segment breakdowns with financial data
- Key performance indicators (KPIs) with numerical values
- Market share data, competitive positioning with figures
- Specific quantitative information about business performance

EXCLUDE if the image is:
- Generic corporate logos without context
- Decorative elements or design flourishes
- Stock photography or lifestyle imagery
- Generic brand messaging without data
- Abstract graphics without financial content
- Repetitive header/footer elements
- Pure marketing/branding content without substance

Be strict - when in doubt, EXCLUDE. Only include if there's clear financial value.

Here are the {count} descriptions to evaluate:

{descriptions}

Respond with ONLY a JSON array in this exact format:
[
  {{"index": 0, "include": true, "reason": "one sentence why"}},
  {{"index": 1, "include": false, "reason": "one sentence why"}},
  ...
]

The "index" field must match the index number from the descriptions above (0 to {max_index}).
Each entry must have: index (integer), include (boolean), reason (string).
"""


def load_descriptions(input_dir: Path) -> List[Dict]:
    """Load all image descriptions from JSON files."""
    json_files = [f for f in input_dir.glob("image_descriptions_*.json")
                  if "_with_filters" not in f.name and "uuid_tracking" not in f.name]

    if not json_files:
        logger.error(f"No description files found in {input_dir}")
        return []

    all_descriptions = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                descriptions = json.load(f)
            all_descriptions.extend(descriptions)
            logger.info(f"Loaded {len(descriptions)} from {json_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {json_file.name}: {e}")
            continue

    return all_descriptions


def prepare_filter_batches(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 1000,
    descriptions_per_request: int = 20,
    session_id: str = None
) -> bool:
    """
    Prepare JSONL batch files for filtering.
    Each request in the batch evaluates multiple descriptions (default: 20).

    Args:
        input_dir: Directory containing description files
        output_dir: Directory to save batch JSONL files
        batch_size: Number of requests per batch file (default: 1000)
        descriptions_per_request: Number of descriptions per API request (default: 20)

    Returns:
        True if successful
    """
    logger.info("🚀 Filter Batch Preparer")
    logger.info("=" * 80)
    logger.info(f"📁 Input directory: {input_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📦 Batch size: {batch_size} requests per file")
    logger.info(f"🔢 Descriptions per request: {descriptions_per_request}")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all descriptions
    all_descriptions = load_descriptions(input_dir)
    if not all_descriptions:
        logger.error("No descriptions to filter")
        return False

    logger.info(f"\n📊 Total descriptions to filter: {len(all_descriptions)}")

    # Use provided session ID for this run
    batch_run_uuid = session_id if session_id else str(uuid.uuid4())[:8]

    logger.info(f"📋 Session ID: {batch_run_uuid}")

    # Group descriptions into chunks (20 per request)
    requests = []
    metadata_by_key = {}  # Store metadata separately
    for i in range(0, len(all_descriptions), descriptions_per_request):
        chunk = all_descriptions[i:i + descriptions_per_request]

        # Build descriptions text for prompt
        desc_lines = []
        for idx, desc in enumerate(chunk):
            desc_text = desc.get("description") or ""
            desc_text = desc_text.strip() if desc_text else "No description"
            desc_lines.append(f"[{idx}] {desc_text}")

        descriptions_text = "\n\n".join(desc_lines)

        # Create prompt
        prompt = BATCH_FILTER_PROMPT.format(
            count=len(chunk),
            max_index=len(chunk) - 1,
            descriptions=descriptions_text
        )

        # Create unique key for this request
        request_key = f"filter_req_{i // descriptions_per_request:05d}_{batch_run_uuid}"

        # Build batch request (Gemini Batch API format)
        # Note: For Vertex AI, use simple format without system instruction
        batch_request = {
            "key": request_key,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0
                }
            }
        }

        # Store metadata separately (not in JSONL - Vertex AI doesn't support custom fields)
        # We'll save this in a separate metadata file
        metadata = {
            "filter_batch_uuid": batch_run_uuid,
            "chunk_start_index": i,
            "chunk_size": len(chunk),
            "descriptions": [
                {
                    "index": idx,
                    "key": desc.get("key", ""),
                    "document_id": desc.get("document_id"),
                    "page_number": desc.get("page_number"),
                    "image_index": desc.get("image_index"),
                    "batch_uuid": desc.get("batch_uuid"),
                    "description": desc.get("description", "")
                }
                for idx, desc in enumerate(chunk)
            ]
        }

        requests.append(batch_request)
        metadata_by_key[request_key] = metadata

    logger.info(f"\n✅ Created {len(requests)} API requests ({descriptions_per_request} descriptions each)")
    logger.info(f"📊 Total API calls needed: {len(requests)} (vs {len(all_descriptions)} with 1-per-call approach)")
    logger.info(f"💰 Cost reduction: {len(all_descriptions) // len(requests)}x fewer API calls")

    # Split requests into batch files
    batch_files_created = []
    batch_num = 0

    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        batch_num += 1

        batch_file = output_dir / f"filter_batch_{batch_num:03d}_reqs_{len(batch):04d}_{batch_run_uuid}.jsonl"

        logger.info(f"\n📝 Creating batch file {batch_num}: {len(batch)} requests")

        with open(batch_file, "w", encoding="utf-8") as f:
            for request in batch:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        batch_files_created.append(batch_file)
        logger.info(f"  ✅ Created {batch_file.name}")

    # Save metadata (both batch info and request-level metadata)
    batch_info = {
        "created_at": datetime.now().isoformat(),
        "batch_run_uuid": batch_run_uuid,
        "total_descriptions": len(all_descriptions),
        "descriptions_per_request": descriptions_per_request,
        "total_requests": len(requests),
        "requests_per_batch_file": batch_size,
        "total_batch_files": len(batch_files_created),
        "batch_files": [str(b.name) for b in batch_files_created],
    }

    metadata_path = output_dir / "batch_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    # Save request-level metadata separately
    request_metadata_path = output_dir / "request_metadata.json"
    with open(request_metadata_path, "w") as f:
        json.dump(metadata_by_key, f, indent=2)

    logger.info(f"\n📝 Metadata saved to: {metadata_path}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ BATCH PREPARATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Total descriptions: {len(all_descriptions)}")
    logger.info(f"🔢 Descriptions per request: {descriptions_per_request}")
    logger.info(f"📞 Total API requests: {len(requests)}")
    logger.info(f"📦 Batch files created: {len(batch_files_created)}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"\nNext step:")
    logger.info(f"  python 4b_upload_filter_batches.py")
    logger.info(f"{'='*80}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare filter batch files with multiple descriptions per request"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(".generated/image_description_batches_outputs"),
        help="Input directory with image descriptions (default: .generated/image_description_batches_outputs)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".generated/filter_batches"),
        help="Output directory for batch files (default: .generated/filter_batches)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of requests per batch file (default: 1000)"
    )
    parser.add_argument(
        "--descriptions-per-request",
        type=int,
        default=20,
        help="Number of descriptions to evaluate per API request (default: 20)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID for this filter batch run",
    )

    args = parser.parse_args()

    success = prepare_filter_batches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        descriptions_per_request=args.descriptions_per_request,
        session_id=args.session_id
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
