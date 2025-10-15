#!/usr/bin/env python3
"""
image_description_filter.py

Filters image descriptions using Gemini Flash to determine financial relevance.
Only keeps descriptions with direct financial relevance (data, figures, trends).
Filters out generic corporate imagery and filler content.

Usage:
    python image_description_filter.py
    python image_description_filter.py --input .generated/image_description_batches_outputs --output .generated/image_description_batches_outputs_filtered
"""

import argparse
import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

from dotenv import load_dotenv
from google import genai

# Import processing logger
from processing_logger import ProcessingLogger

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Batch filtering prompt for Gemini
BATCH_FILTER_PROMPT = """You are analyzing multiple image descriptions from financial documents (annual reports, earnings presentations, etc.).

Your task: For each image description, determine if it contains meaningful FINANCIAL information that would be valuable to include in a financial analysis document.

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

Image Descriptions:
{descriptions_json}

Respond with ONLY a JSON array where each element has this format:
{{"index": <index>, "include": true/false, "reason": "one sentence why"}}

Be strict - when in doubt, EXCLUDE. Only include if there's clear financial value.

Return the array in the same order as the input."""


class ImageDescriptionFilter:
    """Filters image descriptions based on financial relevance"""

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        """
        Initialize filter.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model to use for filtering
        """
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Statistics
        self.stats = {
            "total_evaluated": 0,
            "included": 0,
            "excluded": 0,
            "failed": 0,
        }

    def evaluate_batch(self, descriptions: List[Dict]) -> List[Dict]:
        """
        Evaluate a batch of descriptions in a single API call.

        Args:
            descriptions: List of description dicts with 'description' and 'key' fields

        Returns:
            List of dicts with 'include' (bool), 'reason' (str), and 'error' (str if failed)
        """
        if not descriptions:
            return []

        try:
            # Prepare descriptions JSON for prompt
            desc_list = []
            for i, desc in enumerate(descriptions):
                desc_list.append({
                    "index": i,
                    "key": desc.get("key", f"unknown_{i}"),
                    "description": desc.get("description", "")[:500]  # Truncate to 500 chars to save tokens
                })

            descriptions_json = json.dumps(desc_list, indent=2)

            # Create prompt
            prompt = BATCH_FILTER_PROMPT.format(descriptions_json=descriptions_json)

            # Call Gemini
            logger.info(f"Evaluating batch of {len(descriptions)} descriptions...")
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            # Parse response
            response_text = response.text.strip()

            # Try to extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            results = json.loads(response_text)

            # Validate it's an array
            if not isinstance(results, list):
                raise ValueError("Response is not a JSON array")

            # Create results map by index
            results_map = {}
            for result in results:
                if "index" in result:
                    results_map[result["index"]] = result

            # Build output in same order as input
            output = []
            for i in range(len(descriptions)):
                if i in results_map:
                    result = results_map[i]
                    output.append({
                        "include": result.get("include", False),
                        "reason": result.get("reason", "No reason provided"),
                        "error": None
                    })
                else:
                    # Missing from response
                    output.append({
                        "include": False,
                        "reason": "Missing from response",
                        "error": "Not included in model response"
                    })

            return output

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response_text[:500]}")
            # Return all false on parse error
            return [{
                "include": False,
                "reason": "Failed to parse batch response",
                "error": f"JSON parse error: {e}"
            } for _ in descriptions]

        except Exception as e:
            logger.error(f"Failed to evaluate batch: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return all false on error
            return [{
                "include": False,
                "reason": "Batch evaluation failed",
                "error": str(e)
            } for _ in descriptions]

    def filter_descriptions(
        self,
        descriptions: List[Dict],
        batch_size: int = 20
    ) -> List[Dict]:
        """
        Filter a list of image descriptions using batch evaluation.

        Args:
            descriptions: List of description dicts with 'description' and 'key' fields
            batch_size: Number of descriptions to evaluate per API call (default: 20)

        Returns:
            Filtered list of description dicts with 'filter_result' added
        """
        logger.info(f"Filtering {len(descriptions)} descriptions in batches of {batch_size}...")

        filtered_descriptions = []

        # Process in batches to avoid token limits
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(descriptions) + batch_size - 1) // batch_size

            logger.info(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} descriptions)")

            # Evaluate entire batch in one API call
            results = self.evaluate_batch(batch)

            # Add results to descriptions
            for desc, result in zip(batch, results):
                desc["filter_result"] = result

                self.stats["total_evaluated"] += 1

                if result.get("error"):
                    self.stats["failed"] += 1
                    logger.warning(f"  ⚠️  {desc.get('key', 'unknown')}: {result['error']}")
                elif result["include"]:
                    self.stats["included"] += 1
                    filtered_descriptions.append(desc)
                    logger.info(f"  ✅ {desc.get('key', 'unknown')}: INCLUDE - {result['reason']}")
                else:
                    self.stats["excluded"] += 1
                    logger.info(f"  ❌ {desc.get('key', 'unknown')}: EXCLUDE - {result['reason']}")

        logger.info(f"\n📊 Filtering complete:")
        logger.info(f"  Total: {self.stats['total_evaluated']}")
        logger.info(f"  Included: {self.stats['included']}")
        logger.info(f"  Excluded: {self.stats['excluded']}")
        logger.info(f"  Failed: {self.stats['failed']}")

        return filtered_descriptions


def filter_batch_results(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 20
) -> bool:
    """
    Filter all batch result files.

    Args:
        input_dir: Directory containing original description files
        output_dir: Directory to save filtered results
        batch_size: Concurrent evaluations
        delay: Delay between batches

    Returns:
        True if successful
    """
    # Find all description JSON files
    json_files = list(input_dir.glob("image_descriptions_*.json"))

    if not json_files:
        logger.error(f"No description files found in {input_dir}")
        return False

    logger.info(f"Found {len(json_files)} description files")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize filter
    try:
        filter_engine = ImageDescriptionFilter()
    except ValueError as e:
        logger.error(f"Failed to initialize filter: {e}")
        return False

    # Initialize processing logger
    proc_logger = ProcessingLogger()

    total_stats = {
        "files_processed": 0,
        "total_descriptions": 0,
        "included": 0,
        "excluded": 0,
        "failed": 0,
    }

    # Process each file
    for json_file in json_files:
        logger.info(f"\nProcessing {json_file.name}...")
        filter_start = datetime.now()

        try:
            # Load descriptions
            with open(json_file, "r", encoding="utf-8") as f:
                descriptions = json.load(f)

            logger.info(f"  Loaded {len(descriptions)} descriptions")
            total_stats["total_descriptions"] += len(descriptions)

            # Filter descriptions
            filtered = filter_engine.filter_descriptions(
                descriptions,
                batch_size=batch_size
            )

            filter_duration = (datetime.now() - filter_start).total_seconds()

            # Update stats
            total_stats["files_processed"] += 1
            total_stats["included"] += filter_engine.stats["included"]
            total_stats["excluded"] += filter_engine.stats["excluded"]
            total_stats["failed"] += filter_engine.stats["failed"]

            # Log filtering per document
            filter_by_doc = {}
            batch_uuid_by_doc = {}
            for desc in descriptions:
                doc_id = desc.get("document_id", "unknown")
                filter_by_doc.setdefault(doc_id, {"in": 0, "out": 0})
                if doc_id not in batch_uuid_by_doc:
                    batch_uuid_by_doc[doc_id] = desc.get("batch_uuid", "")

                if desc.get("filter_result", {}).get("include", False):
                    filter_by_doc[doc_id]["in"] += 1
                else:
                    filter_by_doc[doc_id]["out"] += 1

            for doc_id, counts in filter_by_doc.items():
                proc_logger.log_filter(
                    doc_id=doc_id,
                    batch_uuid=batch_uuid_by_doc[doc_id],
                    images_filtered_in=counts["in"],
                    images_filtered_out=counts["out"],
                    duration_seconds=filter_duration / len(filter_by_doc),  # Approximate
                    status="success"
                )

            # Save filtered results
            output_file = output_dir / json_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)

            logger.info(f"  ✅ Saved {len(filtered)} filtered descriptions to {output_file.name}")

            # Also save full results with filter metadata
            metadata_file = output_dir / f"{json_file.stem}_with_filters.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to process {json_file.name}: {e}")
            continue

    # Print final summary
    print("\n" + "=" * 60)
    print("📊 FILTERING SUMMARY")
    print("=" * 60)
    print(f"📄 Files processed: {total_stats['files_processed']}")
    print(f"🖼️  Total descriptions: {total_stats['total_descriptions']}")
    print(f"✅ Included: {total_stats['included']}")
    print(f"❌ Excluded: {total_stats['excluded']}")
    print(f"⚠️  Failed: {total_stats['failed']}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Filter image descriptions by financial relevance"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".generated/image_description_batches_outputs"),
        help="Input directory with description files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".generated/image_description_batches_outputs_filtered"),
        help="Output directory for filtered results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of descriptions per API call (default: 20)",
    )

    args = parser.parse_args()

    print("🚀 Image Description Filter")
    print("=" * 60)
    print(f"📁 Input directory: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"📦 Batch size: {args.batch_size} descriptions per call")
    print("=" * 60)

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return 1

    success = filter_batch_results(
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
