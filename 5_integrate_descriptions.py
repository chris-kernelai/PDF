#!/usr/bin/env python3
"""
image_description_integrator.py

Step 4: Integrate image descriptions into markdown files
- Reads downloaded image descriptions from batch results
- Inserts descriptions into markdown files at appropriate locations
- Creates enhanced markdown files with image descriptions

Usage:
    python image_description_integrator.py
    python image_description_integrator.py --input-dir processed --output-dir processed_enhanced
    python image_description_integrator.py --batch-prefix image_description_batches
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from collections import defaultdict

# Import processing logger
from src.processing_logger import ProcessingLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ImageDescriptionIntegrator:
    """Integrates image descriptions into markdown files"""

    def __init__(
        self,
        markdown_dir: Path,
        output_dir: Path,
        descriptions_dir: Path,
        image_format: str = "detailed",
        overwrite: bool = False,
    ):
        """
        Initialize integrator.

        Args:
            markdown_dir: Directory containing original markdown files
            output_dir: Directory to save enhanced markdown files
            descriptions_dir: Directory containing downloaded descriptions
            image_format: Format for image descriptions - "detailed", "inline", or "section"
            overwrite: Whether to overwrite existing enhanced files
        """
        self.markdown_dir = Path(markdown_dir)
        self.output_dir = Path(output_dir)
        self.descriptions_dir = Path(descriptions_dir)
        self.image_format = image_format
        self.overwrite = overwrite

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_descriptions_added": 0,
            "image_folders_deleted": 0,
        }

    def load_uuid_tracking(self) -> Dict[str, Dict]:
        """
        Load UUID tracking data to determine latest batch per document.

        Returns:
            Dictionary mapping document_id to tracking info
        """
        tracking_file = self.descriptions_dir / "uuid_tracking.json"

        if not tracking_file.exists():
            logger.warning("No UUID tracking file found - will use all descriptions")
            return {}

        try:
            with open(tracking_file, "r") as f:
                tracking_data = json.load(f)
            logger.info(f"Loaded UUID tracking for {len(tracking_data)} documents")
            return tracking_data
        except Exception as e:
            logger.error(f"Failed to load UUID tracking: {e}")
            return {}

    def load_all_descriptions(self) -> Dict[str, List[Dict]]:
        """
        Load all image descriptions from batch result files, filtering by latest UUID.

        Returns:
            Dictionary mapping document_id to list of image descriptions
        """
        logger.info(f"Loading descriptions from {self.descriptions_dir}")

        # Load UUID tracking first
        uuid_tracking = self.load_uuid_tracking()

        descriptions_by_doc = defaultdict(list)
        all_descriptions = []

        # Find all description JSON files - try multiple patterns
        # Pattern 1: image_descriptions_*.json (original format)
        json_files = [f for f in self.descriptions_dir.glob("image_descriptions_*.json")
                      if "_with_filters" not in f.name and "uuid_tracking" not in f.name]

        # Pattern 2: filtered_descriptions.json (filter results format)
        filtered_file = self.descriptions_dir / "filtered_descriptions.json"
        if filtered_file.exists():
            json_files.append(filtered_file)

        # Pattern 3: filter_results_all.json (all filter results including excluded)
        all_results_file = self.descriptions_dir / "filter_results_all.json"
        if not json_files and all_results_file.exists():
            json_files.append(all_results_file)

        if not json_files:
            logger.warning(f"No description files found in {self.descriptions_dir}")
            return descriptions_by_doc

        logger.info(f"Found {len(json_files)} description files")

        # Load all descriptions
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    descriptions = json.load(f)

                all_descriptions.extend(descriptions)
                logger.debug(f"Loaded {len(descriptions)} descriptions from {json_file.name}")

            except Exception as e:
                logger.error(f"Failed to load {json_file.name}: {e}")
                continue

        # Filter by latest UUID
        filtered_count = 0
        for desc in all_descriptions:
            # Extract document_id from key if document_id is invalid (e.g., "page")
            doc_id = desc.get("document_id", "")
            if not doc_id or doc_id == "page":
                # Parse from key format: "37503_page_022_img_49" -> "37503"
                key = desc.get("key", "")
                if key and "_" in key:
                    doc_id = key.split("_")[0]
                else:
                    logger.warning(f"Could not extract document_id from key: {key}")
                    continue

            batch_uuid = desc.get("batch_uuid", "")

            # If we have UUID tracking, only keep descriptions from latest UUID
            if uuid_tracking and doc_id in uuid_tracking:
                latest_uuid = uuid_tracking[doc_id]["latest_uuid"]
                if batch_uuid != latest_uuid:
                    logger.debug(f"Filtering out {desc['key']} (UUID: {batch_uuid}, latest: {latest_uuid})")
                    filtered_count += 1
                    continue

            descriptions_by_doc[doc_id].append(desc)

        if filtered_count > 0:
            logger.info(f"🔄 Filtered out {filtered_count} descriptions from older batches")

        # Sort descriptions by page and image index
        for doc_id in descriptions_by_doc:
            descriptions_by_doc[doc_id].sort(
                key=lambda x: (x["page_number"], x["image_index"])
            )

        total_descriptions = sum(len(descs) for descs in descriptions_by_doc.values())
        logger.info(
            f"Loaded {total_descriptions} descriptions for {len(descriptions_by_doc)} documents"
        )

        return descriptions_by_doc

    def format_image_description(
        self, description: str, page: int, img_idx: int, format_style: str
    ) -> str:
        """
        Format an image description for insertion into markdown.

        Args:
            description: The image description text
            page: Page number
            img_idx: Image index on the page
            format_style: "detailed", "inline", or "section"

        Returns:
            Formatted markdown text
        """
        if not description or not description.strip():
            return ""

        if format_style == "inline":
            # Single line format
            return f"**Image (Page {page}):** {description}\n\n"

        elif format_style == "section":
            # Section format with header
            return f"""
#### Image {img_idx} (Page {page})

{description}

---

"""

        else:  # detailed (default)
            # Blockquote format with metadata
            return f"""
> **📊 Image Description** (Page {page}, Image {img_idx})
>
> {description.replace(chr(10), chr(10) + '> ')}

"""

    def find_page_markers(self, markdown_content: str) -> List[Tuple[int, int]]:
        """
        Find page markers in markdown content.

        Args:
            markdown_content: The markdown content

        Returns:
            List of tuples (page_number, position_in_content)
        """
        page_markers = []

        # Look for page markers like "<!-- PAGE 1 -->"
        for match in re.finditer(r'<!-- PAGE (\d+) -->', markdown_content):
            page_num = int(match.group(1))
            position = match.start()
            page_markers.append((page_num, position))

        # Also look for "Page N" headers at the beginning of lines
        for match in re.finditer(r'^Page (\d+)\s*$', markdown_content, re.MULTILINE):
            page_num = int(match.group(1))
            position = match.start()
            # Only add if not already found via HTML comment
            if not any(p[0] == page_num for p in page_markers):
                page_markers.append((page_num, position))

        # Sort by page number
        page_markers.sort(key=lambda x: x[0])

        return page_markers

    def insert_descriptions_by_page(
        self,
        markdown_content: str,
        descriptions: List[Dict],
    ) -> str:
        """
        Insert image descriptions into markdown content at appropriate page locations.

        Args:
            markdown_content: Original markdown content
            descriptions: List of image descriptions for this document

        Returns:
            Enhanced markdown content with descriptions
        """
        if not descriptions:
            return markdown_content

        # Find page markers
        page_markers = self.find_page_markers(markdown_content)

        if not page_markers:
            logger.warning("No page markers found, appending descriptions at end")
            return self._append_descriptions_at_end(markdown_content, descriptions)

        # Group descriptions by page
        descriptions_by_page = defaultdict(list)
        for desc in descriptions:
            descriptions_by_page[desc["page_number"]].append(desc)

        # Build new content by inserting descriptions after each page marker
        result = []
        last_pos = 0

        for i, (page_num, marker_pos) in enumerate(page_markers):
            # Add content up to this page marker
            result.append(markdown_content[last_pos:marker_pos])

            # Add the page marker itself
            if i + 1 < len(page_markers):
                next_marker_pos = page_markers[i + 1][1]
            else:
                next_marker_pos = len(markdown_content)

            result.append(markdown_content[marker_pos:next_marker_pos])

            # Add descriptions for this page (before the next page marker)
            if page_num in descriptions_by_page:
                page_descs = descriptions_by_page[page_num]
                logger.debug(f"Inserting {len(page_descs)} descriptions for page {page_num}")

                desc_text = "\n### 📷 Images on This Page\n\n"
                for desc in page_descs:
                    desc_text += self.format_image_description(
                        desc["description"],
                        desc["page_number"],
                        desc["image_index"],
                        self.image_format,
                    )

                result.append(desc_text)
                self.stats["total_descriptions_added"] += len(page_descs)

            last_pos = next_marker_pos

        # Add any remaining content
        if last_pos < len(markdown_content):
            result.append(markdown_content[last_pos:])

        return "".join(result)

    def _append_descriptions_at_end(
        self, markdown_content: str, descriptions: List[Dict]
    ) -> str:
        """Append all descriptions at the end of the document"""
        result = [markdown_content]
        result.append("\n\n---\n\n## 📷 Image Descriptions\n\n")

        for desc in descriptions:
            result.append(
                self.format_image_description(
                    desc["description"],
                    desc["page_number"],
                    desc["image_index"],
                    "section",
                )
            )
            self.stats["total_descriptions_added"] += 1

        return "".join(result)

    def process_document(
        self, markdown_file: Path, descriptions: List[Dict]
    ) -> Optional[Path]:
        """
        Process a single markdown file and add image descriptions.

        Args:
            markdown_file: Path to markdown file
            descriptions: List of descriptions for this document

        Returns:
            Path to enhanced markdown file, or None on failure
        """
        try:
            # Determine output path
            output_file = self.output_dir / markdown_file.name

            # Check if already exists and we're not overwriting
            if output_file.exists() and not self.overwrite:
                logger.info(f"Skipping {markdown_file.name} - output already exists")
                self.stats["skipped_files"] += 1
                return None

            # Read original markdown
            with open(markdown_file, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # Insert descriptions
            if descriptions:
                logger.info(
                    f"Processing {markdown_file.name} with {len(descriptions)} descriptions"
                )
                enhanced_content = self.insert_descriptions_by_page(
                    markdown_content, descriptions
                )
            else:
                logger.info(f"Processing {markdown_file.name} (no descriptions)")
                enhanced_content = markdown_content

            # Add metadata header
            metadata_header = f"""---
**Enhanced with Image Descriptions**
**Original File:** {markdown_file.name}
**Descriptions Added:** {len(descriptions)}
**Enhanced Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

"""
            final_content = metadata_header + enhanced_content

            # Write enhanced markdown
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_content)

            logger.info(f"✅ Saved enhanced file: {output_file.name}")
            self.stats["processed_files"] += 1

            # Delete the image folder if it exists
            # Extract document ID from filename
            stem = markdown_file.stem
            if stem.startswith("doc_"):
                doc_id = stem.replace("doc_", "").split("_")[0]
            else:
                doc_id = stem.split("_")[0]

            image_folder = Path("images") / doc_id
            if image_folder.exists() and image_folder.is_dir():
                try:
                    import shutil
                    shutil.rmtree(image_folder)
                    self.stats["image_folders_deleted"] += 1
                    logger.info(f"🗑️  Deleted image folder: {image_folder}")
                except Exception as e:
                    logger.warning(f"Failed to delete image folder {image_folder}: {e}")

            return output_file

        except Exception as e:
            logger.error(f"Failed to process {markdown_file.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.stats["failed_files"] += 1
            return None

    def process_all(self, descriptions_by_doc: Dict[str, List[Dict]], only_new: bool = False) -> bool:
        """
        Process all markdown files.

        Args:
            descriptions_by_doc: Dictionary of descriptions by document ID
            only_new: If True, only process files that have descriptions but no enhanced version yet

        Returns:
            True if successful
        """
        # Find all markdown files
        markdown_files = list(self.markdown_dir.glob("*.md"))

        if not markdown_files:
            logger.error(f"No markdown files found in {self.markdown_dir}")
            return False

        self.stats["total_files"] = len(markdown_files)
        logger.info(f"Found {len(markdown_files)} markdown files to process")

        # Process each file
        for markdown_file in markdown_files:
            # Extract document ID from filename (assumes doc_12345.md or similar)
            stem = markdown_file.stem
            if stem.startswith("doc_"):
                doc_id = stem.replace("doc_", "").split("_")[0]
            else:
                doc_id = stem.split("_")[0]

            # Get descriptions for this document
            descriptions = descriptions_by_doc.get(doc_id, [])

            # If only_new mode, skip if no descriptions or output already exists
            if only_new:
                output_file = self.output_dir / markdown_file.name
                if not descriptions:
                    logger.debug(f"Skipping {markdown_file.name} - no descriptions available")
                    continue
                if output_file.exists():
                    logger.debug(f"Skipping {markdown_file.name} - output already exists")
                    continue
                logger.info(f"Processing new file with descriptions: {markdown_file.name}")

            # Process the file
            self.process_document(markdown_file, descriptions)

        return True

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 80)
        print("📊 INTEGRATION SUMMARY")
        print("=" * 80)
        print(f"📄 Total markdown files: {self.stats['total_files']}")
        print(f"✅ Processed files: {self.stats['processed_files']}")
        print(f"⏭️  Skipped files: {self.stats['skipped_files']}")
        print(f"❌ Failed files: {self.stats['failed_files']}")
        print(f"📷 Total descriptions added: {self.stats['total_descriptions_added']}")
        print(f"🗑️  Image folders deleted: {self.stats['image_folders_deleted']}")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Integrate image descriptions into markdown files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("processed"),
        help="Directory containing original markdown files (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_images"),
        help="Directory to save enhanced markdown files (default: processed_images)",
    )
    parser.add_argument(
        "--descriptions-dir",
        type=Path,
        default=None,
        help="Directory containing description JSON files (default: .generated/{batch_prefix}_outputs_filtered)",
    )
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="Use unfiltered descriptions instead of filtered ones",
    )
    parser.add_argument(
        "--batch-prefix",
        default="image_description_batches",
        help="Batch prefix to locate description files (default: image_description_batches)",
    )
    parser.add_argument(
        "--format",
        choices=["detailed", "inline", "section"],
        default="detailed",
        help="Format for image descriptions (default: detailed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing enhanced files",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Only process files that have descriptions but no enhanced version yet (for periodic backfill)",
    )

    args = parser.parse_args()

    print("🚀 Image Description Integrator")
    print("=" * 80)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")

    # Determine descriptions directory
    if args.descriptions_dir:
        descriptions_dir = args.descriptions_dir
    else:
        if args.unfiltered:
            descriptions_dir = Path(f".generated/{args.batch_prefix}_outputs")
        else:
            descriptions_dir = Path(f".generated/{args.batch_prefix}_outputs_filtered")

    print(f"📁 Descriptions directory: {descriptions_dir}")
    print(f"🎨 Format: {args.format}")
    print(f"♻️  Overwrite: {args.overwrite}")
    print(f"🔄 Only new: {args.only_new}")
    print("=" * 80)

    # Check directories exist
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    if not descriptions_dir.exists():
        logger.error(f"Descriptions directory not found: {descriptions_dir}")
        logger.error("Run image_description_batch_downloader.py first")
        return 1

    # Initialize integrator
    integrator = ImageDescriptionIntegrator(
        markdown_dir=args.input_dir,
        output_dir=args.output_dir,
        descriptions_dir=descriptions_dir,
        image_format=args.format,
        overwrite=args.overwrite,
    )

    # Initialize processing logger
    proc_logger = ProcessingLogger()

    # Load descriptions
    descriptions_by_doc = integrator.load_all_descriptions()

    if not descriptions_by_doc:
        logger.warning("No descriptions loaded - will process files without descriptions")

    integration_start = datetime.now()

    # Process all files
    success = integrator.process_all(descriptions_by_doc, only_new=args.only_new)

    integration_duration = (datetime.now() - integration_start).total_seconds()

    # Log integration per document
    for doc_id, descriptions in descriptions_by_doc.items():
        if descriptions:
            batch_uuid = descriptions[0].get("batch_uuid", "")
            proc_logger.log_integration(
                doc_id=doc_id,
                batch_uuid=batch_uuid,
                images_integrated=len(descriptions),
                duration_seconds=integration_duration / len(descriptions_by_doc) if descriptions_by_doc else 0,
                status="success"
            )

    # Print summary
    integrator.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
