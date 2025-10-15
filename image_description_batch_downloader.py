#!/usr/bin/env python3
"""
image_description_batch_downloader.py

Step 3: Downloads image description batch results from GCS
- Downloads batch results from GCS
- Processes JSONL results to extract image descriptions
- Saves results locally for inspection before integration

Usage:
    python image_description_batch_downloader.py
    python image_description_batch_downloader.py [batch_job_name] [gcs_output_dir]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from google.cloud import storage


class ImageDescriptionDownloader:
    """Downloads Gemini batch results for image descriptions from GCS"""

    def __init__(self):
        self.gcs_client = storage.Client()

    async def download_batch_results(
        self,
        batch_job_name: str,
        gcs_output_dir: str = None,
        batch_file: str = None,
        batch_prefix: str = "image_description_batches",
    ) -> Dict[str, any]:
        """Download batch results from GCS and save locally for inspection"""
        # Check if results already exist locally
        job_id = batch_job_name.split("/")[-1]  # Extract job ID
        output_dir = f".generated/{batch_prefix}_outputs"
        output_file = f"{output_dir}/image_descriptions_{job_id}.json"
        summary_file = f"{output_dir}/image_descriptions_summary_{job_id}.txt"

        if os.path.exists(output_file) and os.path.exists(summary_file):
            print(f"⏭️  Results already exist locally for job {job_id}")
            print(f"📁 JSON file: {output_file}")

            # Load existing results to return count
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                return {
                    "success": True,
                    "processed_count": len(existing_results),
                    "output_file": output_file,
                    "results": existing_results,
                    "skipped": True,
                }
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing results: {e}")
                print("🔄 Proceeding with fresh download...")

        # If no GCS output dir provided, construct it from environment variables
        if not gcs_output_dir:
            bucket_name = os.environ.get("GCS_BUCKET")
            gcs_output_prefix = os.environ.get("GCS_OUTPUT_PREFIX", "gemini_batches/output")
            if not bucket_name:
                return {"success": False, "error": "GCS_BUCKET environment variable required"}

            # If we have a batch file, construct job-specific path
            if batch_file:
                batch_stem = Path(batch_file).stem
                gcs_output_dir = f"gs://{bucket_name}/{gcs_output_prefix}/{batch_stem}/"
                print(f"🔧 Constructed job-specific GCS path: {gcs_output_dir}")
            else:
                gcs_output_dir = f"gs://{bucket_name}/{gcs_output_prefix}/"
                print(f"🔧 Constructed generic GCS output directory: {gcs_output_dir}")

        print(f"🚀 Downloading image description results from: {gcs_output_dir}")
        print(f"📊 Batch job: {batch_job_name}")

        # Download results from GCS
        results_data = await self._download_from_gcs(gcs_output_dir)
        if not results_data:
            return {"success": False, "error": "Failed to download results from GCS"}

        # Process results
        processed_results = await self._process_results(results_data)
        if not processed_results:
            return {"success": False, "error": "Failed to process results"}

        # Save results locally for inspection
        output_file = await self._save_results_locally(processed_results, batch_job_name, batch_prefix)

        return {
            "success": True,
            "processed_count": len(processed_results),
            "output_file": output_file,
            "results": processed_results,
        }

    async def _download_from_gcs(self, gcs_output_dir: str) -> Optional[List[Dict]]:
        """Download results from GCS"""
        try:
            print(f"📥 Downloading results from GCS: {gcs_output_dir}")

            # Parse GCS URI
            if gcs_output_dir.startswith("gs://"):
                gcs_output_dir = gcs_output_dir[5:]  # Remove gs:// prefix

            bucket_name, prefix = gcs_output_dir.split("/", 1)
            bucket = self.gcs_client.bucket(bucket_name)

            # List all files in the output directory
            blobs = bucket.list_blobs(prefix=prefix)
            results_files = [blob for blob in blobs if blob.name.endswith(".jsonl")]

            if not results_files:
                print("❌ No JSONL files found in GCS output directory")
                return None

            print(f"📄 Found {len(results_files)} result files")

            # Download and parse all result files
            all_results = []
            for blob in results_files:
                print(f"📥 Downloading: {blob.name}")
                content = blob.download_as_text()

                # Parse JSONL content
                for line in content.strip().split("\n"):
                    if line.strip():
                        try:
                            result = json.loads(line)
                            all_results.append(result)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Failed to parse JSON line: {e}")
                            continue

            print(f"📊 Downloaded {len(all_results)} result entries")
            return all_results

        except Exception as e:
            print(f"❌ Error downloading from GCS: {e}")
            import traceback

            print(f"❌ Full traceback: {traceback.format_exc()}")
            return None

    async def _process_results(self, results_data: List[Dict]) -> List[Dict]:
        """Process batch results to extract image descriptions"""
        print(f"🔄 Processing {len(results_data)} result entries")

        processed_results = []

        for i, result in enumerate(results_data):
            try:
                # Extract image info from key
                # Key format: {doc_id}_page_{page}_img_{index}
                key = result.get("key", result.get("custom_id", ""))

                if not key:
                    print(f"⚠️  Warning: No key found in result {i}")
                    continue

                # Parse key to extract components
                try:
                    parts = key.split("_")
                    doc_id = parts[0]
                    page_idx = parts.index("page")
                    img_idx = parts.index("img")

                    page_number = int(parts[page_idx + 1])
                    image_index = int(parts[img_idx + 1])
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Warning: Could not parse key format: {key} - {e}")
                    continue

                # Extract description from response
                description = None
                try:
                    response = result.get("response", {})
                    candidates = response.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            description = parts[0].get("text", "")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to extract description for {key}: {e}")
                    description = None

                processed_results.append({
                    "document_id": doc_id,
                    "page_number": page_number,
                    "image_index": image_index,
                    "key": key,
                    "description": description,
                    "raw_result": result
                })

                desc_length = len(description) if description else 0
                print(f"✅ Processed {key} (description length: {desc_length})")

            except Exception as e:
                print(f"❌ Error processing result {i}: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                continue

        print(f"📊 Successfully processed {len(processed_results)} results")
        return processed_results

    async def _save_results_locally(
        self, processed_results: List[Dict], batch_job_name: str, batch_prefix: str = "image_description_batches"
    ) -> str:
        """Save processed results locally for inspection"""
        # Create output directory
        output_dir = f".generated/{batch_prefix}_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Create filename from batch job name
        job_id = batch_job_name.split("/")[-1]  # Extract job ID
        output_file = f"{output_dir}/image_descriptions_{job_id}.json"

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {output_file}")

        # Also create a human-readable summary
        summary_file = f"{output_dir}/image_descriptions_summary_{job_id}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Batch Job: {batch_job_name}\n")
            f.write(f"Processed Images: {len(processed_results)}\n")

            # Group by document
            docs = {}
            for result in processed_results:
                doc_id = result["document_id"]
                if doc_id not in docs:
                    docs[doc_id] = []
                docs[doc_id].append(result)

            f.write(f"Documents: {len(docs)}\n")
            f.write("=" * 80 + "\n\n")

            for doc_id, images in sorted(docs.items()):
                f.write(f"Document ID: {doc_id}\n")
                f.write(f"Images: {len(images)}\n")
                f.write("-" * 40 + "\n")

                for img_result in sorted(images, key=lambda x: (x["page_number"], x["image_index"])):
                    page = img_result["page_number"]
                    idx = img_result["image_index"]
                    desc = img_result["description"]

                    f.write(f"\n  Page {page}, Image {idx}:\n")
                    if desc:
                        # Wrap description at 76 characters
                        desc_lines = [desc[i:i+76] for i in range(0, len(desc), 76)]
                        for line in desc_lines[:5]:  # First 5 lines
                            f.write(f"    {line}\n")
                        if len(desc) > 380:
                            f.write(f"    ... ({len(desc)} characters total)\n")
                    else:
                        f.write("    No description extracted\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"📋 Summary saved to: {summary_file}")

        return output_file


async def download_batch_job_results(
    batch_job_name: str, gcs_output_dir: str = None, batch_prefix: str = "image_description_batches"
) -> bool:
    """Download results for a completed batch job"""
    print("🚀 Image Description Batch Result Downloader")
    print("=" * 80)
    print(f"📁 Using batch prefix: {batch_prefix}")

    # Check required environment variables
    required_vars = ["GCP_PROJECT", "GCP_LOCATION", "GCS_BUCKET"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False

    # Initialize downloader
    try:
        downloader = ImageDescriptionDownloader()
        print("✅ Image description downloader initialized")
    except Exception as e:
        print(f"❌ Error initializing downloader: {e}")
        return False

    # Download results
    try:
        result = await downloader.download_batch_results(batch_job_name, gcs_output_dir, None, batch_prefix)

        if result["success"]:
            if result.get("skipped", False):
                print("\n⏭️  SKIPPED!")
                print(f"📊 Using existing {result['processed_count']} image description results")
                print(f"💾 Results from: {result['output_file']}")
            else:
                print("\n✅ SUCCESS!")
                print(f"📊 Downloaded {result['processed_count']} image description results")
                print(f"💾 Results saved to: {result['output_file']}")

            print("\n📋 You can now inspect the results before integrating with markdown")
            print(f"   - JSON file: {result['output_file']}")
            summary_file = result['output_file'].replace('.json', '_summary.txt')
            print(f"   - Summary file: {summary_file}")
            print(f"   - Output directory: .generated/{batch_prefix}_outputs/")
            return True
        else:
            print("\n❌ FAILED!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error downloading batch results: {e}")
        import traceback

        print(f"❌ Full traceback: {traceback.format_exc()}")
        return False


async def download_all_batch_results(batch_prefix: str = "image_description_batches", gcs_output_dir: str = None) -> bool:
    """Download results for all completed batch jobs from tracking file"""
    print("🚀 Image Description Batch Result Downloader (All Jobs)")
    print("=" * 80)
    print(f"📁 Using batch prefix: {batch_prefix}")

    # Check required environment variables
    required_vars = ["GCP_PROJECT", "GCP_LOCATION", "GCS_BUCKET"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False

    # Load tracking file
    tracking_file = f".generated/{batch_prefix}/batch_jobs_tracking.json"
    if not os.path.exists(tracking_file):
        print(f"❌ Tracking file not found: {tracking_file}")
        print("Run NEW_GEMINI_batch_uploader.py first to create batch jobs")
        return False

    with open(tracking_file, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    if not jobs:
        print("❌ No jobs found in tracking file")
        return False

    print(f"📊 Found {len(jobs)} batch jobs to process")

    # Initialize downloader
    try:
        downloader = ImageDescriptionDownloader()
        print("✅ Image description downloader initialized")
    except Exception as e:
        print(f"❌ Error initializing downloader: {e}")
        return False

    # Process each job
    success_count = 0
    total_images = 0

    for i, job_entry in enumerate(jobs, 1):
        batch_job_name = job_entry["job_name"]
        print(f"\n📦 Processing job {i}/{len(jobs)}: {batch_job_name}")

        try:
            # Get the batch file from the job entry
            batch_file = job_entry.get("batch_file")
            result = await downloader.download_batch_results(batch_job_name, gcs_output_dir, batch_file, batch_prefix)

            if result["success"]:
                if result.get("skipped", False):
                    print(f"⏭️  SKIPPED! Using existing {result['processed_count']} image description results")
                    print(f"💾 Results from: {result['output_file']}")
                else:
                    print(f"✅ SUCCESS! Downloaded {result['processed_count']} image description results")
                    print(f"💾 Results saved to: {result['output_file']}")
                success_count += 1
                total_images += result["processed_count"]
            else:
                print(f"❌ FAILED! Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error processing job {batch_job_name}: {e}")
            import traceback

            print(f"❌ Full traceback: {traceback.format_exc()}")
            continue

    # Final summary
    print("\n📊 FINAL SUMMARY:")
    print(f"✅ Successfully processed: {success_count}/{len(jobs)} jobs")
    print(f"🖼️  Total image descriptions downloaded: {total_images}")
    print(f"📁 Output directory: .generated/{batch_prefix}_outputs/")

    if success_count > 0:
        print("\n📋 You can now inspect the results before integrating with markdown")
        return True
    else:
        print("\n❌ No jobs were successfully processed")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download Gemini image description batch results from GCS")
    parser.add_argument(
        "--batch-prefix",
        default="image_description_batches",
        help="Batch folder prefix (default: image_description_batches)",
    )
    parser.add_argument(
        "batch_job_name",
        nargs="?",
        help="Specific batch job name to process (optional). If not provided, processes all jobs from tracking file.",
    )
    parser.add_argument(
        "gcs_output_dir",
        nargs="?",
        help="GCS output directory (optional). If not provided, constructs from environment variables.",
    )

    args = parser.parse_args()

    if args.batch_job_name:
        # Process specific job
        print(f"🚀 Processing specific batch job: {args.batch_job_name}")
        success = asyncio.run(download_batch_job_results(args.batch_job_name, args.gcs_output_dir, args.batch_prefix))
        return success
    else:
        # No specific job - process all jobs from tracking file
        print("🚀 Processing ALL batch jobs from tracking file...")
        success = asyncio.run(download_all_batch_results(args.batch_prefix, args.gcs_output_dir))
        return success


if __name__ == "__main__":
    import asyncio

    success = main()
    sys.exit(0 if success else 1)
