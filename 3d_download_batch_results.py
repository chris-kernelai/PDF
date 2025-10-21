#!/usr/bin/env python3
"""
gemini_batch_downloader.py

Step 3: Download image description batch results from Vertex Gemini,
extract the responses, and persist them for integration.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from google.cloud import storage

from src.pipeline import init_client, validate_environment

# Import processing logger
from src.processing_logger import ProcessingLogger

# Load environment
load_dotenv()

TRACKING_FILE = ".generated/image_description_batches/batch_jobs_tracking.json"




def download_vertex_results(client, job_name: str, gcs_client) -> List[Dict]:
    """Download results from Vertex AI (GCS)"""
    print(f"📥 Downloading results for job: {job_name}")

    try:
        # Get the batch job
        job = client.batches.get(name=job_name)

        # Check job state first
        print(f"   Job state: {job.state}")

        # For Vertex AI, output is written to GCS
        if not hasattr(job, 'dest'):
            print("❌ No dest attribute in job object")
            return []

        # The GCS URI is in job.dest.gcs_uri (directory, not single file)
        if not hasattr(job.dest, 'gcs_uri') or not job.dest.gcs_uri:
            print("❌ No gcs_uri in dest")
            return []

        base_gcs_uri = job.dest.gcs_uri
        print(f"📄 Base output URI: {base_gcs_uri}")

        # Parse GCS URI (remove gs:// prefix)
        if base_gcs_uri.startswith("gs://"):
            base_gcs_uri = base_gcs_uri[5:]

        # Split bucket and prefix
        parts = base_gcs_uri.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # List all blobs in the output directory
        bucket = gcs_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"❌ No output files found in gs://{bucket_name}/{prefix}")
            return []

        print(f"📋 Found {len(blobs)} file(s) in output directory:")
        for blob in blobs:
            print(f"   - {blob.name}")

        # Download and parse all JSONL files
        results = []
        for blob in blobs:
            if blob.name.endswith('.jsonl') or 'prediction' in blob.name.lower():
                print(f"📥 Downloading {blob.name}...")
                content = blob.download_as_text()

                # Parse JSONL
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            result = json.loads(line)
                            results.append(result)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Failed to parse JSON line: {e}")
                            continue

        print(f"✅ Downloaded {len(results)} results")
        return results

    except Exception as e:
        print(f"❌ Error downloading results: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def update_uuid_tracking(processed_results: List[Dict], output_dir: str):
    """Update UUID tracking file to record latest UUID per document"""
    tracking_file = f"{output_dir}/uuid_tracking.json"

    # Load existing tracking data
    tracking_data = {}
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r") as f:
                tracking_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load UUID tracking file: {e}")

    # Extract UUID and timestamp from results
    for result in processed_results:
        doc_id = result["document_id"]
        batch_uuid = result["batch_uuid"]

        # Update tracking data (latest UUID wins)
        if doc_id not in tracking_data:
            tracking_data[doc_id] = {
                "latest_uuid": batch_uuid,
                "updated_at": datetime.now().isoformat(),
                "uuids": [batch_uuid]
            }
        else:
            # Add UUID if not already tracked
            if batch_uuid not in tracking_data[doc_id]["uuids"]:
                tracking_data[doc_id]["uuids"].append(batch_uuid)
            # Update as latest
            tracking_data[doc_id]["latest_uuid"] = batch_uuid
            tracking_data[doc_id]["updated_at"] = datetime.now().isoformat()

    # Save tracking file
    with open(tracking_file, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(f"📝 Updated UUID tracking: {tracking_file}")


def process_results(results: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Process batch results to extract image descriptions.
    
    Returns:
        Tuple of (processed_results, stats) where stats contains filtering information
    """
    print(f"\n🔄 Processing {len(results)} result entries")

    processed_results = []
    stats = {
        "total_results": len(results),
        "skipped_not_relevant": 0,
        "kept_relevant": 0,
        "parse_errors": 0,
    }

    for i, result in enumerate(results):
        try:
            # Extract image info from key
            # Key format: {batch_uuid}_{doc_id}_page_{page}_img_{index}
            key = result.get("key", result.get("custom_id", ""))

            if not key:
                print(f"⚠️  Warning: No key found in result {i}")
                continue

            # Parse key to extract components
            try:
                parts = key.split("_")
                # First part is UUID, second is doc_id
                batch_uuid = parts[0]
                doc_id = parts[1]
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

            # Filter based on financially_relevant key
            if description:
                try:
                    # Try to parse as JSON to check for financially_relevant key
                    # Handle both single JSON objects and markdown-wrapped JSON
                    desc_text = description.strip()
                    
                    # Remove markdown code fences if present
                    if desc_text.startswith("```json"):
                        desc_text = desc_text[7:]  # Remove ```json
                    elif desc_text.startswith("```"):
                        desc_text = desc_text[3:]  # Remove ```
                    if desc_text.endswith("```"):
                        desc_text = desc_text[:-3]
                    desc_text = desc_text.strip()
                    
                    # Try to parse the JSON
                    desc_json = json.loads(desc_text)
                    
                    # Check if financially_relevant key exists
                    if "financially_relevant" in desc_json:
                        if desc_json["financially_relevant"] is False:
                            # Skip this image entirely
                            stats["skipped_not_relevant"] += 1
                            print(f"  ⏭️  {key}: Skipped (not financially relevant)")
                            continue
                        else:
                            # Remove the financially_relevant key and convert back to JSON string
                            del desc_json["financially_relevant"]
                            description = json.dumps(desc_json, indent=2, ensure_ascii=False)
                            stats["kept_relevant"] += 1
                            
                except (json.JSONDecodeError, ValueError):
                    # If JSON parsing fails, keep the description as-is
                    stats["parse_errors"] += 1

            processed_results.append({
                "batch_uuid": batch_uuid,
                "document_id": doc_id,
                "page_number": page_number,
                "image_index": image_index,
                "key": key,
                "description": description,
            })

            desc_preview = (description[:60] + "...") if description and len(description) > 60 else (description or "None")
            print(f"  ✅ {key}: {desc_preview}")

        except Exception as e:
            print(f"❌ Error processing result {i}: {e}")
            continue

    print(f"\n📊 Successfully processed {len(processed_results)} results")
    if stats["skipped_not_relevant"] > 0:
        print(f"   ⏭️  Filtered out {stats['skipped_not_relevant']} non-financially-relevant images")
    if stats["kept_relevant"] > 0:
        print(f"   ✅ Kept {stats['kept_relevant']} financially relevant images")
    if stats["parse_errors"] > 0:
        print(f"   ⚠️  {stats['parse_errors']} descriptions kept as-is (JSON parse errors)")
    
    return processed_results, stats


def save_results(processed_results: List[Dict], job_name: str, batch_prefix: str = "image_description_batches") -> str:
    """Save processed results locally for inspection"""
    # Create output directory
    output_dir = f".generated/{batch_prefix}_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Update UUID tracking file
    update_uuid_tracking(processed_results, output_dir)

    # Create filename from batch job name
    job_id = job_name.split("/")[-1]  # Extract job ID
    output_file = f"{output_dir}/image_descriptions_{job_id}.json"

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    # Also create a human-readable summary
    summary_file = f"{output_dir}/image_descriptions_summary_{job_id}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Batch Job: {job_name}\n")
        f.write(f"Processed Images: {len(processed_results)}\n")
        f.write("=" * 80 + "\n\n")

        # Group by document
        docs = {}
        for result in processed_results:
            doc_id = result["document_id"]
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(result)

        f.write(f"Documents: {len(docs)}\n\n")

        for doc_id, images in sorted(docs.items()):
            f.write(f"Document: {doc_id}\n")
            f.write(f"Images: {len(images)}\n")
            f.write("-" * 80 + "\n")

            for img_result in sorted(images, key=lambda x: (x["page_number"], x["image_index"])):
                page = img_result["page_number"]
                idx = img_result["image_index"]
                desc = img_result["description"]

                f.write(f"\n  Page {page}, Image {idx}:\n")
                if desc:
                    # Wrap description at 76 characters
                    desc_lines = [desc[i:i+76] for i in range(0, len(desc), 76)]
                    for line in desc_lines[:3]:  # First 3 lines
                        f.write(f"    {line}\n")
                    if len(desc) > 228:
                        f.write(f"    ... ({len(desc)} characters total)\n")
                else:
                    f.write("    No description extracted\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"📋 Summary saved to: {summary_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download Vertex Gemini batch results")
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID to filter results",
    )
    args = parser.parse_args()

    print("🚀 Gemini Batch Downloader (Vertex)")
    print("=" * 60)

    # Load tracking file
    if not os.path.exists(TRACKING_FILE):
        print(f"❌ Tracking file not found: {TRACKING_FILE}")
        print("   Make sure you've run gemini_batch_uploader.py first")
        return 1

    with open(TRACKING_FILE, "r") as f:
        tracking_data = json.load(f)

    # Require new format with session_id
    if isinstance(tracking_data, list):
        print("❌ Old tracking file format detected (no session_id)")
        print("   Please re-run 3b_upload_batches.py to generate new session")
        return 1
    elif isinstance(tracking_data, dict) and "jobs" in tracking_data:
        jobs = tracking_data["jobs"]
        session_id = tracking_data.get("session_id")
        session_start = tracking_data.get("session_start")
        
        # Validate session ID (now required)
        if session_id != args.session_id:
            print(f"❌ Session ID mismatch!")
            print(f"   Expected: {args.session_id}")
            print(f"   Found: {session_id}")
            return 1
        print(f"🔑 Session ID: {session_id} (matched)")
            
        print(f"📅 Session started: {session_start}")
        print(f"📊 Jobs in this session: {len(jobs)}")
    else:
        print("❌ Invalid tracking file format")
        return 1

    if not jobs:
        print("❌ No jobs recorded in tracking file")
        return 1

    # Validate environment before proceeding
    try:
        validate_environment()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    # Initialize client
    try:
        client = init_client()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    # Initialize GCS client for Vertex mode
    gcs_client = storage.Client()

    # Initialize processing logger
    proc_logger = ProcessingLogger()

    # Process each job
    success_count = 0
    total_images = 0

    for i, entry in enumerate(jobs, 1):
        job_name = entry["job_name"]
        batch_file = entry.get("batch_file", "unknown")

        print(f"\n📦 Processing job {i}/{len(jobs)}")
        print(f"   File: {os.path.basename(batch_file)}")
        print(f"   Job ID: {job_name}")

        download_start = datetime.now()

        try:
            # Download results from Vertex
            results = download_vertex_results(client, job_name, gcs_client)

            if not results:
                print(f"❌ No results downloaded for job {job_name}")
                continue

            # Process results
            processed_results, _ = process_results(results)

            if not processed_results:
                print(f"❌ No results processed for job {job_name}")
                continue

            download_duration = (datetime.now() - download_start).total_seconds()

            # Log download per document
            images_by_doc = {}
            batch_uuid_by_doc = {}
            for result in processed_results:
                doc_id = result["document_id"]
                images_by_doc[doc_id] = images_by_doc.get(doc_id, 0) + 1
                if doc_id not in batch_uuid_by_doc:
                    batch_uuid_by_doc[doc_id] = result["batch_uuid"]

            for doc_id, img_count in images_by_doc.items():
                proc_logger.log_download(
                    doc_id=doc_id,
                    batch_uuid=batch_uuid_by_doc[doc_id],
                    images_downloaded=img_count,
                    duration_seconds=download_duration / len(images_by_doc),  # Approximate
                    status="success"
                )

            # Save results
            output_file = save_results(processed_results, job_name)

            success_count += 1
            total_images += len(processed_results)
            print(f"✅ Successfully processed {len(processed_results)} image descriptions")

        except Exception as e:
            print(f"❌ Error processing job {job_name}: {e}")
            import traceback
            print(traceback.format_exc())
            continue

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {success_count}/{len(jobs)} jobs")
    print(f"🖼️  Total image descriptions: {total_images}")
    print(f"📁 Output directory: .generated/image_description_batches_outputs/")

    if success_count > 0:
        print("\nNext step:")
        print("   python image_description_integrator.py")
        return 0
    else:
        print("\n❌ No jobs were successfully processed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
