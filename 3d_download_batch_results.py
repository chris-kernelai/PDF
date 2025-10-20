#!/usr/bin/env python3
"""
gemini_batch_downloader.py

Step 3: Downloads image description batch results from Gemini
- Supports both Developer API and Vertex AI
- Downloads batch results
- Processes JSONL results to extract image descriptions
- Saves results locally for inspection before integration

Usage:
    python gemini_batch_downloader.py developer
    python gemini_batch_downloader.py vertex
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.cloud import storage

# Import processing logger
from src.processing_logger import ProcessingLogger

# Load environment
load_dotenv()

TRACKING_FILE = ".generated/image_description_batches/batch_jobs_tracking.json"


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


def init_client(mode):
    """Initialize Gemini client in Developer or Vertex mode"""
    if mode == "developer":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("❌ GEMINI_API_KEY not found for Developer mode")
        print("🌐 Using Developer API mode\n")
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    elif mode == "vertex":
        if not os.environ.get("GCP_PROJECT") or not os.environ.get("GCP_LOCATION"):
            raise RuntimeError("❌ GCP_PROJECT and GCP_LOCATION required for Vertex mode")
        print("☁️  Using Vertex AI mode\n")
        return genai.Client(
            vertexai=True,
            project=os.environ["GCP_PROJECT"],
            location=os.environ["GCP_LOCATION"],
        )
    else:
        raise RuntimeError(f"❌ Invalid mode: {mode}. Use 'developer' or 'vertex'")


def download_developer_results(client, job_name: str) -> List[Dict]:
    """Download results from Developer API"""
    print(f"📥 Downloading results for job: {job_name}")

    try:
        # Get the batch job
        job = client.batches.get(name=job_name)

        # Check if output file is available
        if not hasattr(job, 'dest') or not job.dest or not job.dest.file_name:
            print("❌ No output file available yet")
            return []

        output_file_uri = job.dest.file_name
        print(f"📄 Output file: {output_file_uri}")

        # Download the file content directly - download as bytes
        content_bytes = client.files.download(file=output_file_uri)

        # Decode content
        content = content_bytes.decode('utf-8')

        # Parse JSONL
        results = []
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


def process_results(results: List[Dict]) -> List[Dict]:
    """Process batch results to extract image descriptions"""
    print(f"\n🔄 Processing {len(results)} result entries")

    processed_results = []

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
    return processed_results


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
    parser = argparse.ArgumentParser(description="Download Gemini batch results")
    parser.add_argument(
        "mode",
        nargs="?",
        default="vertex",
        choices=["developer", "vertex"],
        help="API mode: 'vertex' (default) for Vertex AI or 'developer' for Gemini Developer API",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID to filter results",
    )
    args = parser.parse_args()

    print("🚀 Gemini Batch Downloader")
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
    validate_environment(args.mode)

    # Initialize client
    try:
        client = init_client(args.mode)
    except RuntimeError as e:
        print(e)
        return 1

    # Initialize GCS client for Vertex mode
    gcs_client = None
    if args.mode == "vertex":
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
            # Download results based on mode
            if args.mode == "developer":
                results = download_developer_results(client, job_name)
            else:
                results = download_vertex_results(client, job_name, gcs_client)

            if not results:
                print(f"❌ No results downloaded for job {job_name}")
                continue

            # Process results
            processed_results = process_results(results)

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
