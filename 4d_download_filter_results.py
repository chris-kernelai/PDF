#!/usr/bin/env python3
"""
4d_download_filter_results.py

Step 4: Download filter batch results and process JSON arrays.
Parses responses where each contains decisions for 20 descriptions.

Usage:
    python 4d_download_filter_results.py
    python 4d_download_filter_results.py developer
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.cloud import storage

# Load environment
load_dotenv()


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
        raise RuntimeError(f"❌ Invalid mode: {mode}")


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

        # The GCS URI is in job.dest.gcs_uri
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


def process_results(results: List[Dict], batch_file: str, request_metadata: Dict) -> List[Dict]:
    """
    Process batch results to extract filter decisions.
    Each result contains decisions for multiple descriptions (20 per request).

    Args:
        results: List of batch results from Gemini
        batch_file: Path to original batch file
        request_metadata: Metadata dict loaded from request_metadata.json
    """
    print(f"\n🔄 Processing {len(results)} result entries")

    all_filter_results = []

    for i, result in enumerate(results):
        try:
            # Debug: print first result to see structure
            if i == 0:
                print(f"\n🔍 DEBUG: First result structure:")
                print(json.dumps(result, indent=2)[:1000])
                print("...\n")

            # Extract key
            key = result.get("key", result.get("custom_id", ""))

            if not key:
                print(f"⚠️  Warning: No key found in result {i}")
                continue

            # Get metadata for this request from separate metadata file
            metadata = request_metadata.get(key, {})
            if not metadata:
                print(f"⚠️  Warning: No metadata found for key {key}")
                continue

            descriptions = metadata.get("descriptions", [])

            # Extract response
            response = result.get("response", {})
            candidates = response.get("candidates", [])
            if not candidates:
                print(f"⚠️  Warning: No candidates in response for {key}")
                continue

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                print(f"⚠️  Warning: No parts in content for {key}")
                continue

            response_text = parts[0].get("text", "")

            # Strip markdown code blocks if present (```json ... ```)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove ```
            response_text = response_text.strip()

            # Parse JSON array from response
            try:
                filter_decisions = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse JSON response for {key}: {e}")
                print(f"   Response text: {response_text[:200]}")
                continue

            if not isinstance(filter_decisions, list):
                print(f"⚠️  Warning: Response is not a list for {key}")
                continue

            # Match decisions back to descriptions
            for decision in filter_decisions:
                idx = decision.get("index")
                if idx is None or idx < 0 or idx >= len(descriptions):
                    print(f"⚠️  Warning: Invalid index {idx} in response for {key}")
                    continue

                desc_metadata = descriptions[idx]

                filter_result = {
                    "key": desc_metadata.get("key", ""),
                    "document_id": desc_metadata.get("document_id"),
                    "page_number": desc_metadata.get("page_number"),
                    "image_index": desc_metadata.get("image_index"),
                    "batch_uuid": desc_metadata.get("batch_uuid"),
                    "description": desc_metadata.get("description", ""),
                    "include": decision.get("include", False),
                    "reason": decision.get("reason", ""),
                }

                all_filter_results.append(filter_result)

                status = "✅ INCLUDE" if filter_result["include"] else "❌ EXCLUDE"
                print(f"  {status}: {filter_result['document_id']} page {filter_result['page_number']} img {filter_result['image_index']}")

        except Exception as e:
            print(f"❌ Error processing result {i}: {e}")
            import traceback
            print(traceback.format_exc())
            continue

    print(f"\n📊 Successfully processed {len(all_filter_results)} filter decisions")
    return all_filter_results


def main():
    parser = argparse.ArgumentParser(description="Download filter batch results")
    parser.add_argument(
        "mode",
        nargs="?",
        default="vertex",
        choices=["developer", "vertex"],
        help="API mode: 'vertex' (default) or 'developer'",
    )
    parser.add_argument(
        "--batch-prefix",
        default="filter_batches",
        help="Batch folder prefix (default: filter_batches)",
    )
    args = parser.parse_args()

    print("🚀 Filter Batch Downloader")
    print("=" * 60)

    # Load tracking file
    tracking_file = f".generated/{args.batch_prefix}/filter_jobs_tracking.json"
    if not os.path.exists(tracking_file):
        print(f"❌ Tracking file not found: {tracking_file}")
        print("   Make sure you've run 4b_upload_filter_batches.py first")
        return 1

    with open(tracking_file, "r") as f:
        jobs = json.load(f)

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

    # Load request metadata
    request_metadata_file = f".generated/{args.batch_prefix}/request_metadata.json"
    if not os.path.exists(request_metadata_file):
        print(f"❌ Request metadata file not found: {request_metadata_file}")
        return 1

    with open(request_metadata_file, "r") as f:
        request_metadata = json.load(f)
    print(f"📋 Loaded metadata for {len(request_metadata)} requests")

    # Create output directory
    output_dir = Path(".generated/image_description_batches_outputs_filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_filter_results = []

    # Process each job
    for i, entry in enumerate(jobs, 1):
        job_name = entry["job_name"]
        batch_file = entry.get("batch_file", "unknown")

        print(f"\n📦 Processing job {i}/{len(jobs)}")
        print(f"   File: {os.path.basename(batch_file)}")
        print(f"   Job ID: {job_name}")

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
            filter_results = process_results(results, batch_file, request_metadata)

            if not filter_results:
                print(f"❌ No results processed for job {job_name}")
                continue

            all_filter_results.extend(filter_results)
            print(f"✅ Successfully processed {len(filter_results)} filter decisions")

        except Exception as e:
            print(f"❌ Error processing job {job_name}: {e}")
            import traceback
            print(traceback.format_exc())
            continue

    if not all_filter_results:
        print("\n❌ No filter results processed")
        return 1

    # Save all filter results
    print(f"\n💾 Saving filter results...")

    # Save all results (both included and excluded)
    all_results_file = output_dir / "filter_results_all.json"
    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(all_filter_results, f, indent=2, ensure_ascii=False)
    print(f"📝 Saved all results to: {all_results_file}")

    # Save only included descriptions
    included = [r for r in all_filter_results if r.get("include", False)]
    filtered_file = output_dir / "filtered_descriptions.json"
    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(included, f, indent=2, ensure_ascii=False)
    print(f"📝 Saved included descriptions to: {filtered_file}")

    # Summary
    total = len(all_filter_results)
    included_count = len(included)
    excluded_count = total - included_count

    print(f"\n{'='*80}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Total descriptions evaluated: {total}")
    print(f"✅ Included (relevant): {included_count} ({included_count*100//total}%)")
    print(f"❌ Excluded (not relevant): {excluded_count} ({excluded_count*100//total}%)")
    print(f"📁 Output directory: {output_dir}")
    print(f"\nNext step:")
    print(f"  python 5_integrate_descriptions.py")
    print(f"{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
