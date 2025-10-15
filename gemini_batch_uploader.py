#!/usr/bin/env python3
"""
gemini_batch_uploader.py

Step 2: Uploads batch files to Gemini and creates batch jobs
- Supports both Developer API (via api_key) and Vertex AI (via GCP project/location)
- Uploads JSONL file (to Files API in Developer mode, or to GCS in Vertex mode)
- Creates batch job
- Returns job ID for monitoring

Usage:
    python gemini_batch_uploader.py developer
    python gemini_batch_uploader.py vertex
    python gemini_batch_uploader.py developer --batch-prefix image_description_batches
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig

# -------------------------------------------------------------------
# Load environment
# -------------------------------------------------------------------
load_dotenv()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def validate_jsonl_file(file_path):
    """Validate that the JSONL file is properly formatted"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "key" not in data or "request" not in data:
                        print(f"❌ Invalid JSONL format at line {line_num}: missing 'key' or 'request'")
                        return False
                    if "contents" not in data["request"]:
                        print(f"❌ Invalid JSONL format at line {line_num}: missing 'contents' in request")
                        return False
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error at line {line_num}: {e}")
                    return False
        return True
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False


def init_client(mode):
    """Initialize Gemini client in Developer or Vertex mode"""
    if mode == "developer":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("❌ GEMINI_API_KEY not found for Developer mode")
        print("🌐 Using Developer API mode")
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    elif mode == "vertex":
        if not os.environ.get("GCP_PROJECT") or not os.environ.get("GCP_LOCATION"):
            raise RuntimeError("❌ GCP_PROJECT and GCP_LOCATION required for Vertex mode")
        print("☁️  Using Vertex AI mode")
        return genai.Client(
            vertexai=True,
            project=os.environ["GCP_PROJECT"],
            location=os.environ["GCP_LOCATION"],
        )
    else:
        raise RuntimeError(f"❌ Invalid mode: {mode}. Use 'developer' or 'vertex'")


def upload_file_developer(client, local_path):
    """Upload file to Gemini Files API (Developer API mode)"""
    uploaded = client.files.upload(
        file=local_path,
        config={"mime_type": "jsonl", "display_name": Path(local_path).name},
    )
    print(f"✅ Uploaded to Gemini Files API: {uploaded.name}")
    return uploaded.name


def upload_file_vertex(local_path, bucket_name, gcs_prefix):
    """Upload file to GCS for Vertex mode"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    gcs_path = f"{gcs_prefix}/{Path(local_path).name}"
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"✅ Uploaded to GCS: {uri}")
    return uri


def create_batch_job(client, mode, model, src_uri, gcs_output_uri=None):
    """Create batch job in Developer or Vertex mode"""
    if mode == "developer":
        job = client.batches.create(model=model, src=src_uri)
    else:  # Vertex
        job = client.batches.create(
            model=model,
            src=src_uri,
            config=CreateBatchJobConfig(dest=gcs_output_uri),
        )
    print(f"✅ Created batch job: {job.name}")
    return job.name


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Upload batch files to Gemini and create batch jobs")
    parser.add_argument(
        "mode",
        choices=["developer", "vertex"],
        help="API mode: 'developer' for Gemini Developer API or 'vertex' for Vertex AI",
    )
    parser.add_argument(
        "--batch-prefix",
        default="image_description_batches",
        help="Batch folder prefix (default: image_description_batches)",
    )
    args = parser.parse_args()

    print("🚀 Gemini Batch Uploader")
    print("=" * 40)
    print(f"📁 Using batch prefix: {args.batch_prefix}")

    # Initialize client
    try:
        client = init_client(args.mode)
        mode = args.mode
    except RuntimeError as e:
        print(e)
        return False

    # Check batch folder
    batch_folder = f".generated/{args.batch_prefix}"
    if not os.path.exists(batch_folder):
        print(f"❌ Batch folder not found: {batch_folder}")
        return False

    # Look for image description batch files
    batch_files = glob.glob(f"{batch_folder}/image_description_batch_*_imgs_*.jsonl")

    if not batch_files:
        print(f"❌ No batch files found in {batch_folder}")
        print("   Looking for pattern:")
        print(f"   - {batch_folder}/image_description_batch_*_imgs_*.jsonl")
        return False

    print(f"📄 Found {len(batch_files)} batch files")

    # Config for Vertex
    bucket_name = os.environ.get("GCS_BUCKET")
    gcs_input_prefix = os.environ.get("GCS_INPUT_PREFIX", "gemini_batches/input")
    gcs_output_prefix = os.environ.get("GCS_OUTPUT_PREFIX", "gemini_batches/output")
    model = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

    # Override model for Vertex mode if needed
    if mode == "vertex" and os.environ.get("GCP_PROJECT") and os.environ.get("GCP_LOCATION"):
        model = os.environ.get(
            "GEMINI_MODEL",
            f"projects/{os.environ['GCP_PROJECT']}/locations/{os.environ['GCP_LOCATION']}/publishers/google/models/gemini-2.0-flash-exp",
        )

    batch_jobs = []
    processed_files = []

    for i, batch_file in enumerate(batch_files, 1):
        print(f"\n📤 Processing {i}/{len(batch_files)}: {os.path.basename(batch_file)}")

        if not validate_jsonl_file(batch_file):
            print(f"❌ Skipping invalid file: {batch_file}")
            continue

        try:
            if mode == "developer":
                src_uri = upload_file_developer(client, batch_file)
                job_name = create_batch_job(client, mode, model, src_uri)
            else:
                if not bucket_name:
                    print("❌ Missing GCS_BUCKET for Vertex mode")
                    return False
                src_uri = upload_file_vertex(batch_file, bucket_name, gcs_input_prefix)
                gcs_output_uri = f"gs://{bucket_name}/{gcs_output_prefix}/{Path(batch_file).stem}/"
                job_name = create_batch_job(client, mode, model, src_uri, gcs_output_uri)

            batch_jobs.append(job_name)
            processed_files.append(
                {
                    "batch_file": batch_file,
                    "job_name": job_name,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print(f"   ✅ Job created: {job_name}")
        except Exception as e:
            import traceback

            print(f"❌ Error creating job for {batch_file}: {e}")
            print(traceback.format_exc())
            continue

    # Save tracking info
    if processed_files:
        tracking_file = os.path.join(batch_folder, "batch_jobs_tracking.json")
        with open(tracking_file, "w") as f:
            json.dump(processed_files, f, indent=2)
        print(f"\n📝 Job tracking saved to: {tracking_file}")

    if batch_jobs:
        print(f"\n✅ SUCCESS! Created {len(batch_jobs)} batch jobs:")
        for i, job in enumerate(batch_jobs, 1):
            print(f"   {i}. {job}")
        return True
    else:
        print("\n❌ FAILED! No jobs created")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
