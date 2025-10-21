#!/usr/bin/env python3
"""
gemini_batch_uploader.py

Step 2: Upload batch JSONL files to GCS and create Vertex Gemini batch jobs.
"""

import argparse
import glob
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig

from src.pipeline import init_client, validate_environment

# -------------------------------------------------------------------
# Load environment
# -------------------------------------------------------------------
load_dotenv()

# Session ID will be set from command line argument or generated if not provided
SESSION_ID = None


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


def upload_file_vertex(local_path, bucket_name, gcs_prefix):
    """Upload file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    gcs_path = f"{gcs_prefix}/{Path(local_path).name}"
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"✅ Uploaded to GCS: {uri}")
    return uri


def create_batch_job(client, model, src_uri, gcs_output_uri):
    """Create a Vertex Gemini batch job."""
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
        "--batch-prefix",
        default="image_description_batches",
        help="Batch folder prefix (default: image_description_batches)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID for this upload run",
    )
    args = parser.parse_args()

    # Set session ID from argument (now required)
    global SESSION_ID
    SESSION_ID = args.session_id

    print("🚀 Gemini Batch Uploader (Vertex)")
    print("=" * 40)
    print(f"📁 Using batch prefix: {args.batch_prefix}")
    print(f"🔑 Session ID: {SESSION_ID}")

    # Validate environment before proceeding

    try:
        validate_environment()
    except RuntimeError as exc:
        print(str(exc))
        return False

    # Initialize client
    try:
        client = init_client()
    except RuntimeError as exc:
        print(str(exc))
        return False

    # Check batch folder
    batch_folder = f".generated/{args.batch_prefix}"
    if not os.path.exists(batch_folder):
        print(f"❌ Batch folder not found: {batch_folder}")
        return False

    # Session ID is already set from command line argument

    # Look for image description batch files (with session ID format)
    batch_files = glob.glob(f"{batch_folder}/image_description_batch_*_imgs_*_*.jsonl")

    if not batch_files:
        print(f"❌ No batch files found in {batch_folder}")
        print("   Looking for pattern:")
        print(f"   - {batch_folder}/image_description_batch_*_imgs_*_*.jsonl")
        return False

    print(f"📄 Found {len(batch_files)} batch files")

    # Config for Vertex
    bucket_name = os.environ.get("GCS_BUCKET")
    gcs_input_prefix = os.environ.get("GCS_INPUT_PREFIX", "gemini_batches/input")
    gcs_output_prefix = os.environ.get("GCS_OUTPUT_PREFIX", "gemini_batches/output")

    # Default Vertex batch model
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")

    batch_jobs = []
    processed_files = []

    for i, batch_file in enumerate(batch_files, 1):
        print(f"\n📤 Processing {i}/{len(batch_files)}: {os.path.basename(batch_file)}")

        if not validate_jsonl_file(batch_file):
            print(f"❌ Skipping invalid file: {batch_file}")
            continue

        try:
            if not bucket_name:
                print("❌ Missing GCS_BUCKET for Vertex mode")
                return False
            src_uri = upload_file_vertex(batch_file, bucket_name, gcs_input_prefix)
            gcs_output_uri = f"gs://{bucket_name}/{gcs_output_prefix}/{Path(batch_file).stem}/"
            job_name = create_batch_job(client, model, src_uri, gcs_output_uri)

            batch_jobs.append(job_name)
            processed_files.append(
                {
                    "batch_file": batch_file,
                    "job_name": job_name,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": SESSION_ID,
                }
            )
            print(f"   ✅ Job created: {job_name}")
        except Exception as e:
            import traceback

            print(f"❌ Error creating job for {batch_file}: {e}")
            print(traceback.format_exc())
            continue

    # Save tracking info with session metadata
    if processed_files:
        tracking_file = os.path.join(batch_folder, "batch_jobs_tracking.json")
        tracking_data = {
            "session_id": SESSION_ID,
            "session_start": datetime.now().isoformat(),
            "jobs": processed_files
        }
        with open(tracking_file, "w") as f:
            json.dump(tracking_data, f, indent=2)
        print(f"\n📝 Job tracking saved to: {tracking_file}")
        print(f"   Session ID: {SESSION_ID} (use this to filter downloads)")

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
