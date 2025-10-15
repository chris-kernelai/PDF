#!/usr/bin/env python3
"""
gemini_batch_monitor.py

Reads batch_jobs_tracking.json (created by gemini_batch_uploader.py)
and checks the status of each Gemini batch job.

Usage:
    python gemini_batch_monitor.py developer
    python gemini_batch_monitor.py vertex
"""

import argparse
import json
import os
import sys

from google import genai
from dotenv import load_dotenv

# Load environment
load_dotenv()

TRACKING_FILE = ".generated/image_description_batches/batch_jobs_tracking.json"


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


def main():
    parser = argparse.ArgumentParser(description="Monitor Gemini batch job status")
    parser.add_argument(
        "mode",
        choices=["developer", "vertex"],
        help="API mode: 'developer' for Gemini Developer API or 'vertex' for Vertex AI",
    )
    args = parser.parse_args()

    if not os.path.exists(TRACKING_FILE):
        print(f"❌ Tracking file not found: {TRACKING_FILE}")
        print("   Make sure you've run gemini_batch_uploader.py first")
        return 1

    with open(TRACKING_FILE, "r") as f:
        jobs = json.load(f)

    if not jobs:
        print("❌ No jobs recorded in tracking file")
        return 1

    try:
        client = init_client(args.mode)
    except RuntimeError as e:
        print(e)
        return 1

    print("📊 Gemini Batch Job Status")
    print("=" * 60)

    all_complete = True
    any_failed = False

    for i, entry in enumerate(jobs, 1):
        job_name = entry["job_name"]
        batch_file = entry.get("batch_file", "unknown")

        try:
            job = client.batches.get(name=job_name)

            print(f"\n📦 Job {i}/{len(jobs)}")
            print(f"   File: {os.path.basename(batch_file)}")
            print(f"   Job ID: {job_name}")
            print(f"   Created: {entry['timestamp']}")
            print(f"   Status: {job.state}")

            # Convert state to string for comparison
            state_str = str(job.state)

            # Check if job is complete
            if not any(s in state_str.upper() for s in ["COMPLETED", "SUCCEEDED"]):
                all_complete = False

            # Check for failures
            if any(s in state_str.upper() for s in ["FAILED", "CANCELLED"]):
                any_failed = True
                print(f"   ❌ Job failed or cancelled")

            # Show error if present
            if hasattr(job, "error") and job.error:
                print(f"   Error: {job.error}")
                any_failed = True

            # Show output file location for Developer mode
            if args.mode == "developer":
                if hasattr(job, "output_file") and job.output_file:
                    print(f"   Output file: {job.output_file.name}")

            # Show output location for Vertex mode
            if args.mode == "vertex":
                if hasattr(job, "dest") and hasattr(job.dest, "responses_file"):
                    print(f"   Output: {job.dest.responses_file}")

        except Exception as e:
            print(f"\n❌ Failed to fetch job {i}/{len(jobs)}: {job_name}")
            print(f"   Error: {e}")
            any_failed = True

    # Summary
    print("\n" + "=" * 60)
    if all_complete and not any_failed:
        print("✅ All batch jobs completed successfully!")
        print("\nNext step:")
        print(f"   python image_description_batch_downloader.py {args.mode}")
    elif any_failed:
        print("❌ Some batch jobs failed or encountered errors")
    else:
        print("⏳ Batch jobs still processing...")
        print("   Run this script again to check status")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
