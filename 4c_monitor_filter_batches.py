#!/usr/bin/env python3
"""
4c_monitor_filter_batches.py

Step 3: Monitor filter batch job status.

Usage:
    python 4c_monitor_filter_batches.py
    python 4c_monitor_filter_batches.py developer
"""

import argparse
import json
import os
import sys

from google import genai
from dotenv import load_dotenv

# Load environment
load_dotenv()


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


def main():
    parser = argparse.ArgumentParser(description="Monitor filter batch job status")
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

    try:
        client = init_client(args.mode)
    except RuntimeError as e:
        print(e)
        return 1

    print("📊 Filter Batch Job Status")
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
        print(f"   python 4d_download_filter_results.py")
    elif any_failed:
        print("❌ Some batch jobs failed or encountered errors")
    else:
        print("⏳ Batch jobs still processing...")
        print("   Run this script again to check status")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
