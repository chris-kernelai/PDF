#!/usr/bin/env python3
"""
gemini_batch_monitor.py

Reads batch_jobs_tracking.json (created by gemini_batch_uploader.py)
and checks the status of each Vertex Gemini batch job.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.pipeline import init_client, validate_environment

# Load environment
load_dotenv()

TRACKING_FILE = ".generated/image_description_batches/batch_jobs_tracking.json"




def main():
    parser = argparse.ArgumentParser(description="Monitor Gemini batch job status (Vertex)")
    args = parser.parse_args()

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
        print(f"🔑 Session ID: {session_id}")
        print(f"📅 Session started: {session_start}")
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

    try:
        client = init_client()
    except RuntimeError as exc:
        print(str(exc))
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

            # Show output location for Vertex mode
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
        print("   python 3d_download_batch_results.py")
    elif any_failed:
        print("❌ Some batch jobs failed or encountered errors")
    else:
        print("⏳ Batch jobs still processing...")
        print("   Run this script again to check status")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
