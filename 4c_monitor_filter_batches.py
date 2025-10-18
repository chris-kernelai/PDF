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
        tracking_data = json.load(f)

    # Require new format with session_id
    if isinstance(tracking_data, list):
        print("❌ Old tracking file format detected (no session_id)")
        print("   Please re-run 4b_upload_filter_batches.py to generate new session")
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
    validate_environment(args.mode)

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
