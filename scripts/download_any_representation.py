#!/usr/bin/env python3
"""
Download any available representation using the same CSV + S3 method as download_compromised_docling.py.

- Reads document_locations_v2_rows.csv by default (override with --csv-file)
- If --doc-id is provided: downloads all representations for that doc
- If no --doc-id: picks a random doc from the CSV and downloads one random representation
- Optional --rep-type to filter (e.g., DOCLING, DOCLING_IMG)
- Saves under scripts/examples/<doc_id>/<representation_type>/<filename>
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


S3_BUCKET_DEFAULT = "primer-production-librarian-documents"
DEFAULT_OUTPUT_DIR = Path("scripts/examples")
DEFAULT_CSV_PATH = Path("document_locations_v2_rows.csv")


def get_session(profile: Optional[str]):
    # Prefer explicit profile, then AWS_PROFILE env, then default resolver
    chosen_profile = profile or os.environ.get("AWS_PROFILE")
    if chosen_profile:
        return boto3.Session(profile_name=chosen_profile)
    return boto3.Session()


def load_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def download_from_s3(s3_client, bucket: str, key: str, output_path: Path) -> bool:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(output_path))
        return True
    except ClientError as e:
        print(f"❌ S3 error for s3://{bucket}/{key}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading s3://{bucket}/{key}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download any available representation (CSV + S3)")
    parser.add_argument("--doc-id", type=int, help="Specific document ID to fetch (downloads all its reps)")
    parser.add_argument("--rep-type", help="Filter by representation_type (e.g., DOCLING, DOCLING_IMG)")
    parser.add_argument("--profile", help="AWS profile to use (defaults to $AWS_PROFILE or 'production')")
    parser.add_argument("--bucket", default=S3_BUCKET_DEFAULT, help="Fallback S3 bucket (default: primer-production-librarian-documents)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory (default: scripts/examples)")
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV_PATH, help="CSV file (default: document_locations_v2_rows.csv)")
    args = parser.parse_args()

    # Load CSV
    rows = load_rows(args.csv_file)

    # Filter rows by doc id
    if args.doc_id is not None:
        candidate_rows = [r for r in rows if r.get("kdocument_id") and str(r["kdocument_id"]) == str(args.doc_id)]
        mode = f"doc_id={args.doc_id}"
    else:
        doc_ids = list({r["kdocument_id"] for r in rows if r.get("kdocument_id")})
        if not doc_ids:
            print("❌ No document IDs in CSV")
            return 1
        chosen_doc_id = random.choice(doc_ids)
        candidate_rows = [r for r in rows if r.get("kdocument_id") and str(r["kdocument_id"]) == str(chosen_doc_id)]
        mode = f"random doc_id={chosen_doc_id}"

    # Representation filtering: if unspecified, default to DOCLING and DOCLING_IMG
    if args.rep_type:
        candidate_rows = [r for r in candidate_rows if r.get("representation_type") == args.rep_type]
    else:
        candidate_rows = [r for r in candidate_rows if r.get("representation_type") in ("DOCLING", "DOCLING_IMG")]

    if not candidate_rows:
        print(f"❌ No matching rows found in CSV ({mode}{', rep_type='+args.rep_type if args.rep_type else ''})")
        return 1

    # Always download both reps for the chosen doc (if available)
    rows_to_download = candidate_rows

    # Init S3
    try:
        # Default to 'production' if nothing provided
        profile_to_use = args.profile or os.environ.get("AWS_PROFILE") or "production"
        session = get_session(profile_to_use)
        s3_client = session.client("s3")
        print(f"✅ S3 client ready (profile={profile_to_use})")
    except NoCredentialsError:
        print("❌ AWS credentials not found. If using SSO, run: aws sso login --profile production")
        return 1
    except Exception as e:
        print(f"❌ Failed to init S3: {e}")
        return 1

    # Download
    success = 0
    failed = 0
    for r in rows_to_download:
        doc_id_str = str(r.get("kdocument_id"))
        rep_type = (r.get("representation_type") or "unknown").upper()
        s3_key = r.get("s3_key")
        bucket = r.get("bucket") or args.bucket
        if not s3_key:
            print(f"⚠️  Missing s3_key for doc {doc_id_str} rep {rep_type}, skipping")
            failed += 1
            continue

        # Dump all files flat into output dir
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # Build filename: doc_<id>.md for DOCLING, doc_<id>_img.md for DOCLING_IMG
        suffix = "_img" if rep_type == "DOCLING_IMG" else ""
        filename = f"doc_{doc_id_str}{suffix}.md"
        out_path = out_dir / filename
        print(f"📥 {doc_id_str} {rep_type}: s3://{bucket}/{s3_key} -> {out_path}")
        if download_from_s3(s3_client, bucket, s3_key, out_path):
            success += 1
        else:
            failed += 1

    print(f"\n✅ Done. Successful: {success}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


