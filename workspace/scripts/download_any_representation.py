#!/usr/bin/env python3
"""
Download any available representation from Supabase + S3.

- Queries librarian.document_locations_v2 table in Supabase
- If --doc-id is provided: downloads all representations for that doc
- If no --doc-id: picks a random doc from Supabase and downloads one random representation
- Optional --rep-type to filter (e.g., DOCLING, DOCLING_IMG)
- Saves under scripts/examples/<filename>
"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from dotenv import load_dotenv

load_dotenv()


S3_BUCKET_DEFAULT = "primer-production-librarian-documents"
DEFAULT_OUTPUT_DIR = Path("workspace/scripts/examples")


def get_supabase_config() -> Dict[str, str]:
    """Load Supabase config from environment variables."""
    try:
        return {
            "host": os.environ["DB_HOST"],
            "port": int(os.environ.get("DB_PORT", "5432")),
            "database": os.environ.get("DB_NAME", "postgres"),
            "user": os.environ["DB_USER"],
            "password": os.environ["DB_PASSWORD"],
        }
    except KeyError as e:
        missing = e.args[0]
        raise RuntimeError(
            f"Missing required Supabase environment variable: {missing}\n"
            "Required: DB_HOST, DB_USER, DB_PASSWORD\n"
            "Optional: DB_NAME (default: postgres), DB_PORT (default: 5432)"
        ) from e


async def load_rows_from_supabase(
    doc_id: Optional[int] = None,
    rep_type: Optional[str] = None,
) -> List[Dict]:
    """Query document_locations_v2 from Supabase."""
    config = get_supabase_config()
    conn = await asyncpg.connect(**config)
    
    try:
        # Build query based on filters
        if doc_id is not None:
            if rep_type:
                query = """
                    SELECT kdocument_id, representation_type::text, s3_key, s3_bucket
                    FROM librarian.document_locations_v2
                    WHERE kdocument_id = $1 AND representation_type::text = $2
                """
                rows = await conn.fetch(query, doc_id, rep_type)
            else:
                query = """
                    SELECT kdocument_id, representation_type::text, s3_key, s3_bucket
                    FROM librarian.document_locations_v2
                    WHERE kdocument_id = $1
                """
                rows = await conn.fetch(query, doc_id)
        else:
            # Get random document
            if rep_type:
                query = """
                    SELECT kdocument_id, representation_type::text, s3_key, s3_bucket
                    FROM librarian.document_locations_v2
                    WHERE representation_type::text = $1
                """
                rows = await conn.fetch(query, rep_type)
            else:
                query = """
                    SELECT kdocument_id, representation_type::text, s3_key, s3_bucket
                    FROM librarian.document_locations_v2
                """
                rows = await conn.fetch(query)
        
        # Convert to list of dicts
        result = []
        for row in rows:
            result.append({
                "kdocument_id": row["kdocument_id"],
                "representation_type": row["representation_type"],
                "s3_key": row["s3_key"],
                "s3_bucket": row["s3_bucket"],
            })
        
        return result
    finally:
        await conn.close()


def get_session(profile: Optional[str]):
    # Prefer explicit profile, then AWS_PROFILE env, then default resolver
    chosen_profile = profile or os.environ.get("AWS_PROFILE")
    if chosen_profile:
        return boto3.Session(profile_name=chosen_profile)
    return boto3.Session()


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


async def async_main():
    parser = argparse.ArgumentParser(description="Download any available representation from Supabase + S3")
    parser.add_argument("--doc-id", type=int, help="Specific document ID to fetch (downloads all its reps)")
    parser.add_argument("--rep-type", help="Filter by representation_type (e.g., DOCLING, DOCLING_IMG)")
    parser.add_argument("--profile", help="AWS profile to use (defaults to $AWS_PROFILE or 'production')")
    parser.add_argument("--bucket", default=S3_BUCKET_DEFAULT, help="Fallback S3 bucket (default: primer-production-librarian-documents)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory (default: scripts/examples)")
    args = parser.parse_args()

    # Load rows from Supabase
    print("📊 Querying Supabase for document locations...")
    try:
        rows = await load_rows_from_supabase(doc_id=args.doc_id, rep_type=args.rep_type)
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to query Supabase: {e}")
        return 1

    if not rows:
        print("❌ No matching documents found in Supabase")
        if args.doc_id:
            print(f"   (doc_id={args.doc_id}, rep_type={args.rep_type or 'any'})")
        return 1

    # Filter/select rows
    if args.doc_id is not None:
        candidate_rows = rows
        mode = f"doc_id={args.doc_id}"
    else:
        # Pick a random doc_id from the results
        doc_ids = list({r["kdocument_id"] for r in rows})
        chosen_doc_id = random.choice(doc_ids)
        candidate_rows = [r for r in rows if r["kdocument_id"] == chosen_doc_id]
        mode = f"random doc_id={chosen_doc_id}"

    # Representation filtering: if unspecified, default to DOCLING and DOCLING_IMG
    if args.rep_type:
        candidate_rows = [r for r in candidate_rows if r.get("representation_type") == args.rep_type]
    else:
        candidate_rows = [r for r in candidate_rows if r.get("representation_type") in ("DOCLING", "DOCLING_IMG")]

    if not candidate_rows:
        print(f"❌ No matching rows after filtering ({mode}{', rep_type='+args.rep_type if args.rep_type else ''})")
        return 1

    print(f"✅ Found {len(candidate_rows)} representation(s) to download ({mode})")

    # Download all matching representations
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
        bucket = r.get("s3_bucket") or args.bucket
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


def main():
    """Synchronous entry point that runs the async main function."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())


