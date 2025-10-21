#!/usr/bin/env python3
"""
Download DOCLING markdown representations from S3 for specified document IDs.
Saves them to data/processed/ directory.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List

import aioboto3
import asyncpg
from dotenv import load_dotenv

load_dotenv()


class DoclingDownloader:
    """Downloads DOCLING representations from S3."""

    def __init__(self, aws_profile: str = "production"):
        # AWS configuration
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE", "production")
        self.aws_region = os.getenv("AWS_REGION", "eu-west-2")
        self.s3_bucket = os.getenv("S3_BUCKET", "primer-production-librarian-documents")

        # Database configuration
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        # Output directory
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = None
        self.s3_client = None
        self.db_pool = None

    async def initialize(self):
        """Initialize S3 and database connections."""
        # Initialize S3 client
        # If AWS credentials are set explicitly, leave profile unset so boto3 uses them directly
        session_args = {"region_name": self.aws_region}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            # Explicit credentials take precedence, don't set profile
            print(f"Using explicit AWS credentials (not profile)")
        else:
            # Use profile
            session_args["profile_name"] = self.aws_profile
            print(f"Using AWS profile: {self.aws_profile}")
        
        self.session = aioboto3.Session(**session_args)
        self.s3_client = await self.session.client("s3").__aenter__()

        # Initialize database connection pool
        self.db_pool = await asyncpg.create_pool(**self.db_config)
        print("✅ Initialized S3 and database connections")

    async def close(self):
        """Close all connections."""
        if self.s3_client:
            await self.s3_client.__aexit__(None, None, None)
        if self.db_pool:
            await self.db_pool.close()

    async def get_docling_locations(self, document_ids: List[int]) -> dict:
        """
        Get S3 locations for DOCLING representations.

        Args:
            document_ids: List of document IDs

        Returns:
            Dict mapping document_id to S3 key
        """
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT kdocument_id, s3_key, file_format
                FROM librarian.document_locations_v2
                WHERE kdocument_id = ANY($1)
                AND representation_type::text = 'DOCLING'
                ORDER BY kdocument_id
            """
            rows = await conn.fetch(query, document_ids)

            locations = {}
            for row in rows:
                doc_id = row["kdocument_id"]
                s3_key = row["s3_key"]
                locations[doc_id] = s3_key

            return locations

    async def download_markdown(self, document_id: int, s3_key: str) -> bool:
        """
        Download a markdown file from S3.

        Args:
            document_id: Document ID
            s3_key: S3 key for the file

        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_dir / f"doc_{document_id}.md"

        # Skip if already exists
        if output_path.exists():
            print(f"  ⏭️  doc_{document_id}.md already exists, skipping")
            return True

        try:
            # Download from S3
            response = await self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = await response["Body"].read()

            # Write to file
            output_path.write_bytes(content)
            print(f"  ✅ Downloaded doc_{document_id}.md")
            return True

        except Exception as e:
            print(f"  ❌ Failed to download doc_{document_id}.md: {e}")
            return False

    async def download_batch(self, document_ids: List[int]):
        """
        Download DOCLING markdowns for a batch of document IDs.

        Args:
            document_ids: List of document IDs to download
        """
        if not document_ids:
            print("No document IDs provided")
            return

        print(f"\n📥 Downloading DOCLING markdowns for {len(document_ids)} documents...")

        # Get S3 locations from database
        locations = await self.get_docling_locations(document_ids)

        if not locations:
            print("❌ No DOCLING representations found for provided document IDs")
            return

        print(f"Found {len(locations)} DOCLING representations in database")

        # Download each markdown
        success_count = 0
        failed_count = 0

        for doc_id, s3_key in locations.items():
            success = await self.download_markdown(doc_id, s3_key)
            if success:
                success_count += 1
            else:
                failed_count += 1

        # Missing documents
        missing = set(document_ids) - set(locations.keys())
        if missing:
            print(f"\n⚠️  Warning: {len(missing)} documents have no DOCLING representation:")
            for doc_id in sorted(missing)[:10]:  # Show first 10
                print(f"    - doc_{doc_id}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

        print(f"\n📊 Summary:")
        print(f"  ✅ Successfully downloaded: {success_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  ⏭️  Missing DOCLING: {len(missing)}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download DOCLING markdowns from S3 for specified document IDs"
    )
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        type=int,
        help="Document IDs to download (space-separated)",
    )
    parser.add_argument(
        "--from-to-process",
        action="store_true",
        help="Download markdowns for all PDFs in data/to_process/",
    )

    args = parser.parse_args()

    # Determine document IDs
    if args.from_to_process:
        # Get document IDs from PDFs in data/to_process
        to_process_dir = Path("data/to_process")
        pdf_files = list(to_process_dir.glob("doc_*.pdf"))
        document_ids = []
        for pdf in pdf_files:
            try:
                doc_id = int(pdf.stem.replace("doc_", ""))
                document_ids.append(doc_id)
            except ValueError:
                continue
        print(f"Found {len(document_ids)} PDFs in data/to_process/")
    elif args.doc_ids:
        document_ids = args.doc_ids
    else:
        parser.error("Must provide either --doc-ids or --from-to-process")
        return

    if not document_ids:
        print("No document IDs to process")
        return

    # Download markdowns
    downloader = DoclingDownloader()
    try:
        await downloader.initialize()
        await downloader.download_batch(document_ids)
    finally:
        await downloader.close()


if __name__ == "__main__":
    asyncio.run(main())

