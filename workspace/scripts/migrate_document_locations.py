#!/usr/bin/env python3
"""
Migrate data from document_locations_v2 to document_locations with tag extraction.

This script:
1. Reads rows from document_locations_v2 for specified doc_ids
2. Downloads the text files from S3 in batches
3. Extracts YAML frontmatter tags from the beginning of files
4. Inserts rows into document_locations with tags in the tags column
5. Uploads cleaned files (without tags) back to S3
6. Verifies the operation

Usage:
    python migrate_document_locations.py --start-id 27000 --end-id 28000
    python migrate_document_locations.py --doc-ids 27338,27856,29647
    python migrate_document_locations.py --start-id 27000 --end-id 28000 --batch-size 50
"""

import argparse
import asyncio
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import aioboto3
import asyncpg
from dotenv import load_dotenv
import os

# Add workspace to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

load_dotenv()


class DocumentLocationMigrator:
    """Migrates document_locations_v2 rows to document_locations with tag extraction."""

    def __init__(
        self,
        aws_profile: Optional[str] = None,
        aws_region: Optional[str] = None,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        s3_bucket: Optional[str] = None,
    ):
        # AWS configuration
        env_profile = os.getenv("AWS_PROFILE")
        if os.getenv("AWS_ACCESS_KEY_ID"):
            self.aws_profile = aws_profile
        elif aws_profile:
            self.aws_profile = aws_profile
        elif env_profile:
            self.aws_profile = env_profile
        else:
            self.aws_profile = None

        self.aws_region = aws_region or os.getenv("AWS_REGION", "eu-west-2")
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET", "primer-production-librarian-documents")

        # Database configuration
        self.db_config = {
            "host": db_host or os.getenv("DB_HOST"),
            "port": db_port or int(os.getenv("DB_PORT", "5432")),
            "database": db_name or os.getenv("DB_NAME", "postgres"),
            "user": db_user or os.getenv("DB_USER"),
            "password": db_password or os.getenv("DB_PASSWORD"),
        }

        self.session = None
        self.s3_client = None
        self.db_pool = None

        # Statistics
        self.stats = {
            "total_rows": 0,
            "migrated": 0,
            "skipped": 0,
            "failed": 0,
            "tags_extracted": 0,
            "no_tags": 0,
        }

    async def initialize(self):
        """Initialize S3 and database connections."""
        # Initialize S3 client
        session_args = {"region_name": self.aws_region}
        if self.aws_profile:
            session_args["profile_name"] = self.aws_profile
        self.session = aioboto3.Session(**session_args)
        self.s3_client = await self.session.client("s3").__aenter__()

        # Initialize database connection pool
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                print(f"🔄 Connecting to database (attempt {attempt + 1}/{max_retries})...")
                self.db_pool = await asyncpg.create_pool(**self.db_config)
                print("✅ Database connection established")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Database connection failed: {e}")
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"❌ Database connection failed after {max_retries} attempts")
                    raise

    async def close(self):
        """Close all connections."""
        if self.s3_client:
            await self.s3_client.__aexit__(None, None, None)
        if self.db_pool:
            await self.db_pool.close()
        print("✅ Closed all connections")

    def extract_tags(self, content: str, representation_type: str) -> Tuple[str, str, bool]:
        """
        Extract YAML frontmatter tags from the beginning of the document.

        Uses two-stage process with hardcoded expectations:
        1. Identify --- delimiters and verify line numbers match expected
        2. Remove exact lines only if structure matches expectations

        Hardcoded expected structures:
        - DOCLING: 8 lines of tags (lines 0-7)
        - DOCLING_IMG: 15 lines of tags (lines 0-14)

        Args:
            content: File content
            representation_type: 'DOCLING' or 'DOCLING_IMG'

        Returns:
            Tuple of (tags_text, cleaned_content, success)
            success is False if structure doesn't match expectations
        """
        lines = content.split('\n')

        if not lines or lines[0] != '---':
            # No tags at start
            print(f"  ℹ️  No frontmatter found (file doesn't start with ---)")
            return "", content, True

        # Hardcoded expectations per representation type
        EXPECTED_STRUCTURE = {
            'DOCLING': {
                'tag_lines': 8,       # Lines 0-7 (inclusive)
                'num_blocks': 1,
                'delimiter_positions': [0, 7]
            },
            'DOCLING_IMG': {
                'tag_lines': 15,      # Lines 0-14 (inclusive)
                'num_blocks': 2,
                'delimiter_positions': [0, 5, 7, 14]
            }
        }

        expected = EXPECTED_STRUCTURE.get(representation_type)
        if not expected:
            print(f"  ❌ Unknown representation type: {representation_type}")
            return "", content, False

        # Stage 1: Identify frontmatter blocks by finding --- delimiters
        delimiter_lines = []
        for i, line in enumerate(lines):
            if line == '---':
                delimiter_lines.append(i)
                # Stop searching after we have enough delimiters or we're past expected range
                if len(delimiter_lines) >= len(expected['delimiter_positions']) or i > 20:
                    break

        # Verify delimiter positions match expectations
        actual_delimiters = delimiter_lines[:len(expected['delimiter_positions'])]
        if actual_delimiters != expected['delimiter_positions']:
            print(f"  ❌ Delimiter mismatch for {representation_type}:")
            print(f"     Expected delimiters at lines: {expected['delimiter_positions']}")
            print(f"     Found delimiters at lines: {delimiter_lines}")
            return "", content, False

        # Stage 2: Extract tags using hardcoded line count
        tag_line_count = expected['tag_lines']

        # Verify we have enough lines
        if len(lines) <= tag_line_count:
            print(f"  ❌ File too short: {len(lines)} lines, need at least {tag_line_count + 1}")
            return "", content, False

        # Extract tags (lines 0 to tag_line_count-1)
        tags_lines = lines[0:tag_line_count]
        tags_text = '\n'.join(tags_lines)

        print(f"  ✓ Verified {representation_type} structure: {expected['num_blocks']} block(s), {tag_line_count} tag lines")

        # Stage 3: Remove exact tag lines, then strip trailing whitespace
        # Remove lines 0 through tag_line_count-1 (inclusive)
        remaining_lines = lines[tag_line_count:]

        # Count and skip leading blank lines
        blank_lines_skipped = 0
        for i, line in enumerate(remaining_lines):
            if line.strip() == '':
                blank_lines_skipped += 1
            else:
                break

        # Remove the leading blank lines
        remaining_lines = remaining_lines[blank_lines_skipped:]

        if blank_lines_skipped > 0:
            print(f"  ✓ Removed {tag_line_count} tag lines + {blank_lines_skipped} blank lines = {tag_line_count + blank_lines_skipped} total lines")

        # Final content
        cleaned_content = '\n'.join(remaining_lines)

        return tags_text, cleaned_content, True

    async def download_from_s3(self, s3_key: str) -> Optional[str]:
        """Download file content from S3."""
        try:
            response = await self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = await response['Body'].read()
            return content.decode('utf-8')
        except Exception as e:
            print(f"❌ Error downloading {s3_key}: {e}")
            return None

    async def upload_to_s3(self, s3_key: str, content: str) -> bool:
        """Upload file content to S3."""
        try:
            content_bytes = content.encode('utf-8')
            await self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=content_bytes,
                ContentType='text/plain'
            )
            return True
        except Exception as e:
            print(f"❌ Error uploading {s3_key}: {e}")
            return False

    async def fetch_rows_from_v2(self, doc_ids: List[int]) -> List[dict]:
        """Fetch rows from document_locations_v2 for specified doc_ids."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT
                    id,
                    kdocument_id,
                    representation_type::text as representation_type,
                    s3_bucket,
                    s3_key,
                    content_length,
                    checksum,
                    file_format,
                    page_count,
                    token_count,
                    created_at,
                    updated_at
                FROM librarian.document_locations_v2
                WHERE kdocument_id = ANY($1)
                ORDER BY kdocument_id, representation_type
            """

            rows = await conn.fetch(query, doc_ids)
            return [dict(row) for row in rows]

    async def check_existing_in_target(self, kdocument_id: int, representation_type: str) -> bool:
        """Check if a row already exists in document_locations."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT EXISTS(
                    SELECT 1
                    FROM librarian.document_locations
                    WHERE kdocument_id = $1 AND representation_type::text = $2
                )
            """
            exists = await conn.fetchval(query, kdocument_id, representation_type)
            return exists

    async def insert_into_target(self, row: dict, tags: str) -> bool:
        """Insert row into document_locations with tags."""
        async with self.db_pool.acquire() as conn:
            query = """
                INSERT INTO librarian.document_locations
                (kdocument_id, representation_type, s3_bucket, s3_key, content_length,
                 checksum, file_format, page_count, token_count, tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (kdocument_id, representation_type) DO NOTHING
                RETURNING id
            """

            try:
                result = await conn.fetchrow(
                    query,
                    row['kdocument_id'],
                    row['representation_type'],
                    row['s3_bucket'],
                    row['s3_key'],
                    row['content_length'],
                    row['checksum'],
                    row['file_format'],
                    row['page_count'],
                    row['token_count'],
                    tags if tags else None,
                    row['created_at'],
                    datetime.utcnow()  # Update the updated_at timestamp
                )

                if result:
                    return True
                else:
                    print(f"⚠️  Row already exists for doc {row['kdocument_id']}, rep {row['representation_type']}")
                    return False

            except Exception as e:
                print(f"❌ Error inserting row for doc {row['kdocument_id']}: {e}")
                return False

    async def verify_migration(self, kdocument_id: int, representation_type: str, original_s3_key: str) -> bool:
        """Verify that migration was successful."""
        try:
            # Check database entry exists
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, tags
                    FROM librarian.document_locations
                    WHERE kdocument_id = $1 AND representation_type::text = $2
                """
                row = await conn.fetchrow(query, kdocument_id, representation_type)

                if not row:
                    print(f"❌ Verification failed: No database entry for doc {kdocument_id}, rep {representation_type}")
                    return False

            # Check S3 file exists and doesn't have tags
            content = await self.download_from_s3(original_s3_key)
            if content is None:
                print(f"❌ Verification failed: Cannot download file from S3: {original_s3_key}")
                return False

            tags, _, _ = self.extract_tags(content, representation_type)
            if tags:
                print(f"❌ Verification failed: File still has tags: {original_s3_key}")
                return False

            return True

        except Exception as e:
            print(f"❌ Verification error for doc {kdocument_id}: {e}")
            return False

    async def process_batch(self, rows: List[dict]) -> dict:
        """Process a batch of rows."""
        batch_stats = {
            "migrated": 0,
            "skipped": 0,
            "failed": 0,
            "tags_extracted": 0,
            "no_tags": 0,
        }

        for row in rows:
            kdocument_id = row['kdocument_id']
            representation_type = row['representation_type']
            s3_key = row['s3_key']

            print(f"\n📄 Processing doc {kdocument_id}, representation {representation_type}")

            # Check if already exists in target
            if await self.check_existing_in_target(kdocument_id, representation_type):
                print(f"⏭️  Already exists in document_locations, skipping")
                batch_stats['skipped'] += 1
                continue

            # Download file from S3
            print(f"📥 Downloading from S3: {s3_key}")
            content = await self.download_from_s3(s3_key)

            if content is None:
                print(f"❌ Failed to download file")
                batch_stats['failed'] += 1
                continue

            # Extract tags (pass representation_type for hardcoded verification)
            tags, cleaned_content, success = self.extract_tags(content, representation_type)

            if not success:
                print(f"❌ Tag extraction failed - structure doesn't match expected format")
                batch_stats['failed'] += 1
                continue

            if tags:
                print(f"🏷️  Extracted {len(tags)} characters of tags")
                batch_stats['tags_extracted'] += 1
            else:
                print(f"ℹ️  No tags found in document")
                batch_stats['no_tags'] += 1

            # Insert into target table
            print(f"💾 Inserting into document_locations")
            if not await self.insert_into_target(row, tags):
                print(f"❌ Failed to insert into database")
                batch_stats['failed'] += 1
                continue

            # Upload cleaned content back to S3
            print(f"📤 Uploading cleaned content to S3")
            if not await self.upload_to_s3(s3_key, cleaned_content):
                print(f"❌ Failed to upload cleaned content")
                batch_stats['failed'] += 1
                continue

            # Verify migration
            print(f"🔍 Verifying migration")
            if not await self.verify_migration(kdocument_id, representation_type, s3_key):
                print(f"❌ Verification failed")
                batch_stats['failed'] += 1
                continue

            print(f"✅ Successfully migrated doc {kdocument_id}, rep {representation_type}")
            batch_stats['migrated'] += 1

        return batch_stats

    async def migrate(self, doc_ids: List[int], batch_size: int = 100):
        """Migrate documents in batches."""
        print(f"\n🚀 Starting migration for {len(doc_ids)} document IDs")
        print(f"📦 Batch size: {batch_size}")
        print(f"🪣 S3 Bucket: {self.s3_bucket}")
        print(f"🗄️  Database: {self.db_config['database']}")

        # Fetch all rows from v2
        print(f"\n📊 Fetching rows from document_locations_v2...")
        all_rows = await self.fetch_rows_from_v2(doc_ids)

        if not all_rows:
            print("⚠️  No rows found in document_locations_v2 for specified doc_ids")
            return

        print(f"✅ Found {len(all_rows)} rows to migrate")
        self.stats['total_rows'] = len(all_rows)

        # Process in batches
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_rows) + batch_size - 1) // batch_size

            print(f"\n{'='*60}")
            print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} rows)")
            print(f"{'='*60}")

            batch_stats = await self.process_batch(batch)

            # Update overall stats
            self.stats['migrated'] += batch_stats['migrated']
            self.stats['skipped'] += batch_stats['skipped']
            self.stats['failed'] += batch_stats['failed']
            self.stats['tags_extracted'] += batch_stats['tags_extracted']
            self.stats['no_tags'] += batch_stats['no_tags']

            print(f"\n📊 Batch {batch_num} Summary:")
            print(f"   ✅ Migrated: {batch_stats['migrated']}")
            print(f"   ⏭️  Skipped: {batch_stats['skipped']}")
            print(f"   ❌ Failed: {batch_stats['failed']}")
            print(f"   🏷️  Tags extracted: {batch_stats['tags_extracted']}")
            print(f"   ℹ️  No tags: {batch_stats['no_tags']}")

    def print_summary(self):
        """Print final migration summary."""
        print(f"\n{'='*60}")
        print(f"📊 MIGRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total rows:        {self.stats['total_rows']}")
        print(f"✅ Migrated:       {self.stats['migrated']}")
        print(f"⏭️  Skipped:        {self.stats['skipped']}")
        print(f"❌ Failed:         {self.stats['failed']}")
        print(f"🏷️  Tags extracted: {self.stats['tags_extracted']}")
        print(f"ℹ️  No tags:        {self.stats['no_tags']}")
        print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(
        description='Migrate document_locations_v2 to document_locations with tag extraction'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start-id', type=int, help='Start document ID (inclusive)')
    group.add_argument('--doc-ids', type=str, help='Comma-separated list of document IDs')

    parser.add_argument('--end-id', type=int, help='End document ID (inclusive, requires --start-id)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('--profile', type=str, help='AWS profile to use')

    args = parser.parse_args()

    # Build doc_ids list
    if args.doc_ids:
        doc_ids = [int(x.strip()) for x in args.doc_ids.split(',')]
    elif args.start_id:
        if not args.end_id:
            parser.error('--end-id is required when using --start-id')
        doc_ids = list(range(args.start_id, args.end_id + 1))
    else:
        parser.error('Either --doc-ids or --start-id/--end-id must be specified')

    print(f"🎯 Target document IDs: {len(doc_ids)} documents")
    print(f"   Range: {min(doc_ids)} to {max(doc_ids)}")

    # Confirm before proceeding
    response = input("\n⚠️  This will migrate data and modify S3 files. Continue? (yes/NO): ")
    if response.lower() != 'yes':
        print("❌ Migration cancelled")
        sys.exit(0)

    # Initialize migrator
    migrator = DocumentLocationMigrator(aws_profile=args.profile)

    try:
        await migrator.initialize()
        await migrator.migrate(doc_ids, batch_size=args.batch_size)
        migrator.print_summary()

        if migrator.stats['failed'] > 0:
            print("\n⚠️  Some migrations failed. Please review the logs above.")
            sys.exit(1)
        else:
            print("\n🎉 Migration completed successfully!")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())
