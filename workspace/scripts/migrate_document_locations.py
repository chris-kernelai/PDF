#!/usr/bin/env python3
"""
Extract and migrate representation_tags in document_locations_v2.

This script:
1. Reads rows from document_locations_v2 for specified doc_ids
2. Downloads the text files from S3 in batches
3. Extracts YAML frontmatter representation_tags from the beginning of files
4. Updates the representation_tags column in document_locations_v2
5. Uploads cleaned files (without representation_tags) back to S3
6. Verifies the operation

Usage:
    # First, login to AWS SSO (if using SSO):
    aws sso login --profile your-profile-name
    
    # Then run the script:
    python migrate_document_locations.py --start-id 27000 --end-id 28000 --profile your-profile-name
    python migrate_document_locations.py --doc-ids 27338,27856,29647 --profile your-profile-name
    python migrate_document_locations.py --start-id 27000 --end-id 28000 --batch-size 50
    python migrate_document_locations.py --start-id 27000 --end-id 28000 --limit 1 --yes  # Test with 1 doc
    python migrate_document_locations.py --start-id 27000 --end-id 28000 --limit 1 --download-original ./original --yes  # Download originals
    python migrate_document_locations.py --start-id 27000 --end-id 28000 --limit 1 --download-cleaned ./cleaned --yes  # Download cleaned
    python migrate_document_locations.py --doc-ids 27290 --force --yes  # Overwrite existing rows
    
    # AWS SSO example with original files:
    python migrate_document_locations.py --doc-ids 27290 --profile production --download-original ./originals --yes
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import aioboto3
import asyncpg
from dotenv import load_dotenv
import os

# Add workspace to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

load_dotenv()


class DocumentLocationMigrator:
    """Extracts representation_tags from S3 files and updates document_locations_v2."""

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
        download_cleaned_dir: Optional[Path] = None,
        download_original_dir: Optional[Path] = None,
        force: bool = False,
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

        # Download cleaned files option
        self.download_cleaned_dir = download_cleaned_dir
        if self.download_cleaned_dir:
            self.download_cleaned_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Will save cleaned files to: {self.download_cleaned_dir}")

        # Download original files option
        self.download_original_dir = download_original_dir
        if self.download_original_dir:
            self.download_original_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Will save original files (with tags) to: {self.download_original_dir}")

        # Force mode
        self.force = force
        if self.force:
            print("⚠️  Force mode enabled - will overwrite existing rows")

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
            "total_doc_ids": 0,
            "filtered_tags_extracted": 0,  # Docs skipped because tags already in v2
            "filtered_complete": 0,  # Docs skipped because already in document_locations
            "total_rows": 0,
            "migrated": 0,
            "skipped": 0,
            "failed": 0,
            "tags_extracted": 0,
            "no_tags": 0,
            "overwritten": 0,
            "copied_to_locations": 0,
            "copy_skipped": 0,
            "copy_failed": 0,
            "copy_overwritten": 0,
            "downloaded_cleaned": 0,
            "downloaded_original": 0,
        }

    async def initialize(self):
        """Initialize S3 and database connections."""
        # Initialize S3 client
        session_args = {"region_name": self.aws_region}
        if self.aws_profile:
            session_args["profile_name"] = self.aws_profile
            print(f"🔑 Using AWS profile: {self.aws_profile}")
        self.session = aioboto3.Session(**session_args)
        self.s3_client = await self.session.client("s3").__aenter__()

        # Test AWS credentials
        try:
            # Try to get the caller identity to verify credentials work
            sts_client = await self.session.client("sts").__aenter__()
            identity = await sts_client.get_caller_identity()
            print(f"✅ AWS credentials valid (Account: {identity['Account']})")
            await sts_client.__aexit__(None, None, None)
        except Exception as e:
            error_msg = str(e)
            if "Unable to locate credentials" in error_msg or "ExpiredToken" in error_msg:
                print(f"❌ AWS credentials error: {e}")
                if self.aws_profile:
                    print(f"💡 Run this first: aws sso login --profile {self.aws_profile}")
                else:
                    print(f"💡 Run this first: aws sso login --profile <your-profile>")
                    print(f"   Then add --profile <your-profile> to the script command")
                raise Exception("AWS credentials not configured. Please login with AWS SSO.")
            else:
                print(f"⚠️  Warning: Could not verify AWS credentials: {e}")

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
        Extract YAML frontmatter representation_tags from the beginning of the document.

        Uses two-stage process with hardcoded expectations:
        1. Identify --- delimiters and verify line numbers match expected
        2. Remove exact lines only if structure matches expectations

        Hardcoded expected structures:
        - DOCLING: 8 lines of representation_tags (lines 0-7)
        - DOCLING_IMG: 15 lines of representation_tags (lines 0-14)

        Args:
            content: File content
            representation_type: 'DOCLING' or 'DOCLING_IMG'

        Returns:
            Tuple of (tags_text, cleaned_content, success)
            success is False if structure doesn't match expectations
        """
        lines = content.split('\n')

        if not lines or lines[0] != '---':
            # No representation_tags at start
            print(f"  ℹ️  No frontmatter found (file doesn't start with ---)")
            return "", content, True

        # Hardcoded expectations per representation type
        EXPECTED_STRUCTURE = {
            'DOCLING': {
                'tag_lines': None,    # Will be determined by delimiter position
                'num_blocks': 1,
                'delimiter_positions': [
                    [0, 7],  # Some files have 8 tag lines (lines 0-7)
                    [0, 8]   # Some files have 9 tag lines (lines 0-8)
                ]
            },
            'DOCLING_IMG': {
                'tag_lines': None,    # Will be determined by delimiter position
                'num_blocks': 2,
                'delimiter_positions': [
                    [0, 5, 7, 14],  # Some files have 15 tag lines
                    [0, 5, 7, 15]   # Some files have 16 tag lines
                ]
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
                if len(delimiter_lines) >= 4 or i > 20:
                    break

        # Verify delimiter positions match one of the acceptable patterns
        matched_pattern = None
        for pattern in expected['delimiter_positions']:
            actual_delimiters = delimiter_lines[:len(pattern)]
            if actual_delimiters == pattern:
                matched_pattern = pattern
                break
        
        if not matched_pattern:
            print(f"  ❌ Delimiter mismatch for {representation_type}:")
            print(f"     Expected delimiters at lines (one of): {expected['delimiter_positions']}")
            print(f"     Found delimiters at lines: {delimiter_lines}")
            return "", content, False

        # Stage 2: Extract representation_tags using the last delimiter position + 1
        tag_line_count = matched_pattern[-1] + 1

        # Verify we have enough lines
        if len(lines) <= tag_line_count:
            print(f"  ❌ File too short: {len(lines)} lines, need at least {tag_line_count + 1}")
            return "", content, False

        # Extract representation_tags (lines 0 to tag_line_count-1)
        tags_lines = lines[0:tag_line_count]
        tags_text = '\n'.join(tags_lines)

        print(f"  ✓ Verified {representation_type} structure: {expected['num_blocks']} block(s), {tag_line_count} tag lines (delimiters at {matched_pattern})")

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
            error_msg = str(e)
            if "Unable to locate credentials" in error_msg or "ExpiredToken" in error_msg:
                print(f"❌ AWS credentials error: {e}")
                if self.aws_profile:
                    print(f"💡 Try running: aws sso login --profile {self.aws_profile}")
                else:
                    print(f"💡 Try running: aws sso login --profile <your-profile>")
                    print(f"   Then add --profile <your-profile> to the script command")
            else:
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

    def save_original_file(self, kdocument_id: int, representation_type: str, content: str) -> bool:
        """Save original content (with tags) to local directory for evaluation."""
        if not self.download_original_dir:
            return True  # Not an error, just not enabled
        
        try:
            # Create subdirectory for this document
            doc_dir = self.download_original_dir / f"doc_{kdocument_id}"
            doc_dir.mkdir(exist_ok=True)
            
            # Save with representation type in filename
            filename = f"{representation_type.lower()}_original.md"
            filepath = doc_dir / filename
            
            filepath.write_text(content, encoding='utf-8')
            print(f"  💾 Saved original file: {filepath}")
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to save original file locally: {e}")
            return False

    def save_cleaned_file(self, kdocument_id: int, representation_type: str, content: str) -> bool:
        """Save cleaned content to local directory for evaluation."""
        if not self.download_cleaned_dir:
            return True  # Not an error, just not enabled
        
        try:
            # Create subdirectory for this document
            doc_dir = self.download_cleaned_dir / f"doc_{kdocument_id}"
            doc_dir.mkdir(exist_ok=True)
            
            # Save with representation type in filename
            filename = f"{representation_type.lower()}.md"
            filepath = doc_dir / filename
            
            filepath.write_text(content, encoding='utf-8')
            print(f"  💾 Saved cleaned file: {filepath}")
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to save cleaned file locally: {e}")
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

    async def check_already_processed(self, kdocument_id: int, representation_type: str) -> Tuple[bool, bool]:
        """
        Check if representation_tags already exist in document_locations_v2.
        
        Returns:
            Tuple[bool, bool]: (already_processed, has_existing_tags)
        """
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT representation_tags
                FROM librarian.document_locations_v2
                WHERE kdocument_id = $1 AND representation_type::text = $2
            """
            result = await conn.fetchval(query, kdocument_id, representation_type)
            has_tags = result is not None and result != ""
            return has_tags, has_tags

    async def update_tags_in_v2(self, kdocument_id: int, representation_type: str, representation_tags: str) -> bool:
        """Update representation_tags column in document_locations_v2."""
        async with self.db_pool.acquire() as conn:
            query = """
                UPDATE librarian.document_locations_v2
                SET representation_tags = $3,
                    updated_at = $4
                WHERE kdocument_id = $1 AND representation_type::text = $2
                RETURNING id
            """

            try:
                result = await conn.fetchrow(
                    query,
                    kdocument_id,
                    representation_type,
                    representation_tags if representation_tags else None,
                    datetime.utcnow()
                )

                if result:
                    return True
                else:
                    print(f"⚠️  No row found for doc {kdocument_id}, rep {representation_type}")
                    return False

            except Exception as e:
                print(f"❌ Error updating row for doc {kdocument_id}: {e}")
                return False

    async def check_exists_in_document_locations(self, kdocument_id: int, representation_type: str) -> bool:
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

    async def copy_row_to_document_locations(self, row: dict) -> Tuple[bool, bool, bool]:
        """
        Copy row from document_locations_v2 to document_locations with computed token_count.
        
        Returns:
            Tuple[bool, bool, bool]: (success, was_inserted, was_overwrite)
            - success: operation completed without error
            - was_inserted: True if new row was inserted (False if already existed)
            - was_overwrite: True if existing row was updated (only with --force)
        """
        async with self.db_pool.acquire() as conn:
            # Compute token_count as content_length / 5
            token_count = None
            if row['content_length'] is not None:
                token_count = int(row['content_length'] / 5)

            if self.force:
                # Use DO UPDATE to overwrite existing rows
                query = """
                    INSERT INTO librarian.document_locations
                    (kdocument_id, representation_type, s3_bucket, s3_key, content_length,
                     checksum, file_format, page_count, token_count, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (kdocument_id, representation_type) DO UPDATE SET
                        s3_bucket = EXCLUDED.s3_bucket,
                        s3_key = EXCLUDED.s3_key,
                        content_length = EXCLUDED.content_length,
                        checksum = EXCLUDED.checksum,
                        file_format = EXCLUDED.file_format,
                        page_count = EXCLUDED.page_count,
                        token_count = EXCLUDED.token_count,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id, (xmax = 0) AS inserted
                """
            else:
                # Use DO NOTHING to skip existing rows
                query = """
                    INSERT INTO librarian.document_locations
                    (kdocument_id, representation_type, s3_bucket, s3_key, content_length,
                     checksum, file_format, page_count, token_count, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
                    token_count,
                    row['created_at'],
                    datetime.utcnow()  # Update the updated_at timestamp
                )

                if result:
                    if self.force and 'inserted' in result:
                        was_overwrite = not result['inserted']
                        return True, True, was_overwrite  # success, was_inserted=True, was_overwrite
                    return True, True, False  # success, was_inserted=True, not overwrite
                else:
                    # ON CONFLICT DO NOTHING - row already exists
                    return True, False, False  # success, was_inserted=False, not overwrite

            except Exception as e:
                print(f"❌ Error copying row for doc {row['kdocument_id']}: {e}")
                return False, False, False

    async def verify_migration(self, kdocument_id: int, representation_type: str, original_s3_key: str) -> bool:
        """Verify that tag extraction was successful."""
        try:
            # Check database entry has representation_tags
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, representation_tags
                    FROM librarian.document_locations_v2
                    WHERE kdocument_id = $1 AND representation_type::text = $2
                """
                row = await conn.fetchrow(query, kdocument_id, representation_type)

                if not row:
                    print(f"❌ Verification failed: No database entry for doc {kdocument_id}, rep {representation_type}")
                    return False

                if not row['representation_tags']:
                    print(f"⚠️  Warning: representation_tags column is empty for doc {kdocument_id}, rep {representation_type}")
                    # This is ok if the file had no tags

            # Check S3 file exists and doesn't have representation_tags
            content = await self.download_from_s3(original_s3_key)
            if content is None:
                print(f"❌ Verification failed: Cannot download file from S3: {original_s3_key}")
                return False

            representation_tags, _, _ = self.extract_tags(content, representation_type)
            if representation_tags:
                print(f"❌ Verification failed: File still has representation_tags: {original_s3_key}")
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
            "overwritten": 0,
        }

        for row in rows:
            kdocument_id = row['kdocument_id']
            representation_type = row['representation_type']
            s3_key = row['s3_key']

            print(f"\n📄 Processing doc {kdocument_id}, representation {representation_type}")

            # Check if already processed
            already_processed, _ = await self.check_already_processed(kdocument_id, representation_type)
            if already_processed and not self.force:
                print(f"⏭️  representation_tags already extracted, skipping (use --force to overwrite)")
                batch_stats['skipped'] += 1
                continue
            elif already_processed and self.force:
                print(f"⚠️  representation_tags already exist, but will overwrite (--force enabled)")
                is_overwrite = True
            else:
                is_overwrite = False

            # Download file from S3
            print(f"📥 Downloading from S3: {s3_key}")
            content = await self.download_from_s3(s3_key)

            if content is None:
                print(f"❌ Failed to download file")
                batch_stats['failed'] += 1
                continue

            # Save original content if requested (before processing)
            if self.download_original_dir:
                if self.save_original_file(kdocument_id, representation_type, content):
                    self.stats['downloaded_original'] += 1

            # Extract representation_tags (pass representation_type for hardcoded verification)
            representation_tags, cleaned_content, success = self.extract_tags(content, representation_type)

            if not success:
                print(f"❌ Tag extraction failed - structure doesn't match expected format")
                batch_stats['failed'] += 1
                continue

            if representation_tags:
                print(f"🏷️  Extracted {len(representation_tags)} characters of representation_tags")
                batch_stats['tags_extracted'] += 1
            else:
                print(f"ℹ️  No representation_tags found in document")
                batch_stats['no_tags'] += 1

            # Update representation_tags in document_locations_v2
            print(f"💾 Updating representation_tags in document_locations_v2")
            if not await self.update_tags_in_v2(kdocument_id, representation_type, representation_tags):
                print(f"❌ Failed to update database")
                batch_stats['failed'] += 1
                continue

            # Upload cleaned content back to S3
            print(f"📤 Uploading cleaned content to S3")
            if not await self.upload_to_s3(s3_key, cleaned_content):
                print(f"❌ Failed to upload cleaned content")
                batch_stats['failed'] += 1
                continue

            # Save cleaned content locally if requested
            if self.download_cleaned_dir:
                if self.save_cleaned_file(kdocument_id, representation_type, cleaned_content):
                    self.stats['downloaded_cleaned'] += 1

            # Verify migration
            print(f"🔍 Verifying tag extraction")
            if not await self.verify_migration(kdocument_id, representation_type, s3_key):
                print(f"❌ Verification failed")
                batch_stats['failed'] += 1
                continue

            if is_overwrite:
                print(f"✅ Successfully overwritten doc {kdocument_id}, rep {representation_type}")
                batch_stats['overwritten'] += 1
            else:
                print(f"✅ Successfully processed doc {kdocument_id}, rep {representation_type}")
            batch_stats['migrated'] += 1

        return batch_stats

    async def copy_to_document_locations_batch(self, rows: List[dict]) -> dict:
        """Copy a batch of rows to document_locations."""
        batch_stats = {
            "copied": 0,
            "skipped": 0,
            "failed": 0,
            "overwritten": 0,
        }

        for row in rows:
            kdocument_id = row['kdocument_id']
            representation_type = row['representation_type']

            print(f"\n📄 Copying doc {kdocument_id}, representation {representation_type}")

            # Copy row to document_locations (let database handle conflicts)
            print(f"💾 Copying to document_locations (token_count computed from content_length)")
            success, was_inserted, was_overwrite = await self.copy_row_to_document_locations(row)
            
            if not success:
                print(f"❌ Failed to copy to document_locations")
                batch_stats['failed'] += 1
                continue

            if was_inserted:
                if was_overwrite:
                    print(f"✅ Successfully overwritten in document_locations: doc {kdocument_id}, rep {representation_type}")
                    batch_stats['overwritten'] += 1
                else:
                    print(f"✅ Successfully inserted into document_locations: doc {kdocument_id}, rep {representation_type}")
                batch_stats['copied'] += 1
            else:
                print(f"⏭️  Already exists in document_locations: doc {kdocument_id}, rep {representation_type}")
                batch_stats['skipped'] += 1

        return batch_stats

    async def filter_docs_with_tags_in_v2(self, doc_ids: List[int]) -> List[int]:
        """
        Filter out doc_ids that already have tags extracted in document_locations_v2.
        Only filters if --force is not enabled.
        
        Returns:
            List of doc_ids that need tag extraction
        """
        if self.force:
            return doc_ids
        
        async with self.db_pool.acquire() as conn:
            # Find doc_ids where BOTH DOCLING and DOCLING_IMG have non-null tags in v2
            query = """
                SELECT kdocument_id
                FROM librarian.document_locations_v2
                WHERE kdocument_id = ANY($1)
                  AND representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                  AND representation_tags IS NOT NULL
                  AND representation_tags != ''
                GROUP BY kdocument_id
                HAVING COUNT(DISTINCT representation_type::text) = 2
            """
            completed_rows = await conn.fetch(query, doc_ids)
            completed_doc_ids = {row['kdocument_id'] for row in completed_rows}
            
            if completed_doc_ids:
                remaining = [doc_id for doc_id in doc_ids if doc_id not in completed_doc_ids]
                print(f"✅ {len(completed_doc_ids)} documents already have DOCLING and DOCLING_IMG tags extracted in v2")
                print(f"📋 {len(remaining)} documents need tag extraction")
                return remaining
            else:
                return doc_ids

    async def filter_completed_docs(self, doc_ids: List[int]) -> List[int]:
        """
        Filter out doc_ids that already have both DOCLING and DOCLING_IMG in document_locations.
        Only filters if --force is not enabled.
        
        Returns:
            List of doc_ids that need processing
        """
        if self.force:
            print("⚠️  Force mode: will process all documents even if complete")
            return doc_ids
        
        async with self.db_pool.acquire() as conn:
            # Find doc_ids that have BOTH DOCLING and DOCLING_IMG representations in document_locations
            query = """
                SELECT kdocument_id
                FROM librarian.document_locations
                WHERE kdocument_id = ANY($1)
                  AND representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                GROUP BY kdocument_id
                HAVING COUNT(DISTINCT representation_type::text) = 2
            """
            completed_rows = await conn.fetch(query, doc_ids)
            completed_doc_ids = {row['kdocument_id'] for row in completed_rows}
            
            if completed_doc_ids:
                remaining = [doc_id for doc_id in doc_ids if doc_id not in completed_doc_ids]
                print(f"✅ {len(completed_doc_ids)} documents already have DOCLING and DOCLING_IMG in document_locations")
                print(f"📋 {len(remaining)} documents need copying to document_locations")
                return remaining
            else:
                return doc_ids

    async def migrate(self, doc_ids: List[int], batch_size: int = 100):
        """Migrate documents in batches."""
        print(f"\n🚀 Starting migration for {len(doc_ids)} document IDs")
        print(f"📦 Batch size: {batch_size}")
        print(f"🪣 S3 Bucket: {self.s3_bucket}")
        print(f"🗄️  Database: {self.db_config['database']}")

        # Track original count
        self.stats['total_doc_ids'] = len(doc_ids)
        original_count = len(doc_ids)

        # FIRST: Filter out documents that are already in document_locations (unless --force)
        # This is the main check - if they're already in document_locations, we're done!
        print(f"\n🔍 Checking document_locations for already-complete documents...")
        doc_ids = await self.filter_completed_docs(doc_ids)
        self.stats['filtered_complete'] = original_count - len(doc_ids)
        
        if not doc_ids:
            print("\n✅ All documents already complete in document_locations! Nothing to process.")
            return

        # SECOND: Filter out documents that already have tags extracted in v2 (unless --force)
        # This catches cases where Phase 1 is done but Phase 2 isn't
        print(f"\n🔍 Checking document_locations_v2 for already-extracted tags...")
        after_locations_filter = len(doc_ids)
        doc_ids = await self.filter_docs_with_tags_in_v2(doc_ids)
        self.stats['filtered_tags_extracted'] = after_locations_filter - len(doc_ids)
        
        if not doc_ids:
            print("\n✅ Remaining documents already have tags extracted! Nothing to process.")
            return

        # Fetch all rows from v2
        print(f"\n📊 Fetching rows from document_locations_v2...")
        all_rows = await self.fetch_rows_from_v2(doc_ids)

        if not all_rows:
            print("⚠️  No rows found in document_locations_v2 for specified doc_ids")
            return

        print(f"✅ Found {len(all_rows)} rows to migrate")
        self.stats['total_rows'] = len(all_rows)

        # Process in batches (Phase 1 + Phase 2 per batch)
        total_batches = (len(all_rows) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            print(f"\n{'='*60}")
            print(f"📦 BATCH {batch_num}/{total_batches} ({len(batch)} rows)")
            print(f"{'='*60}")

            # PHASE 1: Extract tags and update document_locations_v2
            print(f"\n📋 Phase 1: Extracting tags from S3 and updating document_locations_v2")
            
            batch_stats = await self.process_batch(batch)

            # Update overall stats
            self.stats['migrated'] += batch_stats['migrated']
            self.stats['skipped'] += batch_stats['skipped']
            self.stats['failed'] += batch_stats['failed']
            self.stats['tags_extracted'] += batch_stats['tags_extracted']
            self.stats['no_tags'] += batch_stats['no_tags']
            self.stats['overwritten'] += batch_stats['overwritten']

            print(f"\n📊 Phase 1 Summary:")
            print(f"   ✅ Processed: {batch_stats['migrated']}")
            if batch_stats['overwritten'] > 0:
                print(f"   🔄 Overwritten: {batch_stats['overwritten']}")
            print(f"   ⏭️  Skipped: {batch_stats['skipped']}")
            print(f"   ❌ Failed: {batch_stats['failed']}")
            print(f"   🏷️  Tags extracted: {batch_stats['tags_extracted']}")
            print(f"   ℹ️  No tags: {batch_stats['no_tags']}")

            # PHASE 2: Copy rows to document_locations
            print(f"\n📋 Phase 2: Copying rows to document_locations")

            copy_stats = await self.copy_to_document_locations_batch(batch)

            # Update overall stats
            self.stats['copied_to_locations'] += copy_stats['copied']
            self.stats['copy_skipped'] += copy_stats['skipped']
            self.stats['copy_failed'] += copy_stats['failed']
            self.stats['copy_overwritten'] += copy_stats['overwritten']

            print(f"\n📊 Phase 2 Summary:")
            print(f"   ✅ Inserted: {copy_stats['copied']}")
            if copy_stats['overwritten'] > 0:
                print(f"   🔄 Overwritten: {copy_stats['overwritten']}")
            print(f"   ⏭️  Skipped: {copy_stats['skipped']}")
            print(f"   ❌ Failed: {copy_stats['failed']}")
            
            print(f"\n{'='*60}")
            print(f"✅ Batch {batch_num}/{total_batches} complete")
            print(f"{'='*60}")

    def print_summary(self):
        """Print final migration summary."""
        print(f"\n{'='*60}")
        print(f"📊 MIGRATION SUMMARY")
        print(f"{'='*60}")
        print(f"\n📋 Input")
        print(f"   Total document IDs requested:   {self.stats['total_doc_ids']}")
        total_filtered = self.stats['filtered_tags_extracted'] + self.stats['filtered_complete']
        if total_filtered > 0:
            if self.stats['filtered_tags_extracted'] > 0:
                print(f"   ⏭️  Skipped (tags extracted):    {self.stats['filtered_tags_extracted']}")
            if self.stats['filtered_complete'] > 0:
                print(f"   ⏭️  Skipped (already complete):  {self.stats['filtered_complete']}")
            print(f"   📋 Processed:                   {self.stats['total_doc_ids'] - total_filtered}")
        
        print(f"\n📋 Phase 1: Tag Extraction (document_locations_v2)")
        print(f"   Total rows:                     {self.stats['total_rows']}")
        print(f"   ✅ Processed:                   {self.stats['migrated']}")
        if self.stats['overwritten'] > 0:
            print(f"   🔄 Overwritten:                 {self.stats['overwritten']}")
        print(f"   ⏭️  Skipped (already processed): {self.stats['skipped']}")
        print(f"   ❌ Failed:                      {self.stats['failed']}")
        print(f"   🏷️  representation_tags extracted: {self.stats['tags_extracted']}")
        print(f"   ℹ️  No representation_tags:        {self.stats['no_tags']}")
        if self.download_original_dir:
            print(f"   💾 Original files downloaded:   {self.stats['downloaded_original']}")
        if self.download_cleaned_dir:
            print(f"   💾 Cleaned files downloaded:    {self.stats['downloaded_cleaned']}")
        print(f"\n📋 Phase 2: Copy to document_locations")
        print(f"   ✅ Inserted:                    {self.stats['copied_to_locations']}")
        if self.stats['copy_overwritten'] > 0:
            print(f"   🔄 Overwritten:                 {self.stats['copy_overwritten']}")
        print(f"   ⏭️  Skipped (already exists):    {self.stats['copy_skipped']}")
        print(f"   ❌ Failed:                      {self.stats['copy_failed']}")
        print(f"{'='*60}")
        if self.download_original_dir:
            print(f"\n📁 Original files (with tags) saved to: {self.download_original_dir}")
            print(f"   (organized by doc_id subdirectories)")
        if self.download_cleaned_dir:
            print(f"\n📁 Cleaned files (tags removed) saved to: {self.download_cleaned_dir}")
            print(f"   (organized by doc_id subdirectories)")


async def main():
    parser = argparse.ArgumentParser(
        description='Extract representation_tags in document_locations_v2 and copy rows to document_locations'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start-id', type=int, help='Start document ID (inclusive)')
    group.add_argument('--doc-ids', type=str, help='Comma-separated list of document IDs')

    parser.add_argument('--end-id', type=int, help='End document ID (inclusive, requires --start-id)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process (useful for testing)')
    parser.add_argument('--download-cleaned', type=str, metavar='DIR', help='Download cleaned files (tags removed) to this directory')
    parser.add_argument('--download-original', type=str, metavar='DIR', help='Download original files (with tags) to this directory')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing rows in both tables')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
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

    # Apply limit if specified
    if args.limit and args.limit < len(doc_ids):
        original_count = len(doc_ids)
        doc_ids = doc_ids[:args.limit]
        print(f"🎯 Target document IDs: {len(doc_ids)} documents (limited from {original_count})")
    else:
        print(f"🎯 Target document IDs: {len(doc_ids)} documents")
    
    print(f"   Range: {min(doc_ids)} to {max(doc_ids)}")

    # Confirm before proceeding
    if not args.yes:
        print("\n⚠️  This will:")
        print("   1. Extract representation_tags from S3 files")
        print("   2. Update document_locations_v2 with extracted tags")
        print("   3. Upload cleaned files back to S3 (tags removed)")
        print("   4. Copy rows to document_locations with computed token_count")
        response = input("\nContinue? (yes/NO): ")
        if response.lower() != 'yes':
            print("❌ Migration cancelled")
            sys.exit(0)
    else:
        print("\n✅ Skipping confirmation (--yes flag provided)")

    # Initialize migrator
    download_cleaned_dir = Path(args.download_cleaned) if args.download_cleaned else None
    download_original_dir = Path(args.download_original) if args.download_original else None
    migrator = DocumentLocationMigrator(
        aws_profile=args.profile,
        download_cleaned_dir=download_cleaned_dir,
        download_original_dir=download_original_dir,
        force=args.force
    )

    try:
        await migrator.initialize()
        await migrator.migrate(doc_ids, batch_size=args.batch_size)
        migrator.print_summary()

        total_failures = migrator.stats['failed'] + migrator.stats['copy_failed']
        if total_failures > 0:
            print(f"\n⚠️  {total_failures} operations failed. Please review the logs above.")
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
