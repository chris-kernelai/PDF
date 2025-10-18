#!/usr/bin/env python3
"""
Standalone script to upload document representations to S3 and update Supabase.
This script uploads files for new representation types: 'docling' and 'docling_img'.
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aioboto3
import asyncpg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DocumentRepresentationUploader:
    """Standalone class to upload document representations to S3 and update Supabase."""

    def __init__(
        self,
        aws_profile: Optional[str] = None,
        aws_region: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        s3_bucket: Optional[str] = None,
    ):
        # Load from environment variables with fallback to parameters
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE", "production")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "eu-west-2")
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

        # S3 bucket for librarian documents
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET", "primer-production-librarian-documents")

        # Database configuration
        self.db_config = {
            "host": db_host or os.getenv("DB_HOST"),
            "port": db_port or int(os.getenv("DB_PORT", "5432")),
            "database": db_name or os.getenv("DB_NAME", "postgres"),
            "user": db_user or os.getenv("DB_USER"),
            "password": db_password or os.getenv("DB_PASSWORD"),
        }

        # Initialize clients
        self.session = None
        self.s3_client = None
        self.db_pool = None

    async def initialize(self):
        """Initialize S3 and database connections."""
        # Initialize S3 client
        self.session = aioboto3.Session(
            profile_name=self.aws_profile, region_name=self.aws_region
        )
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
        print("✅ Closed all connections")

    def strip_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        import re
        # Emoji pattern - covers most emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    async def upload_file_to_s3(self, file_path: str, s3_key: str) -> dict:
        """Upload a file to S3 and return metadata."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content as text and strip emojis
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()

        # Strip emojis
        cleaned_content = self.strip_emojis(text_content)

        # Convert back to bytes for upload
        content = cleaned_content.encode("utf-8")

        # Upload to S3
        await self.s3_client.put_object(Bucket=self.s3_bucket, Key=s3_key, Body=content)

        # Calculate metadata - get file_format from S3 key, not source file
        s3_key_path = Path(s3_key)
        file_format = s3_key_path.suffix.lstrip(".") or "txt"
        checksum = hashlib.sha256(content).hexdigest()

        return {
            "s3_bucket": self.s3_bucket,
            "s3_key": s3_key,
            "content_length": len(content),
            "checksum": checksum,
            "file_format": file_format,
        }


    async def get_existing_representations(
        self, document_ids: list[int] = None
    ) -> dict[int, set[str]]:
        """
        Get existing representation types for documents.

        Args:
            document_ids: Optional list of document IDs to check. If None, checks all.

        Returns:
            dict mapping document_id to set of representation types (e.g., {'docling', 'docling_img'})
        """
        async with self.db_pool.acquire() as conn:
            if document_ids:
                query = """
                    SELECT kdocument_id, representation_type::text
                    FROM librarian.document_locations_v2
                    WHERE kdocument_id = ANY($1)
                    AND representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                """
                rows = await conn.fetch(query, document_ids)
            else:
                query = """
                    SELECT kdocument_id, representation_type::text
                    FROM librarian.document_locations_v2
                    WHERE representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                """
                rows = await conn.fetch(query)

            # Build a dict of document_id -> set of representation types
            existing = {}
            for row in rows:
                doc_id = row["kdocument_id"]
                rep_type = row["representation_type"]
                if doc_id not in existing:
                    existing[doc_id] = set()
                existing[doc_id].add(rep_type)

            return existing

    async def add_document_location(
        self,
        document_id: int,
        representation_type: str,
        s3_metadata: dict,
        page_count: Optional[int] = None,
        token_count: Optional[int] = None,
    ) -> bool:
        """Add a new document location to the document_locations_v2 table."""

        # Create the document location record
        location_data = {
            "kdocument_id": document_id,
            "representation_type": representation_type,
            "s3_bucket": s3_metadata["s3_bucket"],
            "s3_key": s3_metadata["s3_key"],
            "content_length": s3_metadata["content_length"],
            "checksum": s3_metadata["checksum"],
            "file_format": s3_metadata["file_format"],
            "page_count": page_count,
            "token_count": token_count,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        async with self.db_pool.acquire() as conn:
            # Insert into document_locations_v2 table
            insert_query = """
                INSERT INTO librarian.document_locations_v2
                (kdocument_id, representation_type, s3_bucket, s3_key, content_length,
                 checksum, file_format, page_count, token_count, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """

            result = await conn.fetchrow(
                insert_query,
                location_data["kdocument_id"],
                location_data["representation_type"],
                location_data["s3_bucket"],
                location_data["s3_key"],
                location_data["content_length"],
                location_data["checksum"],
                location_data["file_format"],
                location_data["page_count"],
                location_data["token_count"],
                location_data["created_at"],
                location_data["updated_at"],
            )

            print(f"✅ Added document location with ID: {result['id']}")
            return True

    async def upload_representations(
        self,
        document_id: int,
        docling_file: Optional[str] = None,
        docling_img_file: Optional[str] = None,
        docling_filename: Optional[str] = None,
        docling_img_filename: Optional[str] = None,
        page_count: Optional[int] = None,
        token_count: Optional[int] = None,
    ) -> dict:
        """
        Upload both docling and docling_img representations for a document.

        Args:
            document_id: The document ID
            docling_file: Path to the docling file (None to skip)
            docling_img_file: Path to the docling_img file (None to skip)
            docling_filename: Optional custom filename for docling (defaults to original)
            docling_img_filename: Optional custom filename for docling_img (defaults to original)
            page_count: Optional page count
            token_count: Optional token count

        Returns:
            dict: Results of the upload process
        """

        print(f"📄 Processing document ID: {document_id}")

        results = {
            "document_id": document_id,
            "uploads": {},
            "errors": [],
        }

        # Upload docling representation
        if docling_file:
            try:
                docling_path = Path(docling_file)
                docling_filename = docling_filename or docling_path.name
                docling_s3_key = (
                    f"documents/{document_id}/representations/docling/{docling_filename}"
                )

                print(f"📤 Uploading docling file: {docling_file}")
                docling_metadata = await self.upload_file_to_s3(
                    docling_file, docling_s3_key
                )

                await self.add_document_location(
                    document_id=document_id,
                    representation_type="DOCLING",
                    s3_metadata=docling_metadata,
                    page_count=page_count,
                    token_count=token_count,
                )

                results["uploads"]["docling"] = {
                    "s3_key": docling_s3_key,
                    "file_size": docling_metadata["content_length"],
                    "status": "success",
                }

            except Exception as e:
                error_msg = f"Failed to upload docling: {str(e)}"
                print(f"❌ {error_msg}")
                results["errors"].append(error_msg)
        else:
            print(f"⏭️  Skipping docling (already exists)")

        # Upload docling_img representation
        if docling_img_file:
            try:
                docling_img_path = Path(docling_img_file)
                docling_img_filename = docling_img_filename or docling_img_path.name
                docling_img_s3_key = f"documents/{document_id}/representations/docling_img/{docling_img_filename}"

                print(f"📤 Uploading docling_img file: {docling_img_file}")
                docling_img_metadata = await self.upload_file_to_s3(
                    docling_img_file, docling_img_s3_key
                )

                await self.add_document_location(
                    document_id=document_id,
                    representation_type="DOCLING_IMG",
                    s3_metadata=docling_img_metadata,
                    page_count=page_count,
                    token_count=token_count,
                )

                results["uploads"]["docling_img"] = {
                    "s3_key": docling_img_s3_key,
                    "file_size": docling_img_metadata["content_length"],
                    "status": "success",
                }

            except Exception as e:
                error_msg = f"Failed to upload docling_img: {str(e)}"
                print(f"❌ {error_msg}")
                results["errors"].append(error_msg)
        else:
            print(f"⏭️  Skipping docling_img (already exists)")

        return results


async def main():
    """Example usage of the DocumentRepresentationUploader."""

    # Initialize uploader
    uploader = DocumentRepresentationUploader()

    try:
        await uploader.initialize()

        # Example: Upload representations for document ID 123
        document_id = 123  # Replace with actual document ID
        docling_file = "path/to/your/docling_file.txt"  # Replace with actual file path
        docling_img_file = (
            "path/to/your/docling_img_file.png"  # Replace with actual file path
        )

        # Optional: Custom filenames
        docling_filename = "processed_docling.txt"
        docling_img_filename = "processed_docling_img.png"

        # Upload representations
        results = await uploader.upload_representations(
            document_id=document_id,
            docling_file=docling_file,
            docling_img_file=docling_img_file,
            docling_filename=docling_filename,
            docling_img_filename=docling_img_filename,
            page_count=10,  # Optional
            token_count=5000,  # Optional
        )

        # Print results
        print("\n📊 Upload Results:")
        print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        await uploader.close()


if __name__ == "__main__":
    # Install required packages first:
    # pip install aioboto3 asyncpg boto3

    asyncio.run(main())
