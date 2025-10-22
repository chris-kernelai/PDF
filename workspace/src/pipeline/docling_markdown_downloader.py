"""Utilities for downloading Docling markdown representations from Supabase.

This module reuses the same environment configuration as the rest of the
pipeline and downloads existing DOCLING markdown files from S3 when they are
already available in Supabase. These markdown files can be reused for the
images-only workflow, avoiding a redundant Docling conversion step.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import aioboto3
import asyncpg

from .supabase import SupabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class MarkdownDownloadResult:
    """Summary of a markdown download operation."""

    requested: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    missing: List[int] = field(default_factory=list)
    failed: Dict[int, str] = field(default_factory=dict)

    def add_missing(self, doc_id: int) -> None:
        if doc_id not in self.missing:
            self.missing.append(doc_id)

    def add_failure(self, doc_id: int, error: Exception) -> None:
        self.failed[doc_id] = str(error)


class DoclingMarkdownDownloader:
    """Download DOCLING markdown representations for the provided document IDs."""

    def __init__(
        self,
        *,
        output_dir: Path,
        supabase_config: SupabaseConfig | None = None,
        aws_profile: str | None = None,
        aws_region: str | None = None,
        s3_bucket: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.supabase_config = supabase_config or SupabaseConfig.from_env()
        self.aws_profile = aws_profile or os.getenv("AWS_PROFILE")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "eu-west-2")
        self.s3_bucket = s3_bucket or os.getenv(
            "S3_BUCKET", "primer-production-librarian-documents"
        )

        if not self.s3_bucket:
            raise RuntimeError(
                "S3_BUCKET environment variable must be set for markdown download"
            )

    async def _fetch_docling_locations(
        self, conn: asyncpg.Connection
    ) -> Dict[int, str]:
        query = """
            SELECT kdocument_id, s3_key
            FROM librarian.document_locations_v2
            WHERE representation_type::text = 'DOCLING'
        """

        rows = await conn.fetch(query)
        locations = {int(row["kdocument_id"]): row["s3_key"] for row in rows}
        logger.info("Fetched %s DOCLING entries from Supabase", len(locations))
        return locations

    async def _download_single(
        self,
        s3_client,
        doc_id: int,
        s3_key: str,
        result: MarkdownDownloadResult,
    ) -> None:
        target_path = self.output_dir / f"doc_{doc_id}.md"
        if target_path.exists():
            logger.debug("doc_%s.md already present; skipping download", doc_id)
            result.skipped_existing += 1
            return

        try:
            response = await s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            body = await response["Body"].read()
            target_path.write_bytes(body)
            logger.info("Downloaded doc_%s.md from %s", doc_id, s3_key)
            result.downloaded += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to download doc_%s markdown", doc_id)
            result.add_failure(doc_id, exc)

    async def download(self, doc_ids: Iterable[int]) -> MarkdownDownloadResult:
        doc_ids_list = sorted({int(doc_id) for doc_id in doc_ids})
        result = MarkdownDownloadResult(requested=len(doc_ids_list))

        if not doc_ids_list:
            return result

        pool = await asyncpg.create_pool(**self.supabase_config.to_dict())
        session_kwargs: Dict[str, str] = {"region_name": self.aws_region}
        # Only use profile if explicitly set; otherwise let boto3 use default credential chain (IAM role, etc.)
        if self.aws_profile:
            session_kwargs["profile_name"] = self.aws_profile

        session = aioboto3.Session(**session_kwargs)

        try:
            async with session.client("s3") as s3_client:
                async with pool.acquire() as conn:
                    locations = await self._fetch_docling_locations(conn)

                tasks: List[asyncio.Task[None]] = []
                for doc_id in doc_ids_list:
                    s3_key = locations.get(doc_id)
                    if not s3_key:
                        logger.warning(
                            "No DOCLING representation found for document %s", doc_id
                        )
                        result.add_missing(doc_id)
                        continue

                    tasks.append(
                        asyncio.create_task(
                            self._download_single(s3_client, doc_id, s3_key, result)
                        )
                    )

                if tasks:
                    await asyncio.gather(*tasks)
        finally:
            await pool.close()

        return result


__all__ = ["DoclingMarkdownDownloader", "MarkdownDownloadResult"]
