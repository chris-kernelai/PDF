"""Helpers for interacting with Supabase document state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import asyncpg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupabaseConfig:
    """Connection settings for Supabase Postgres."""

    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """Create config from standard environment variables."""
        import os

        try:
            return cls(
                host=os.environ["DB_HOST"],
                port=int(os.environ.get("DB_PORT", "5432")),
                database=os.environ.get("DB_NAME", "postgres"),
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
            )
        except KeyError as exc:  # pragma: no cover - defensive guard
            missing = exc.args[0]
            raise RuntimeError(
                f"Missing required Supabase environment variable: {missing}"
            ) from exc

    def to_dict(self) -> Dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }


async def _create_pool(config: SupabaseConfig) -> asyncpg.Pool:
    """Create an asyncpg connection pool."""
    return await asyncpg.create_pool(**config.to_dict())


async def fetch_existing_representations(
    config: SupabaseConfig,
    document_ids: Optional[Sequence[int]] = None,
    batch_size: int = 1000,
) -> Dict[int, Set[str]]:
    """Return representation types already uploaded for each document."""
    pool = await _create_pool(config)
    existing: Dict[int, Set[str]] = {}

    try:
        async with pool.acquire() as conn:
            offset = 0
            while True:
                if document_ids:
                    query = """
                        SELECT kdocument_id, representation_type::text
                        FROM librarian.document_locations_v2
                        WHERE kdocument_id = ANY($1)
                          AND representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                        ORDER BY kdocument_id
                        LIMIT $2 OFFSET $3
                    """
                    rows = await conn.fetch(query, list(document_ids), batch_size, offset)
                else:
                    query = """
                        SELECT kdocument_id, representation_type::text
                        FROM librarian.document_locations_v2
                        WHERE representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                        ORDER BY kdocument_id
                        LIMIT $1 OFFSET $2
                    """
                    rows = await conn.fetch(query, batch_size, offset)

                if not rows:
                    break

                for row in rows:
                    doc_id = row["kdocument_id"]
                    rep_type = row["representation_type"]
                    existing.setdefault(doc_id, set()).add(rep_type)

                offset += batch_size
                if len(rows) < batch_size:
                    break

    except Exception:
        logger.exception("Failed to fetch existing representations from Supabase")
        return existing
    finally:
        await pool.close()

    logger.info("Found %s documents with existing representations", len(existing))
    return existing


async def fetch_doc_ids_missing_docling_img(
    config: SupabaseConfig,
) -> List[int]:
    """Return document IDs that have DOCLING but lack DOCLING_IMG."""
    pool = await _create_pool(config)
    doc_ids: List[int] = []

    try:
        async with pool.acquire() as conn:
            query = """
                SELECT DISTINCT d.kdocument_id
                FROM librarian.document_locations_v2 d
                WHERE d.representation_type::text = 'DOCLING'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM librarian.document_locations_v2 img
                    WHERE img.kdocument_id = d.kdocument_id
                      AND img.representation_type::text = 'DOCLING_IMG'
                )
                ORDER BY d.kdocument_id
            """
            rows = await conn.fetch(query)
            doc_ids = [row["kdocument_id"] for row in rows]
    except Exception:
        logger.exception("Failed to fetch DOC IDs missing DOCLING_IMG from Supabase")
    finally:
        await pool.close()

    logger.info("Found %s documents missing DOCLING_IMG", len(doc_ids))
    return doc_ids
