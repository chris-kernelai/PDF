"""Document fetching utilities using Supabase as the source of truth."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import aiofiles
import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor

from .supabase import (
    SupabaseConfig,
    fetch_doc_ids_missing_docling_img,
    fetch_existing_representations,
)
from .paths import LOGS_DIR, WORKSPACE_ROOT

logger = logging.getLogger(__name__)


@dataclass
class FetchStats:
    documents_considered: int = 0
    documents_selected: int = 0
    documents_downloaded: int = 0
    documents_skipped_existing: int = 0
    documents_skipped_completed: int = 0
    documents_failed: int = 0
    documents_rejected_not_pdf: int = 0
    failed_errors: List[str] = None  # List of error messages for failed downloads

    def __post_init__(self):
        if self.failed_errors is None:
            self.failed_errors = []


class DocumentFetcher:
    """Fetch documents that still need Docling/Docling_IMG representations."""

    def __init__(
        self,
        *,
        config_path: Path,
        limit: Optional[int] = None,
        randomize: bool = False,
        random_seed: int = 42,
        min_doc_id: Optional[int] = None,
        max_doc_id: Optional[int] = None,
        run_all_images: bool = False,
        download_pdfs: bool = True,
        specific_doc_ids: Optional[List[int]] = None,
    ) -> None:
        import yaml

        with open(config_path, "r") as fh:
            self.config: Dict = yaml.safe_load(fh)

        self.limit = limit
        self.randomize = randomize
        self.random_seed = random_seed
        self.min_doc_id = min_doc_id
        self.max_doc_id = max_doc_id
        self.run_all_images = run_all_images
        self.download_pdfs = download_pdfs
        self.specific_doc_ids = specific_doc_ids

        self.input_folder = self._resolve_path(self.config["paths"]["input_folder"])
        self.input_folder.mkdir(parents=True, exist_ok=True)

        self.supabase_config = SupabaseConfig.from_env()
        self.db_config = {
            "host": os.getenv("K_LIB_DB_HOST"),
            "port": os.getenv("K_LIB_DB_PORT"),
            "user": os.getenv("K_LIB_DB_USER"),
            "password": os.getenv("K_LIB_DB_PASSWORD"),
            "database": os.getenv("K_LIB_DB_NAME"),
            "options": "-c default_transaction_read_only=on",
        }

        self.completed_doc_ids = self._load_completed_documents()
        self.stats = FetchStats()
        self.last_selected_docs: List[Dict] = []

        self.documents_by_type: Counter[str] = Counter()
        self.documents_by_country: Counter[str] = Counter()
        self.companies_by_country: Counter[str] = Counter()
        self.documents_by_type_and_country: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> FetchStats:
        """Fetch documents and optionally download PDFs."""
        if self.specific_doc_ids is not None:
            # Use specific document IDs provided
            documents = self._load_documents_for_ids(self.specific_doc_ids)
        elif self.run_all_images:
            doc_ids = await fetch_doc_ids_missing_docling_img(self.supabase_config)
            # Filter by min/max doc ID range
            if self.min_doc_id is not None:
                doc_ids = [doc_id for doc_id in doc_ids if doc_id >= self.min_doc_id]
            if self.max_doc_id is not None:
                doc_ids = [doc_id for doc_id in doc_ids if doc_id <= self.max_doc_id]
            if self.limit is not None:
                doc_ids = doc_ids[: self.limit]
            documents = self._load_documents_for_ids(doc_ids)
        else:
            documents = self._load_documents_by_filters()

        if not documents:
            logger.warning("No candidate documents found")
            return self.stats

        document_ids = [doc["id"] for doc in documents]
        existing = await fetch_existing_representations(
            self.supabase_config, document_ids
        )

        selected_docs = self._select_documents(documents, existing)
        self.last_selected_docs = selected_docs

        if not selected_docs:
            logger.info("No documents require processing after Supabase filtering")
            return self.stats

        if self.download_pdfs:
            await self._download_documents(selected_docs)

        return self.stats

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if not path.is_absolute():
            path = WORKSPACE_ROOT / path
        return path

    def _load_documents_by_filters(self) -> List[Dict]:
        logger.info("Fetching documents using configured filters")
        documents: List[Dict] = []

        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                doc_types = self.config["filters"]["document_types"]
                countries_by_type = self.config["filters"]["countries_by_type"]

                for doc_type in doc_types:
                    logger.info("Fetching %s documents", doc_type)
                    exclude_countries = countries_by_type.get(doc_type, {}).get(
                        "exclude_countries", []
                    )

                    where_clauses = ["d.document_type::text = %s"]
                    params: List = [doc_type.upper()]

                    if exclude_countries:
                        where_clauses.append("c.country != ALL(%s::text[])")
                        params.append(exclude_countries)

                    where_clauses.append("c.country IS NOT NULL")

                    if self.min_doc_id is not None:
                        where_clauses.append("d.id >= %s")
                        params.append(self.min_doc_id)
                    if self.max_doc_id is not None:
                        where_clauses.append("d.id <= %s")
                        params.append(self.max_doc_id)

                    query = f"""
                        SELECT
                            d.id,
                            d.company_id,
                            d.document_type,
                            d.published_at AS filing_date,
                            d.source_identifier AS title,
                            d.created_at,
                            d.updated_at,
                            c.country,
                            c.ticker,
                            c.name AS company_name
                        FROM librarian.kdocuments d
                        JOIN public.company c ON d.isin = c.isin
                        WHERE {' AND '.join(where_clauses)}
                        ORDER BY d.id
                    """

                    cur.execute(query, params)
                    rows = [dict(row) for row in cur.fetchall()]
                    documents.extend(rows)

                    for row in rows:
                        country = row.get("country") or "UNKNOWN"
                        doc_type_text = row.get("document_type", doc_type.upper())
                        self.documents_by_type[doc_type_text] += 1
                        self.documents_by_country[country] += 1
                        self.documents_by_type_and_country[doc_type_text][country] += 1
                        self.companies_by_country[country] += 1

        finally:
            conn.close()

        self.stats.documents_considered = len(documents)
        logger.info("Found %s candidate documents", len(documents))
        return documents

    def _load_documents_for_ids(self, doc_ids: Sequence[int]) -> List[Dict]:
        if not doc_ids:
            return []

        logger.info("Loading metadata for %s specific document IDs", len(doc_ids))
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT
                        d.id,
                        d.company_id,
                        d.document_type,
                        d.published_at AS filing_date,
                        d.source_identifier AS title,
                        d.created_at,
                        d.updated_at,
                        c.country,
                        c.ticker,
                        c.name AS company_name
                    FROM librarian.kdocuments d
                    JOIN public.company c ON d.isin = c.isin
                    WHERE d.id = ANY(%s)
                    ORDER BY d.id
                """
                cur.execute(query, (doc_ids,))
                rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

        self.stats.documents_considered = len(rows)
        return rows

    def _select_documents(
        self,
        documents: Sequence[Dict],
        existing_reps: Dict[int, Set[str]],
    ) -> List[Dict]:
        docs = list(documents)

        if self.randomize:
            random.seed(self.random_seed)
            random.shuffle(docs)

        # Don't limit before filtering - we need to check all candidates
        # The limit will be applied after filtering to ensure we get enough documents
        
        selected: List[Dict] = []

        for doc in docs:
            doc_id = doc["id"]
            reps = existing_reps.get(doc_id, set())

            if not self.run_all_images:
                # Full run: skip if already has DOCLING representation
                if "DOCLING" in reps:
                    self.stats.documents_skipped_existing += 1
                    continue
            else:
                # Images-only run: skip if has DOCLING_IMG, or if doesn't have DOCLING
                if "DOCLING_IMG" in reps:
                    self.stats.documents_skipped_existing += 1
                    continue
                if "DOCLING" not in reps:
                    self.stats.documents_skipped_existing += 1
                    continue

            if not self.run_all_images and str(doc_id) in self.completed_doc_ids:
                self.stats.documents_skipped_completed += 1
                continue

            selected.append(doc)
            
            # Apply limit AFTER filtering - stop once we have enough valid documents
            if self.limit is not None and len(selected) >= self.limit:
                break

        self.stats.documents_selected = len(selected)
        logger.info("Selected %s documents after filtering", len(selected))
        return selected

    # ------------------------------------------------------------------
    # Download handling
    # ------------------------------------------------------------------

    async def _download_documents(self, documents: Sequence[Dict]) -> None:
        if not documents:
            return

        async with aiohttp.ClientSession() as session:
            await self._download_documents_batch(session, list(documents))

    async def _download_documents_batch(
        self, session: aiohttp.ClientSession, documents: List[Dict]
    ) -> None:
        batch_size = 250
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            document_ids = [doc["id"] for doc in batch]
            url_map = await self._get_download_urls_batch(session, document_ids)

            semaphore = asyncio.Semaphore(
                self.config["download"]["concurrent_downloads"]
            )

            async def download_one(doc: Dict) -> None:
                async with semaphore:
                    doc_id = doc["id"]
                    pdf_filename = f"doc_{doc_id}.pdf"
                    pdf_path = self.input_folder / pdf_filename

                    if pdf_path.exists():
                        logger.info("PDF %s already exists, skipping", pdf_filename)
                        return

                    pdf_url = url_map.get(doc_id)
                    if not pdf_url:
                        error_msg = f"doc_{doc_id}: No download URL"
                        logger.warning("No download URL for %s", doc_id)
                        self.stats.documents_failed += 1
                        self.stats.failed_errors.append(error_msg)
                        return

                    logger.info("Downloading document %s", doc_id)
                    saved_path = await self._download_pdf(session, doc_id, pdf_url)

                    if saved_path:
                        self.stats.documents_downloaded += 1
                    # Note: _download_pdf will handle adding to failed_errors if needed

                    await asyncio.sleep(self.config["download"]["rate_limit_delay"])

            await asyncio.gather(*(download_one(doc) for doc in batch))

    async def _get_download_urls_batch(
        self, session: aiohttp.ClientSession, document_ids: List[int]
    ) -> Dict[int, Optional[str]]:
        payload = {
            "documents": [
                {
                    "document_id": doc_id,
                    "representation_type": "raw",
                    "expires_in": 3600,
                }
                for doc_id in document_ids
            ]
        }

        response = await self._make_request(session, "kdocuments/batch/download", payload)
        if not response or "data" not in response:
            logger.error("Failed to fetch download URLs for batch")
            return {doc_id: None for doc_id in document_ids}

        results = response["data"].get("results", [])
        return {
            result.get("document_id"): result.get("download_url")
            for result in results
        }

    async def _download_pdf(
        self, session: aiohttp.ClientSession, document_id: int, pdf_url: str
    ) -> Optional[Path]:
        if not pdf_url:
            return None

        pdf_filename = f"doc_{document_id}.pdf"
        pdf_path = self.input_folder / pdf_filename

        try:
            async with session.get(pdf_url) as response:
                if response.status != 200:
                    error_msg = f"doc_{document_id}: HTTP {response.status}"
                    logger.error(
                        "Failed to download %s: HTTP %s", document_id, response.status
                    )
                    self.stats.documents_failed += 1
                    self.stats.failed_errors.append(error_msg)
                    return None

                async with aiofiles.open(pdf_path, "wb") as fh:
                    await fh.write(await response.read())

            if not await self._verify_pdf(pdf_path):
                logger.warning("Downloaded file for %s is not a PDF", document_id)
                pdf_path.unlink(missing_ok=True)
                self.stats.documents_rejected_not_pdf += 1
                return None

            return pdf_path
        except Exception as exc:
            error_msg = f"doc_{document_id}: {type(exc).__name__}: {str(exc)}"
            logger.exception("Failed to download document %s", document_id)
            pdf_path.unlink(missing_ok=True)
            self.stats.documents_failed += 1
            self.stats.failed_errors.append(error_msg)
            return None

    async def _verify_pdf(self, pdf_path: Path) -> bool:
        import subprocess

        try:
            result = subprocess.run(
                ["file", "--mime-type", "-b", str(pdf_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            mime_type = result.stdout.strip()
            return mime_type.startswith("application/pdf")
        except subprocess.TimeoutExpired:
            logger.warning("Timeout verifying %s", pdf_path)
            return False
        except Exception:
            logger.warning("Could not verify %s", pdf_path)
            return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_db_connection(self):
        return psycopg2.connect(**self.db_config)

    def _load_completed_documents(self) -> Set[str]:
        log_file = LOGS_DIR / "processing_log.csv"
        if not log_file.exists():
            return set()

        completed_ids: Set[str] = set()
        import csv

        try:
            with open(log_file, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("stage") == "conversion" and row.get("status") == "success":
                        doc_id = row.get("doc_id", "")
                        if doc_id.startswith("doc_"):
                            doc_id = doc_id[4:]
                        if doc_id:
                            completed_ids.add(doc_id)
        except Exception:
            logger.exception("Failed to load completed documents from processing log")
            return set()

        if completed_ids:
            logger.info("Loaded %s completed doc IDs from processing log", len(completed_ids))
        return completed_ids

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        payload: Dict,
        method: str = "POST",
    ) -> Optional[Dict]:
        base_url = self.config["api"]["base_url"]
        headers = self._build_headers()
        timeout = aiohttp.ClientTimeout(total=self.config["api"]["timeout"])

        max_retries = self.config["api"]["max_retries"]
        retry_delay = self.config["api"]["retry_delay"]

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    request_ctx = session.get(
                        f"{base_url}/{endpoint}",
                        params=payload,
                        headers=headers,
                        timeout=timeout,
                    )
                else:
                    request_ctx = session.post(
                        f"{base_url}/{endpoint}",
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )

                async with request_ctx as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 429:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue
                    logger.error(
                        "Request to %s failed with %s", endpoint, response.status
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
            except asyncio.TimeoutError:
                logger.warning("Request to %s timed out (attempt %s)", endpoint, attempt + 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            except aiohttp.ClientError:
                logger.exception("Request error to %s", endpoint)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        return None

    def _build_headers(self) -> Dict[str, str]:
        api_key = os.getenv("API_KEY") or self.config["api"].get("api_key")
        if not api_key or api_key.startswith("${"):
            raise RuntimeError("API_KEY environment variable must be set")
        return {"Authorization": f"Bearer {api_key}"}


__all__ = ["DocumentFetcher", "FetchStats"]
