"""
Document Fetcher

Fetches non-US company filings and slides by querying PostgreSQL database directly,
then downloads PDFs using the batch download API.
"""

import asyncio
import aiohttp
import aiofiles
import os
import yaml
import logging
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dotenv import load_dotenv
from src.document_metadata import MetadataManager
import asyncpg

# Load environment variables from .env file
load_dotenv()


class DocumentFetcher:
    """Fetches documents by querying PostgreSQL database directly."""

    def __init__(self, config_path: str = "config.yaml", limit: Optional[int] = None, randomize: bool = False, random_seed: int = 42, min_doc_id: Optional[int] = None, max_doc_id: Optional[int] = None, run_all_images: bool = False):
        """
        Initialize document fetcher.

        Args:
            config_path: Path to configuration file.
            limit: Maximum number of documents to fetch (None = no limit).
            randomize: Whether to randomize document order for sampling.
            random_seed: Random seed for reproducible sampling (default: 42).
            min_doc_id: Minimum document ID to fetch (filter out documents with ID < this value).
            max_doc_id: Maximum document ID to fetch (filter out documents with ID > this value).
            run_all_images: If True, fetch documents that have DOCLING but not DOCLING_IMG (ignores min/max_doc_id).
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

        # Extract config values
        self.base_url = self.config["api"]["base_url"]
        api_key = os.getenv("API_KEY") or self.config["api"]["api_key"]
        if api_key.startswith("${"):
            raise ValueError("API_KEY environment variable not set")

        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = aiohttp.ClientTimeout(total=self.config["api"]["timeout"])

        self.input_folder = Path(self.config["paths"]["input_folder"])
        self.input_folder.mkdir(parents=True, exist_ok=True)

        self.metadata = MetadataManager(self.config["paths"]["metadata_db"])

        # Document limit and sampling
        self.limit = limit
        self.randomize = randomize
        self.random_seed = random_seed
        self.min_doc_id = min_doc_id
        self.max_doc_id = max_doc_id
        self.run_all_images = run_all_images

        # Load completed documents from processing log
        self.completed_doc_ids = self._load_completed_documents()

        # Database connection parameters for kdocuments (read-only)
        self.db_config = {
            "host": os.getenv("K_LIB_DB_HOST"),
            "port": os.getenv("K_LIB_DB_PORT"),
            "user": os.getenv("K_LIB_DB_USER"),
            "password": os.getenv("K_LIB_DB_PASSWORD"),
            "database": os.getenv("K_LIB_DB_NAME"),
        }

        # Supabase database connection parameters for checking existing representations
        self.supabase_db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        # Cache for existing representations in Supabase
        self.existing_representations: Dict[int, Set[str]] = {}

        if self.min_doc_id:
            self.logger.info(f"Will filter documents with ID >= {self.min_doc_id}")
        if self.max_doc_id:
            self.logger.info(f"Will filter documents with ID <= {self.max_doc_id}")

        if self.randomize:
            random.seed(self.random_seed)
            self.logger.info(f"Randomization enabled with seed {self.random_seed}")

        # Statistics
        self.stats = {
            "companies_found": 0,
            "documents_found": 0,
            "documents_added": 0,
            "documents_skipped": 0,
            "documents_skipped_completed": 0,
            "documents_skipped_has_both_reps": 0,
            "documents_downloaded": 0,
            "documents_failed": 0,
            "documents_rejected_not_pdf": 0,
            "filings_found": 0,
            "slides_found": 0,
        }

        # Detailed statistics using collections
        from collections import defaultdict, Counter
        self.documents_by_type = Counter()  # Count of each document type
        self.documents_by_country = Counter()  # Count of documents per country
        self.companies_by_country = Counter()  # Count of companies per country
        self.documents_by_type_and_country = defaultdict(lambda: defaultdict(int))  # type -> country -> count

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config["logging"]["level"])
        log_file = self.config["logging"]["file"]

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _load_completed_documents(self) -> set:
        """
        Load document IDs that have been successfully converted from processing_log.csv.

        Returns:
            Set of document IDs (as strings) that have been successfully converted.
        """
        import csv

        log_file = Path("processing_log.csv")
        if not log_file.exists():
            return set()

        completed_ids = set()
        try:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only include documents that have completed conversion stage successfully
                    if row.get('stage') == 'conversion' and row.get('status') == 'success':
                        doc_id = row.get('doc_id', '')
                        # Remove 'doc_' prefix if present
                        if doc_id.startswith('doc_'):
                            doc_id = doc_id[4:]
                        if doc_id:
                            completed_ids.add(doc_id)

            if completed_ids:
                self.logger.info(f"Loaded {len(completed_ids)} completed documents from processing log")
            return completed_ids
        except Exception as e:
            self.logger.warning(f"Could not load completed documents from processing log: {e}")
            return set()

    async def _fetch_existing_representations_from_supabase(
        self, document_ids: Optional[List[int]] = None
    ) -> Dict[int, Set[str]]:
        """
        Fetch existing document representations from Supabase with pagination.

        Args:
            document_ids: Optional list of document IDs to check. If None, checks all.

        Returns:
            Dict mapping document_id to set of representation types (e.g., {'DOCLING', 'DOCLING_IMG'})
        """
        self.logger.info("Checking Supabase for existing document representations...")

        try:
            # Create asyncpg connection pool
            pool = await asyncpg.create_pool(**self.supabase_db_config)

            try:
                page_size = 1000
                offset = 0
                existing = {}

                async with pool.acquire() as conn:
                    while True:
                        # Query with pagination
                        if document_ids:
                            query = """
                                SELECT kdocument_id, representation_type::text
                                FROM librarian.document_locations_v2
                                WHERE kdocument_id = ANY($1)
                                AND representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                                ORDER BY kdocument_id
                                LIMIT $2 OFFSET $3
                            """
                            rows = await conn.fetch(query, document_ids, page_size, offset)
                        else:
                            query = """
                                SELECT kdocument_id, representation_type::text
                                FROM librarian.document_locations_v2
                                WHERE representation_type::text IN ('DOCLING', 'DOCLING_IMG')
                                ORDER BY kdocument_id
                                LIMIT $1 OFFSET $2
                            """
                            rows = await conn.fetch(query, page_size, offset)

                        # Break if no more results
                        if not rows:
                            break

                        # Process results
                        for row in rows:
                            doc_id = row["kdocument_id"]
                            rep_type = row["representation_type"]
                            if doc_id not in existing:
                                existing[doc_id] = set()
                            existing[doc_id].add(rep_type)

                        # Move to next page
                        offset += page_size
                        self.logger.debug(f"Fetched {len(rows)} representations (offset: {offset})")

                        # If we got fewer results than page_size, we're done
                        if len(rows) < page_size:
                            break

                self.logger.info(
                    f"Found {len(existing)} documents with existing representations in Supabase"
                )
                return existing

            finally:
                await pool.close()

        except Exception as e:
            self.logger.error(f"Failed to check Supabase for existing representations: {e}")
            # Return empty dict on error - we'll proceed without filtering
            return {}

    async def _fetch_documents_missing_docling_img(self) -> List[int]:
        """
        Fetch document IDs from Supabase that have DOCLING representation but not DOCLING_IMG.

        Returns:
            List of document IDs that need image processing.
        """
        self.logger.info("Fetching documents missing DOCLING_IMG representation from Supabase...")

        try:
            # Create asyncpg connection pool
            pool = await asyncpg.create_pool(**self.supabase_db_config)

            try:
                async with pool.acquire() as conn:
                    # Find documents that have DOCLING but not DOCLING_IMG
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
                    
                    self.logger.info(
                        f"Found {len(doc_ids)} documents with DOCLING but missing DOCLING_IMG"
                    )
                    return doc_ids

            finally:
                await pool.close()

        except Exception as e:
            self.logger.error(f"Failed to fetch documents missing DOCLING_IMG: {e}")
            return []

    async def _make_request(
        self, session: aiohttp.ClientSession, endpoint: str, payload: Dict, method: str = "POST"
    ) -> Optional[Dict]:
        """
        Make API request with retry logic.

        Args:
            session: aiohttp session.
            endpoint: API endpoint path.
            payload: Request payload.
            method: HTTP method (POST or GET).

        Returns:
            Response data or None on failure.
        """
        url = f"{self.base_url}/{endpoint}"
        max_retries = self.config["api"]["max_retries"]
        retry_delay = self.config["api"]["retry_delay"]

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    request_ctx = session.get(
                        url, params=payload, headers=self.headers, timeout=self.timeout
                    )
                else:
                    request_ctx = session.post(
                        url, json=payload, headers=self.headers, timeout=self.timeout
                    )

                async with request_ctx as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited
                        wait_time = retry_delay * (2**attempt)
                        self.logger.warning(
                            f"Rate limited, waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.error(
                            f"Request failed with status {response.status}: {await response.text()}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                        else:
                            return None

            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            except aiohttp.ClientError as e:
                self.logger.error(f"Request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        return None

    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            database=self.db_config["database"],
            options="-c default_transaction_read_only=on",  # READ-ONLY mode
        )

    def check_us_documents(self) -> Dict:
        """
        Check US documents for comparison (even though we don't process them).

        Returns:
            Dict with US document statistics.
        """
        self.logger.info("Checking US documents (for reference, not processing)...")

        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count US companies
                cur.execute("""
                    SELECT COUNT(*) as count
                    FROM librarian.company
                    WHERE country = 'United States'
                """)
                us_companies = cur.fetchone()["count"]

                # Count US documents by type
                query = """
                    SELECT
                        d.document_type::text,
                        COUNT(*) as count
                    FROM librarian.kdocuments d
                    JOIN librarian.company c ON d.company_id = c.id
                    WHERE c.country = 'United States'
                """

                # Add document ID range filters if set
                where_clauses = []
                params = []
                if self.min_doc_id:
                    where_clauses.append("d.id >= %s")
                    params.append(self.min_doc_id)
                if self.max_doc_id:
                    where_clauses.append("d.id <= %s")
                    params.append(self.max_doc_id)

                if where_clauses:
                    query += " AND " + " AND ".join(where_clauses)

                query += " GROUP BY d.document_type::text ORDER BY count DESC"

                cur.execute(query, params)
                us_docs_by_type = cur.fetchall()

                total_us_docs = sum(row["count"] for row in us_docs_by_type)

                self.logger.info(f"US Companies: {us_companies:,}")
                self.logger.info(f"US Documents (in ID range): {total_us_docs:,}")
                if us_docs_by_type:
                    self.logger.info("  By type:")
                    for row in us_docs_by_type:
                        self.logger.info(f"    {row['document_type']:20} {row['count']:6,} documents")

                return {
                    "companies": us_companies,
                    "total_documents": total_us_docs,
                    "by_type": us_docs_by_type
                }
        finally:
            conn.close()

    def check_available_document_types(self) -> List[str]:
        """
        Check all available document types in the database.
        This helps identify if we're missing any document types in our config.

        Returns:
            List of all document types found in the database.
        """
        self.logger.info("Checking all available document types in database...")

        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all document types (no country filter for this check)
                query = """
                    SELECT DISTINCT d.document_type::text
                    FROM librarian.kdocuments d
                    JOIN librarian.company c ON d.company_id = c.id
                    ORDER BY d.document_type::text
                """

                cur.execute(query)
                results = cur.fetchall()

                all_types = [row["document_type"] for row in results]

                # Compare with configured types
                configured_types = [dt.upper() for dt in self.config["filters"]["document_types"]]
                missing_types = [dt for dt in all_types if dt not in configured_types]

                self.logger.info(f"Found {len(all_types)} distinct document types in database:")
                for doc_type in all_types:
                    if doc_type in configured_types:
                        self.logger.info(f"  ✓ {doc_type} (configured)")
                    else:
                        self.logger.info(f"  ✗ {doc_type} (NOT configured - will be skipped)")

                if missing_types:
                    self.logger.warning(
                        f"⚠️  WARNING: {len(missing_types)} document type(s) are NOT in config and will be skipped: "
                        f"{', '.join(missing_types)}"
                    )
                else:
                    self.logger.info("✓ All available document types are configured")

                return all_types
        finally:
            conn.close()

    def fetch_documents_by_type_from_db(self) -> List[Dict]:
        """
        Fetch documents from database with type-specific country filters.

        Returns:
            List of document dictionaries with country information.
        """
        self.logger.info("Fetching documents from database with type-specific country filters...")

        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                all_documents = []
                
                # Get document types and their country filters
                doc_types = self.config["filters"]["document_types"]
                countries_by_type = self.config["filters"]["countries_by_type"]
                
                for doc_type in doc_types:
                    self.logger.info(f"Fetching {doc_type} documents...")
                    
                    # Get exclusion list for this document type
                    exclude_countries = countries_by_type.get(doc_type, {}).get("exclude_countries", [])
                    
                    # Build query with filters
                    where_clauses = ["d.document_type::text = %s"]
                    params = [doc_type.upper()]
                    
                    # Country exclusion filter
                    if exclude_countries:
                        where_clauses.append("c.country != ALL(%s::text[])")
                        params.append(exclude_countries)
                        self.logger.info(f"  Excluding countries: {exclude_countries}")
                    else:
                        self.logger.info(f"  No country exclusions (including all countries)")

                    # Date range filters
                    date_start = self.config["filters"]["date_range"].get("start")
                    date_end = self.config["filters"]["date_range"].get("end")
                    if date_start:
                        where_clauses.append("d.published_at >= %s")
                        params.append(date_start)
                    if date_end:
                        where_clauses.append("d.published_at <= %s")
                        params.append(date_end)

                    # Document ID range filters
                    if self.min_doc_id:
                        where_clauses.append("d.id >= %s")
                        params.append(self.min_doc_id)
                    if self.max_doc_id:
                        where_clauses.append("d.id <= %s")
                        params.append(self.max_doc_id)

                    where_sql = " AND ".join(where_clauses)

                    query = f"""
                        SELECT
                            d.id,
                            d.company_id,
                            d.document_type,
                            d.published_at as filing_date,
                            d.source_identifier as title,
                            d.created_at,
                            d.updated_at,
                            c.country,
                            c.ticker,
                            c.name as company_name
                        FROM librarian.kdocuments d
                        JOIN librarian.company c ON d.company_id = c.id
                        WHERE {where_sql}
                        ORDER BY d.id
                    """

                    self.logger.info(f"Executing query for {doc_type} with filters: min_doc_id={self.min_doc_id}, max_doc_id={self.max_doc_id}")
                    cur.execute(query, params)

                    documents = cur.fetchall()
                    doc_list = [dict(d) for d in documents]
                    
                    self.logger.info(f"Found {len(doc_list)} {doc_type} documents")
                    all_documents.extend(doc_list)

                # Track document types and countries
                for doc in all_documents:
                    doc_type_raw = doc.get("document_type", "")
                    doc_type = doc_type_raw.lower()
                    country = doc.get("country", "Unknown")

                    # Track by document type (raw value)
                    self.documents_by_type[doc_type_raw] += 1

                    # Track by country
                    self.documents_by_country[country] += 1

                    # Track by type AND country
                    self.documents_by_type_and_country[doc_type_raw][country] += 1

                    # Track companies by country
                    self.companies_by_country[country] += 1

                    # Legacy stats (for backward compatibility)
                    if "filing" in doc_type:
                        self.stats["filings_found"] += 1
                    if "slide" in doc_type:
                        self.stats["slides_found"] += 1

                self.stats["documents_found"] = len(all_documents)
                self.stats["companies_found"] = len(set(doc["company_id"] for doc in all_documents))
                
                self.logger.info(
                    f"Total documents found: {len(all_documents)} "
                    f"({self.stats['filings_found']} filings, {self.stats['slides_found']} slides)"
                )
                self.logger.info(f"Total unique companies: {self.stats['companies_found']}")

                return all_documents
        finally:
            conn.close()


    def _add_documents_to_metadata(self, documents: List[Dict]):
        """
        Add documents to metadata database.

        Args:
            documents: List of document dictionaries.
        """
        self.logger.info("Adding documents to metadata database...")

        for doc in documents:
            document_id = doc.get("id")
            company_id = doc.get("company_id")

            # Skip if both DOCLING and DOCLING_IMG representations already exist in Supabase
            if document_id in self.existing_representations:
                reps = self.existing_representations[document_id]
                if "DOCLING" in reps and "DOCLING_IMG" in reps:
                    self.stats["documents_skipped"] += 1
                    self.stats["documents_skipped_has_both_reps"] += 1
                    continue

            # Skip if already successfully converted (in processing log)
            if str(document_id) in self.completed_doc_ids:
                self.stats["documents_skipped"] += 1
                self.stats["documents_skipped_completed"] += 1
                continue

            # Check if we should skip
            skip_existing = self.config["download"]["skip_existing"]
            if skip_existing and self.metadata.document_exists(document_id):
                self.stats["documents_skipped"] += 1
                continue

            # Add to metadata
            added = self.metadata.add_document(
                document_id=document_id,
                company_id=company_id,
                ticker=doc.get("ticker", ""),
                company_name=doc.get("company_name", ""),
                country=doc.get("country", ""),
                document_type=doc.get("document_type", ""),
                filing_date=doc.get("filing_date"),
                title=doc.get("title", ""),
                pdf_url=None,  # URLs are fetched on-demand via batch download endpoint
            )

            if added:
                self.stats["documents_added"] += 1
            else:
                self.stats["documents_skipped"] += 1

        self.logger.info(
            f"Added {self.stats['documents_added']} new documents, skipped {self.stats['documents_skipped']}"
        )

    async def _get_download_urls_batch(
        self, session: aiohttp.ClientSession, document_ids: List[int]
    ) -> Dict[int, Optional[str]]:
        """
        Get download URLs for a batch of documents using batch download endpoint.

        Args:
            session: aiohttp session.
            document_ids: List of document IDs (max 250).

        Returns:
            Dictionary mapping document_id to download_url (or None if failed).
        """
        # Prepare batch request
        payload = {
            "documents": [
                {
                    "document_id": doc_id,
                    "representation_type": "raw",  # Use raw to get original PDF
                    "expires_in": 3600,  # URL valid for 1 hour
                }
                for doc_id in document_ids
            ]
        }

        response = await self._make_request(session, "kdocuments/batch/download", payload)

        if not response or "data" not in response:
            self.logger.error("Failed to get batch download URLs")
            return {doc_id: None for doc_id in document_ids}

        # Parse response
        results = response["data"].get("results", [])
        url_map = {}

        for result in results:
            doc_id = result.get("document_id")
            download_url = result.get("download_url")
            error = result.get("error")

            if error:
                self.logger.warning(f"Document {doc_id} download failed: {error}")
                url_map[doc_id] = None
            else:
                url_map[doc_id] = download_url

        return url_map

    async def _verify_pdf(self, pdf_path: Path) -> bool:
        """
        Verify that a downloaded file is actually a PDF.

        Args:
            pdf_path: Path to the file to verify.

        Returns:
            True if file is a valid PDF, False otherwise.
        """
        try:
            import subprocess

            # Use the 'file' command to check MIME type
            result = subprocess.run(
                ['file', '--mime-type', '-b', str(pdf_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            mime_type = result.stdout.strip()

            # Check if it's a PDF
            if mime_type.startswith('application/pdf'):
                return True

            # If not PDF, log what it actually is
            self.logger.warning(f"File {pdf_path.name} is {mime_type}, not application/pdf")
            return False

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout verifying file type for {pdf_path.name}")
            return False
        except Exception as e:
            self.logger.warning(f"Could not verify file type for {pdf_path.name}: {e}")
            # If verification fails, assume it's okay to be safe
            return True

    async def _download_pdf(
        self, session: aiohttp.ClientSession, document_id: int, pdf_url: str
    ) -> Optional[Path]:
        """
        Download a single PDF file from a presigned URL.

        Args:
            session: aiohttp session.
            document_id: Document ID.
            pdf_url: Presigned URL to download PDF from.

        Returns:
            Path to downloaded PDF file or None on failure.
        """
        if not pdf_url:
            return None

        try:
            # Download from presigned URL (no auth headers needed)
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    pdf_filename = f"doc_{document_id}.pdf"
                    pdf_path = self.input_folder / pdf_filename

                    async with aiofiles.open(pdf_path, "wb") as f:
                        await f.write(await response.read())

                    # Verify it's actually a PDF file
                    if not await self._verify_pdf(pdf_path):
                        self.logger.warning(
                            f"Downloaded file for doc {document_id} is not a valid PDF, deleting"
                        )
                        pdf_path.unlink()
                        self.stats["documents_rejected_not_pdf"] += 1
                        return None

                    return pdf_path
                else:
                    self.logger.error(
                        f"Failed to download PDF for doc {document_id}: HTTP {response.status}"
                    )
                    return None
        except Exception as e:
            self.logger.error(f"Failed to download PDF for doc {document_id}: {e}")
            return None

    async def _download_documents_batch(
        self, session: aiohttp.ClientSession, documents: List[Dict]
    ):
        """
        Download a batch of documents concurrently using batch download endpoint.

        Args:
            session: aiohttp session.
            documents: List of document metadata dicts.
        """
        if not documents:
            return

        # Process in batches of 250 (API limit)
        batch_size = 250
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            document_ids = [doc["document_id"] for doc in batch]

            self.logger.info(f"Getting download URLs for {len(document_ids)} documents...")

            # Step 1: Get download URLs for the batch
            url_map = await self._get_download_urls_batch(session, document_ids)

            # Step 2: Download PDFs concurrently
            semaphore = asyncio.Semaphore(self.config["download"]["concurrent_downloads"])

            async def download_one(doc: Dict):
                async with semaphore:
                    document_id = doc["document_id"]

                    # Check if PDF already exists
                    pdf_filename = f"doc_{document_id}.pdf"
                    pdf_path = self.input_folder / pdf_filename

                    if pdf_path.exists():
                        self.logger.info(f"Document {document_id} already exists, skipping download")
                        self.metadata.mark_downloaded(document_id, pdf_filename)
                        self.stats["documents_skipped"] += 1
                        return

                    pdf_url = url_map.get(document_id)

                    if not pdf_url:
                        self.logger.warning(
                            f"No download URL for document {document_id}, skipping"
                        )
                        self.metadata.mark_failed(
                            document_id, "No download URL available", "download_failed"
                        )
                        self.stats["documents_failed"] += 1
                        return

                    self.logger.info(f"Downloading document {document_id}...")
                    pdf_path = await self._download_pdf(session, document_id, pdf_url)

                    if pdf_path:
                        self.metadata.mark_downloaded(document_id, pdf_path.name)
                        self.stats["documents_downloaded"] += 1
                        self.logger.info(f"Successfully downloaded {pdf_path.name}")
                    else:
                        # Check if it was rejected as non-PDF (warning already logged)
                        # If download returned None, it either failed or was rejected
                        self.metadata.mark_failed(
                            document_id, "Download failed or file not a valid PDF", "download_failed"
                        )
                        self.stats["documents_failed"] += 1

                    # Rate limiting
                    await asyncio.sleep(self.config["download"]["rate_limit_delay"])

            # Download all documents in this batch concurrently
            await asyncio.gather(*[download_one(doc) for doc in batch])

    def _print_detailed_statistics(self):
        """Print detailed breakdown of documents by type and country."""
        self.logger.info("=" * 80)
        self.logger.info("DETAILED STATISTICS")
        self.logger.info("=" * 80)

        # Companies by country
        if self.companies_by_country:
            self.logger.info("\nCOMPANIES BY COUNTRY:")
            self.logger.info("-" * 40)
            for country, count in sorted(self.companies_by_country.items(), key=lambda x: (-x[1], x[0])):
                self.logger.info(f"  {country:30} {count:5,} companies")
            self.logger.info(f"  {'TOTAL':30} {sum(self.companies_by_country.values()):5,} companies")

        # Documents by type
        if self.documents_by_type:
            self.logger.info("\nDOCUMENTS BY TYPE:")
            self.logger.info("-" * 40)
            for doc_type, count in sorted(self.documents_by_type.items(), key=lambda x: (-x[1], x[0])):
                self.logger.info(f"  {doc_type:30} {count:6,} documents")
            self.logger.info(f"  {'TOTAL':30} {sum(self.documents_by_type.values()):6,} documents")

        # Documents by country
        if self.documents_by_country:
            self.logger.info("\nDOCUMENTS BY COUNTRY:")
            self.logger.info("-" * 40)
            for country, count in sorted(self.documents_by_country.items(), key=lambda x: (-x[1], x[0])):
                self.logger.info(f"  {country:30} {count:6,} documents")
            self.logger.info(f"  {'TOTAL':30} {sum(self.documents_by_country.values()):6,} documents")

        # Documents by type AND country (matrix view)
        if self.documents_by_type_and_country:
            self.logger.info("\nDOCUMENTS BY TYPE × COUNTRY:")
            self.logger.info("-" * 80)

            # Get all unique document types and countries
            all_types = sorted(self.documents_by_type_and_country.keys())
            all_countries = sorted(set(
                country
                for type_dict in self.documents_by_type_and_country.values()
                for country in type_dict.keys()
            ))

            # Print header
            header = f"  {'Type':20}"
            for country in all_countries:
                header += f" {country[:8]:>8}"
            header += f" {'TOTAL':>8}"
            self.logger.info(header)
            self.logger.info("  " + "-" * (20 + 9 * (len(all_countries) + 1)))

            # Print rows
            for doc_type in all_types:
                row = f"  {doc_type:20}"
                row_total = 0
                for country in all_countries:
                    count = self.documents_by_type_and_country[doc_type].get(country, 0)
                    row_total += count
                    if count > 0:
                        row += f" {count:8,}"
                    else:
                        row += f" {'-':>8}"
                row += f" {row_total:8,}"
                self.logger.info(row)

            # Print totals row
            totals_row = f"  {'TOTAL':20}"
            grand_total = 0
            for country in all_countries:
                country_total = sum(
                    self.documents_by_type_and_country[doc_type].get(country, 0)
                    for doc_type in all_types
                )
                grand_total += country_total
                totals_row += f" {country_total:8,}"
            totals_row += f" {grand_total:8,}"
            self.logger.info("  " + "-" * (20 + 9 * (len(all_countries) + 1)))
            self.logger.info(totals_row)

        # US vs Non-US comparison
        if hasattr(self, 'us_stats') and self.us_stats:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("US vs NON-US COMPARISON (in current ID range):")
            self.logger.info("=" * 80)

            non_us_total = sum(self.documents_by_country.values())
            us_total = self.us_stats.get("total_documents", 0)
            grand_total = non_us_total + us_total

            if grand_total > 0:
                non_us_pct = (non_us_total / grand_total) * 100
                us_pct = (us_total / grand_total) * 100

                self.logger.info(f"  {'Non-US Documents':30} {non_us_total:8,} ({non_us_pct:5.1f}%)")
                self.logger.info(f"  {'US Documents':30} {us_total:8,} ({us_pct:5.1f}%)")
                self.logger.info(f"  {'-' * 50}")
                self.logger.info(f"  {'TOTAL in ID range':30} {grand_total:8,} (100.0%)")

                self.logger.info(f"\n  {'Non-US Companies':30} {len(self.companies_by_country):8,}")
                self.logger.info(f"  {'US Companies':30} {self.us_stats.get('companies', 0):8,}")

        self.logger.info("=" * 80)

    async def fetch_all_images_mode(self, download_pdfs: bool = True):
        """
        Special mode to fetch documents that have DOCLING but not DOCLING_IMG representation.

        Args:
            download_pdfs: Whether to download PDFs (True) or just populate metadata (False).
        """
        self.logger.info("Starting RUN-ALL-IMAGES mode: fetching documents missing DOCLING_IMG...")

        # Step 1: Fetch document IDs missing DOCLING_IMG from Supabase
        doc_ids_missing_img = await self._fetch_documents_missing_docling_img()

        if not doc_ids_missing_img:
            self.logger.warning("No documents found missing DOCLING_IMG representation")
            return

        self.logger.info(f"Found {len(doc_ids_missing_img)} documents to process")

        # Apply limit if set (for batching)
        if self.limit and len(doc_ids_missing_img) > self.limit:
            self.logger.info(f"Limiting to {self.limit} documents (batch mode)")
            doc_ids_missing_img = doc_ids_missing_img[:self.limit]

        # Step 2: Fetch document metadata from kdocuments database
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Batch query for document metadata
                query = """
                    SELECT
                        d.id,
                        d.company_id,
                        d.document_type,
                        d.published_at as filing_date,
                        d.source_identifier as title,
                        d.created_at,
                        d.updated_at,
                        c.country,
                        c.ticker,
                        c.name as company_name
                    FROM librarian.kdocuments d
                    JOIN librarian.company c ON d.company_id = c.id
                    WHERE d.id = ANY(%s)
                    ORDER BY d.id
                """
                cur.execute(query, (doc_ids_missing_img,))
                documents = [dict(row) for row in cur.fetchall()]

                self.logger.info(f"Retrieved metadata for {len(documents)} documents")

        finally:
            conn.close()

        # Step 3: Add to metadata database (skip those already processed)
        self.logger.info("Adding documents to metadata database...")
        for doc in documents:
            document_id = doc.get("id")
            company_id = doc.get("company_id")

            # Skip if already successfully converted (in processing log)
            if str(document_id) in self.completed_doc_ids:
                self.stats["documents_skipped"] += 1
                self.stats["documents_skipped_completed"] += 1
                continue

            # Add to metadata
            added = self.metadata.add_document(
                document_id=document_id,
                company_id=company_id,
                ticker=doc.get("ticker", ""),
                company_name=doc.get("company_name", ""),
                country=doc.get("country", ""),
                document_type=doc.get("document_type", ""),
                filing_date=doc.get("filing_date"),
                title=doc.get("title", ""),
                pdf_url=None,
            )

            if added:
                self.stats["documents_added"] += 1
            else:
                self.stats["documents_skipped"] += 1

        self.logger.info(
            f"Added {self.stats['documents_added']} new documents, skipped {self.stats['documents_skipped']}"
        )

        # Step 4: Download PDFs
        if download_pdfs:
            pending = self.metadata.get_pending_downloads()
            self.logger.info(f"Downloading {len(pending)} pending PDFs...")

            if pending:
                async with aiohttp.ClientSession() as session:
                    await self._download_documents_batch(session, pending)

        # Print final statistics
        self.logger.info("=" * 60)
        self.logger.info("RUN-ALL-IMAGES Mode Summary:")
        self.logger.info(f"  Documents missing DOCLING_IMG: {len(doc_ids_missing_img)}")
        self.logger.info(f"  Documents added to DB: {self.stats['documents_added']}")
        self.logger.info(f"  Documents skipped: {self.stats['documents_skipped']}")
        if self.stats['documents_skipped_completed'] > 0:
            self.logger.info(f"    - Already completed (in processing log): {self.stats['documents_skipped_completed']}")
        if download_pdfs:
            self.logger.info(f"  PDFs downloaded: {self.stats['documents_downloaded']}")
            self.logger.info(f"  Downloads failed: {self.stats['documents_failed']}")
            if self.stats['documents_rejected_not_pdf'] > 0:
                self.logger.info(f"    - Rejected (not valid PDF): {self.stats['documents_rejected_not_pdf']}")
        self.logger.info("=" * 60)

    async def fetch_all(self, download_pdfs: bool = True):
        """
        Main entry point: fetch documents from database with type-specific country filters.

        Args:
            download_pdfs: Whether to download PDFs (True) or just populate metadata (False).
        """
        # Check if we're in run-all-images mode
        if self.run_all_images:
            await self.fetch_all_images_mode(download_pdfs)
            return

        self.logger.info("Starting document fetch process with type-specific country filters...")

        # Step 0a: Check US documents (for reference)
        self.us_stats = self.check_us_documents()
        self.logger.info("")  # Blank line for readability

        # Step 0b: Check what document types are available
        available_types = self.check_available_document_types()

        # Step 1: Fetch documents from database with type-specific filters
        documents = self.fetch_documents_by_type_from_db()

        if not documents:
            self.logger.warning("No documents found")
            return

        # Step 2: Check Supabase for existing representations
        document_ids = [doc["id"] for doc in documents]
        self.existing_representations = await self._fetch_existing_representations_from_supabase(
            document_ids
        )

        # Randomize document order if requested (for diverse sampling)
        if self.randomize:
            self.logger.info(f"Randomizing {len(documents)} documents with seed {self.random_seed}")
            random.shuffle(documents)

            # If limit is set, take first N documents (now randomized)
            if self.limit and len(documents) > self.limit:
                documents = documents[:self.limit]
                self.logger.info(f"Selected {self.limit} random documents from total pool")
        elif self.limit and len(documents) > self.limit:
            # Apply limit without randomization
            documents = documents[:self.limit]
            self.logger.info(f"Limited to first {self.limit} documents")

        # Step 3: Add to metadata database
        self._add_documents_to_metadata(documents)

        # Print detailed statistics breakdown (before downloads start)
        self._print_detailed_statistics()

        # Step 4: Download PDFs (optional)
        if download_pdfs:
            pending = self.metadata.get_pending_downloads()
            self.logger.info(f"Downloading {len(pending)} pending PDFs...")

            if pending:
                async with aiohttp.ClientSession() as session:
                    await self._download_documents_batch(session, pending)

        # Print final statistics
        self.logger.info("=" * 60)
        self.logger.info("Fetch Summary:")
        self.logger.info(f"  Companies found: {self.stats['companies_found']}")
        self.logger.info(f"  Documents found: {self.stats['documents_found']}")
        self.logger.info(f"    - Filings: {self.stats['filings_found']}")
        self.logger.info(f"    - Slides: {self.stats['slides_found']}")
        if self.min_doc_id or self.max_doc_id:
            self.logger.info(f"  ID range filter: {self.min_doc_id or 'MIN'} to {self.max_doc_id or 'MAX'}")
        self.logger.info(f"  Documents added to DB: {self.stats['documents_added']}")
        self.logger.info(f"  Documents skipped: {self.stats['documents_skipped']}")
        if self.stats['documents_skipped_has_both_reps'] > 0:
            self.logger.info(f"    - Already has both representations in Supabase: {self.stats['documents_skipped_has_both_reps']}")
        if self.stats['documents_skipped_completed'] > 0:
            self.logger.info(f"    - Already completed (in processing log): {self.stats['documents_skipped_completed']}")
        if download_pdfs:
            self.logger.info(f"  PDFs downloaded: {self.stats['documents_downloaded']}")
            self.logger.info(f"  Downloads failed: {self.stats['documents_failed']}")
            if self.stats['documents_rejected_not_pdf'] > 0:
                self.logger.info(f"    - Rejected (not valid PDF): {self.stats['documents_rejected_not_pdf']}")
        self.logger.info("=" * 60)

    async def download_pending(self):
        """Download all pending PDFs from metadata database."""
        pending = self.metadata.get_pending_downloads()

        if not pending:
            self.logger.info("No pending downloads")
            return

        self.logger.info(f"Downloading {len(pending)} pending PDFs...")

        async with aiohttp.ClientSession() as session:
            await self._download_documents_batch(session, pending)

        self.logger.info(f"Downloaded {self.stats['documents_downloaded']} PDFs")


async def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch non-US company documents from Librarian API"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch metadata, don't download PDFs",
    )
    parser.add_argument(
        "--download-pending",
        action="store_true",
        help="Download pending PDFs from metadata database",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show metadata database statistics"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of documents to fetch (e.g., 1000 for test batch)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize document order for diverse sampling (useful with --limit)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--min-doc-id",
        type=int,
        help="Minimum document ID to fetch (filter out documents with ID < this value)",
    )
    parser.add_argument(
        "--max-doc-id",
        type=int,
        help="Maximum document ID to fetch (filter out documents with ID > this value)",
    )
    parser.add_argument(
        "--run-all-images",
        action="store_true",
        help="Fetch documents that have DOCLING but not DOCLING_IMG representation (ignores min/max doc ID)",
    )

    args = parser.parse_args()

    fetcher = DocumentFetcher(
        args.config,
        limit=args.limit,
        randomize=args.randomize,
        random_seed=args.random_seed,
        min_doc_id=args.min_doc_id,
        max_doc_id=args.max_doc_id,
        run_all_images=args.run_all_images
    )

    if args.stats:
        stats = fetcher.metadata.get_statistics()
        print("\nDocument Statistics:")
        for status, count in stats.items():
            print(f"  {status}: {count}")
        return

    if args.download_pending:
        await fetcher.download_pending()
    else:
        await fetcher.fetch_all(download_pdfs=not args.metadata_only)


if __name__ == "__main__":
    asyncio.run(main())
