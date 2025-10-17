"""
Document Fetcher

Fetches non-US company filings and slides from the Librarian API and saves PDFs to to_process/ folder.
"""

import asyncio
import aiohttp
import aiofiles
import os
import yaml
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from document_metadata import MetadataManager

# Load environment variables from .env file
load_dotenv()


class DocumentFetcher:
    """Fetches documents from the Librarian API."""

    def __init__(self, config_path: str = "config.yaml", limit: Optional[int] = None, randomize: bool = False, random_seed: int = 42, max_doc_id: Optional[int] = None):
        """
        Initialize document fetcher.

        Args:
            config_path: Path to configuration file.
            limit: Maximum number of documents to fetch (None = no limit).
            randomize: Whether to randomize document order for sampling.
            random_seed: Random seed for reproducible sampling (default: 42).
            max_doc_id: Maximum document ID to fetch (filter out documents with ID > this value).
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
        self.max_doc_id = max_doc_id

        if self.max_doc_id:
            self.logger.info(f"Will filter out documents with ID > {self.max_doc_id}")

        if self.randomize:
            random.seed(self.random_seed)
            self.logger.info(f"Randomization enabled with seed {self.random_seed}")

        # Statistics
        self.stats = {
            "companies_found": 0,
            "documents_found": 0,
            "documents_added": 0,
            "documents_skipped": 0,
            "documents_downloaded": 0,
            "documents_failed": 0,
            "filings_found": 0,
            "slides_found": 0,
        }

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

    async def fetch_companies(self, session: aiohttp.ClientSession) -> List[Dict]:
        """
        Fetch all non-US companies with pagination.

        Args:
            session: aiohttp session.

        Returns:
            List of company dictionaries.
        """
        self.logger.info("Fetching non-US companies...")
        all_companies = []
        page = 1
        page_size = self.config["download"]["company_page_size"]

        while True:
            payload = {
                "country": self.config["filters"]["countries"],
                "page": page,
                "page_size": page_size,
            }

            self.logger.info(f"Fetching companies page {page}...")
            response = await self._make_request(session, "companies/filter", payload)

            if not response:
                self.logger.error(f"Failed to fetch companies page {page}")
                break

            # Handle response structure: {"message": "...", "data": [...]}
            if isinstance(response, dict) and "data" in response:
                companies = response["data"] if isinstance(response["data"], list) else []
                total = len(companies)  # API returns full page, not total count
            else:
                self.logger.error(f"Unexpected response structure: {response}")
                break

            if not companies:
                break

            all_companies.extend(companies)
            self.logger.info(f"Found {len(companies)} companies on page {page}")

            # Check if there are more pages
            if len(all_companies) >= total:
                break

            page += 1

            # Rate limiting
            await asyncio.sleep(self.config["download"]["rate_limit_delay"])

        self.stats["companies_found"] = len(all_companies)
        self.logger.info(f"Total companies found: {len(all_companies)}")
        return all_companies

    async def fetch_documents(
        self, session: aiohttp.ClientSession, company_ids: List[int]
    ) -> List[Dict]:
        """
        Fetch documents for given company IDs with pagination.

        Args:
            session: aiohttp session.
            company_ids: List of company IDs to fetch documents for.

        Returns:
            List of document dictionaries.
        """
        self.logger.info(f"Fetching documents for {len(company_ids)} companies...")
        all_documents = []
        page = 1
        page_size = self.config["download"]["document_page_size"]

        while True:
            # When randomizing, fetch more documents than limit to ensure diverse sample
            # Don't stop early - we'll randomize and sample after fetching all
            if self.limit and not self.randomize and len(all_documents) >= self.limit:
                self.logger.info(f"Reached document limit of {self.limit}, stopping fetch")
                break

            payload = {
                "company_id": company_ids,
                "document_type": self.config["filters"]["document_types"],
                "page": page,
                "page_size": page_size,
            }

            # Add date range filters if specified
            date_start = self.config["filters"]["date_range"].get("start")
            date_end = self.config["filters"]["date_range"].get("end")
            if date_start:
                payload["filing_date_start"] = date_start
            if date_end:
                payload["filing_date_end"] = date_end

            self.logger.info(f"Fetching documents page {page}...")
            response = await self._make_request(session, "kdocuments/search", payload)

            if not response:
                self.logger.error(f"Failed to fetch documents page {page}")
                break

            # Handle response structure: {"message": "...", "data": [...]}
            if isinstance(response, dict) and "data" in response:
                documents = response["data"] if isinstance(response["data"], list) else []
                total = len(documents)  # API returns full page, not total count
            else:
                self.logger.error(f"Unexpected response structure for documents")
                break

            if not documents:
                break

            # Apply limit if specified (only when not randomizing)
            if self.limit and not self.randomize:
                remaining = self.limit - len(all_documents)
                if len(documents) > remaining:
                    documents = documents[:remaining]

            all_documents.extend(documents)

            # Track document types
            for doc in documents:
                doc_type = doc.get("document_type", "").lower()
                if "filing" in doc_type:
                    self.stats["filings_found"] += 1
                elif "slide" in doc_type:
                    self.stats["slides_found"] += 1

            self.logger.info(
                f"Found {len(documents)} documents on page {page} "
                f"(total: {len(all_documents)}, filings: {self.stats['filings_found']}, "
                f"slides: {self.stats['slides_found']})"
            )

            # Check if we've reached limit or end of results
            if self.limit and not self.randomize and len(all_documents) >= self.limit:
                break
            if len(documents) < page_size:
                break

            page += 1

            # Rate limiting
            await asyncio.sleep(self.config["download"]["rate_limit_delay"])

        self.stats["documents_found"] = len(all_documents)
        self.logger.info(
            f"Total documents found: {len(all_documents)} "
            f"({self.stats['filings_found']} filings, {self.stats['slides_found']} slides)"
        )
        return all_documents

    def _add_documents_to_metadata(
        self, documents: List[Dict], company_lookup: Dict[int, Dict]
    ):
        """
        Add documents to metadata database.

        Args:
            documents: List of document dictionaries.
            company_lookup: Dictionary mapping company_id to company info.
        """
        self.logger.info("Adding documents to metadata database...")

        for doc in documents:
            document_id = doc.get("id")
            company_id = doc.get("company_id")
            company = company_lookup.get(company_id, {})

            # Check if we should skip
            skip_existing = self.config["download"]["skip_existing"]
            if skip_existing and self.metadata.document_exists(document_id):
                self.stats["documents_skipped"] += 1
                continue

            # Add to metadata
            added = self.metadata.add_document(
                document_id=document_id,
                company_id=company_id,
                ticker=doc.get("ticker") or company.get("ticker", ""),
                company_name=company.get("name", ""),
                country=company.get("country", ""),
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
                        self.metadata.mark_failed(
                            document_id, "Download failed", "download_failed"
                        )
                        self.stats["documents_failed"] += 1

                    # Rate limiting
                    await asyncio.sleep(self.config["download"]["rate_limit_delay"])

            # Download all documents in this batch concurrently
            await asyncio.gather(*[download_one(doc) for doc in batch])

    async def fetch_all(self, download_pdfs: bool = True):
        """
        Main entry point: fetch all non-US documents.

        Args:
            download_pdfs: Whether to download PDFs (True) or just populate metadata (False).
        """
        self.logger.info("Starting document fetch process...")

        async with aiohttp.ClientSession() as session:
            # Step 1: Fetch companies
            companies = await self.fetch_companies(session)

            if not companies:
                self.logger.error("No companies found, aborting")
                return

            # Create company lookup
            company_lookup = {c["id"]: c for c in companies}

            # Step 2: Fetch documents
            company_ids = [c["id"] for c in companies]
            documents = await self.fetch_documents(session, company_ids)

            if not documents:
                self.logger.warning("No documents found")
                return

            # Filter by max_doc_id if specified
            if self.max_doc_id:
                original_count = len(documents)
                documents = [d for d in documents if d.get("id", 0) <= self.max_doc_id]
                filtered_count = original_count - len(documents)
                if filtered_count > 0:
                    self.logger.info(f"Filtered out {filtered_count} documents with ID > {self.max_doc_id}")

            # Randomize document order if requested (for diverse sampling)
            if self.randomize:
                self.logger.info(f"Randomizing {len(documents)} documents with seed {self.random_seed}")
                random.shuffle(documents)

                # If limit is set, take first N documents (now randomized)
                if self.limit and len(documents) > self.limit:
                    documents = documents[:self.limit]
                    self.logger.info(f"Selected {self.limit} random documents from total pool")

            # Step 3: Add to metadata database
            self._add_documents_to_metadata(documents, company_lookup)

            # Step 4: Download PDFs (optional)
            if download_pdfs:
                pending = self.metadata.get_pending_downloads()
                self.logger.info(f"Downloading {len(pending)} pending PDFs...")

                if pending:
                    await self._download_documents_batch(session, pending)

        # Print final statistics
        self.logger.info("=" * 60)
        self.logger.info("Fetch Summary:")
        self.logger.info(f"  Companies found: {self.stats['companies_found']}")
        self.logger.info(f"  Documents found: {self.stats['documents_found']}")
        self.logger.info(f"    - Filings: {self.stats['filings_found']}")
        self.logger.info(f"    - Slides: {self.stats['slides_found']}")
        self.logger.info(f"  Documents added to DB: {self.stats['documents_added']}")
        self.logger.info(f"  Documents skipped: {self.stats['documents_skipped']}")
        if download_pdfs:
            self.logger.info(f"  PDFs downloaded: {self.stats['documents_downloaded']}")
            self.logger.info(f"  Downloads failed: {self.stats['documents_failed']}")
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
        "--max-doc-id",
        type=int,
        help="Maximum document ID to fetch (filter out documents with ID > this value)",
    )

    args = parser.parse_args()

    fetcher = DocumentFetcher(
        args.config,
        limit=args.limit,
        randomize=args.randomize,
        random_seed=args.random_seed,
        max_doc_id=args.max_doc_id
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
