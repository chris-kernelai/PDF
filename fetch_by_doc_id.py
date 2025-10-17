#!/usr/bin/env python3
"""
Fetch documents by document ID range from database, then download PDFs via API.

Queries the database directly to get document IDs in a specified range,
then uses the API to download the actual PDFs.
"""

import os
import asyncio
import aiohttp
import aiofiles
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Optional
import logging
from document_metadata import MetadataManager

load_dotenv()


class DocumentFetcherByID:
    """Fetches documents by ID range from database, downloads PDFs via API."""

    def __init__(
        self,
        min_doc_id: int = 0,
        max_doc_id: int = 1000000,
        exclude_us: bool = True,
        include_filings: bool = True,
        include_slides: bool = True,
        output_folder: str = "to_process",
        metadata_db: str = "to_process/metadata.db",
    ):
        self.min_doc_id = min_doc_id
        self.max_doc_id = max_doc_id
        self.exclude_us = exclude_us
        self.include_filings = include_filings
        self.include_slides = include_slides
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("fetch_by_doc_id.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Metadata manager
        self.metadata = MetadataManager(metadata_db)

        # API credentials
        self.api_key = os.getenv("API_KEY")
        self.api_base_url = "https://librarian.production.primerapp.com/api/v1"

        # Stats
        self.stats = {
            "docs_found_in_db": 0,
            "docs_already_downloaded": 0,
            "docs_to_download": 0,
            "docs_downloaded": 0,
            "docs_failed": 0,
        }

    def get_db_connection(self):
        """Create read-only database connection."""
        return psycopg2.connect(
            host=os.getenv("K_LIB_DB_HOST"),
            port=os.getenv("K_LIB_DB_PORT"),
            user=os.getenv("K_LIB_DB_USER"),
            password=os.getenv("K_LIB_DB_PASSWORD"),
            database=os.getenv("K_LIB_DB_NAME"),
            options="-c default_transaction_read_only=on",
        )

    def fetch_document_ids(self) -> List[Dict]:
        """Fetch document IDs from database based on filters."""
        self.logger.info("Connecting to database...")
        conn = self.get_db_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build document type filter
                doc_types = []
                if self.include_filings:
                    doc_types.append("FILING")
                if self.include_slides:
                    doc_types.append("SLIDE")

                if not doc_types:
                    self.logger.error("No document types selected!")
                    return []

                # Build country filter
                country_filter = ""
                if self.exclude_us:
                    # Get non-US company IDs
                    cur.execute("""
                        SELECT id FROM librarian.company
                        WHERE country != 'United States'
                    """)
                    non_us_company_ids = [row["id"] for row in cur.fetchall()]

                    if not non_us_company_ids:
                        self.logger.error("No non-US companies found!")
                        return []

                    country_filter = "AND company_id = ANY(%s::int[])"
                    params = (non_us_company_ids, doc_types, self.min_doc_id, self.max_doc_id)
                else:
                    params = (doc_types, self.min_doc_id, self.max_doc_id)

                # Build document type filter
                doc_type_placeholders = ', '.join(['%s'] * len(doc_types))

                # Fetch documents
                query = f"""
                    SELECT
                        d.id,
                        d.company_id,
                        d.document_type,
                        d.published_at,
                        c.ticker,
                        c.name as company_name,
                        c.country
                    FROM librarian.kdocuments d
                    JOIN librarian.company c ON d.company_id = c.id
                    WHERE d.document_type IN ({doc_type_placeholders})
                    {country_filter}
                    AND d.id >= %s
                    AND d.id <= %s
                    ORDER BY d.id
                """

                self.logger.info(f"Querying documents with ID range [{self.min_doc_id}, {self.max_doc_id}]")
                self.logger.info(f"Document types: {', '.join(doc_types)}")
                self.logger.info(f"Exclude US: {self.exclude_us}")

                if self.exclude_us:
                    cur.execute(query, tuple(doc_types) + (non_us_company_ids, self.min_doc_id, self.max_doc_id))
                else:
                    # Adjust params order when no country filter
                    cur.execute(query, tuple(doc_types) + (self.min_doc_id, self.max_doc_id))

                documents = cur.fetchall()
                self.stats["docs_found_in_db"] = len(documents)

                self.logger.info(f"Found {len(documents):,} documents in database")

                return documents

        finally:
            conn.close()

    async def _get_download_urls_batch(
        self, session: aiohttp.ClientSession, document_ids: List[int]
    ) -> Dict[int, Optional[str]]:
        """Get download URLs for a batch of documents."""
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

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with session.post(
                f"{self.api_base_url}/kdocuments/batch/download",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to get download URLs: HTTP {response.status}")
                    return {doc_id: None for doc_id in document_ids}

                data = await response.json()
                results = data.get("data", {}).get("results", [])

                url_map = {}
                for result in results:
                    doc_id = result.get("document_id")
                    download_url = result.get("download_url")
                    error = result.get("error")

                    if error:
                        self.logger.warning(f"Document {doc_id} error: {error}")
                        url_map[doc_id] = None
                    else:
                        url_map[doc_id] = download_url

                return url_map

        except Exception as e:
            self.logger.error(f"Error getting download URLs: {e}")
            return {doc_id: None for doc_id in document_ids}

    async def _download_pdf(
        self, session: aiohttp.ClientSession, document_id: int, pdf_url: str
    ) -> Optional[Path]:
        """Download a single PDF file."""
        if not pdf_url:
            return None

        try:
            async with session.get(pdf_url) as response:
                if response.status == 200:
                    pdf_filename = f"doc_{document_id}.pdf"
                    pdf_path = self.output_folder / pdf_filename

                    async with aiofiles.open(pdf_path, "wb") as f:
                        await f.write(await response.read())

                    return pdf_path
                else:
                    self.logger.error(f"Failed to download PDF {document_id}: HTTP {response.status}")
                    return None

        except Exception as e:
            self.logger.error(f"Error downloading PDF {document_id}: {e}")
            return None

    async def download_documents(self, documents: List[Dict]):
        """Download PDFs for the given documents."""
        if not documents:
            self.logger.info("No documents to download")
            return

        # Filter out already downloaded
        docs_to_download = []
        for doc in documents:
            doc_id = doc["id"]
            pdf_path = self.output_folder / f"doc_{doc_id}.pdf"

            if pdf_path.exists():
                self.logger.debug(f"Document {doc_id} already downloaded, skipping")
                self.stats["docs_already_downloaded"] += 1
                # Make sure it's in metadata
                if not self.metadata.document_exists(doc_id):
                    self.metadata.add_document(
                        document_id=doc_id,
                        company_id=doc["company_id"],
                        ticker=doc["ticker"],
                        company_name=doc["company_name"],
                        country=doc["country"],
                        document_type=doc["document_type"],
                        filing_date=doc["published_at"],
                        title="",
                        pdf_url=None,
                    )
                    self.metadata.mark_downloaded(doc_id, pdf_path.name)
            else:
                docs_to_download.append(doc)

        self.stats["docs_to_download"] = len(docs_to_download)
        self.logger.info(f"Documents to download: {len(docs_to_download):,}")
        self.logger.info(f"Already downloaded: {self.stats['docs_already_downloaded']:,}")

        if not docs_to_download:
            return

        # Download in batches of 250 (API limit)
        batch_size = 250
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(docs_to_download), batch_size):
                batch = docs_to_download[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(docs_to_download) + batch_size - 1) // batch_size

                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)"
                )

                # Step 1: Get download URLs
                document_ids = [doc["id"] for doc in batch]
                url_map = await self._get_download_urls_batch(session, document_ids)

                # Step 2: Download PDFs
                semaphore = asyncio.Semaphore(5)  # 5 concurrent downloads

                async def download_one(doc: Dict):
                    async with semaphore:
                        doc_id = doc["id"]
                        pdf_url = url_map.get(doc_id)

                        # Add to metadata first
                        if not self.metadata.document_exists(doc_id):
                            self.metadata.add_document(
                                document_id=doc_id,
                                company_id=doc["company_id"],
                                ticker=doc["ticker"],
                                company_name=doc["company_name"],
                                country=doc["country"],
                                document_type=doc["document_type"],
                                filing_date=doc["published_at"],
                                title="",
                                pdf_url=None,
                            )

                        if not pdf_url:
                            self.logger.warning(f"No download URL for document {doc_id}")
                            self.metadata.mark_failed(doc_id, "No download URL", "download_failed")
                            self.stats["docs_failed"] += 1
                            return

                        pdf_path = await self._download_pdf(session, doc_id, pdf_url)

                        if pdf_path:
                            self.metadata.mark_downloaded(doc_id, pdf_path.name)
                            self.stats["docs_downloaded"] += 1
                            self.logger.info(f"Downloaded {pdf_path.name}")
                        else:
                            self.metadata.mark_failed(doc_id, "Download failed", "download_failed")
                            self.stats["docs_failed"] += 1

                        # Rate limiting
                        await asyncio.sleep(0.5)

                await asyncio.gather(*[download_one(doc) for doc in batch])

    async def run(self):
        """Main entry point."""
        self.logger.info("=" * 80)
        self.logger.info("DOCUMENT FETCHER BY ID RANGE")
        self.logger.info("=" * 80)
        self.logger.info(f"ID range: [{self.min_doc_id:,}, {self.max_doc_id:,}]")
        self.logger.info(f"Exclude US: {self.exclude_us}")
        self.logger.info(f"Include filings: {self.include_filings}")
        self.logger.info(f"Include slides: {self.include_slides}")
        self.logger.info("=" * 80)

        # Step 1: Query database for document IDs
        documents = self.fetch_document_ids()

        if not documents:
            self.logger.warning("No documents found matching criteria")
            return

        # Step 2: Download PDFs
        await self.download_documents(documents)

        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Documents found in DB:     {self.stats['docs_found_in_db']:,}")
        self.logger.info(f"Already downloaded:        {self.stats['docs_already_downloaded']:,}")
        self.logger.info(f"New downloads needed:      {self.stats['docs_to_download']:,}")
        self.logger.info(f"Successfully downloaded:   {self.stats['docs_downloaded']:,}")
        self.logger.info(f"Failed downloads:          {self.stats['docs_failed']:,}")
        self.logger.info("=" * 80)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch documents by ID range and download PDFs"
    )
    parser.add_argument(
        "--min-doc-id",
        type=int,
        default=0,
        help="Minimum document ID (default: 0)",
    )
    parser.add_argument(
        "--max-doc-id",
        type=int,
        default=1000000,
        help="Maximum document ID (default: 1000000)",
    )
    parser.add_argument(
        "--include-us",
        action="store_true",
        help="Include US companies (default: exclude US)",
    )
    parser.add_argument(
        "--no-filings",
        action="store_true",
        help="Exclude filings (default: include filings)",
    )
    parser.add_argument(
        "--no-slides",
        action="store_true",
        help="Exclude slides (default: include slides)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="to_process",
        help="Output folder for PDFs (default: to_process)",
    )
    parser.add_argument(
        "--metadata-db",
        type=str,
        default="to_process/metadata.db",
        help="Metadata database path (default: to_process/metadata.db)",
    )

    args = parser.parse_args()

    fetcher = DocumentFetcherByID(
        min_doc_id=args.min_doc_id,
        max_doc_id=args.max_doc_id,
        exclude_us=not args.include_us,
        include_filings=not args.no_filings,
        include_slides=not args.no_slides,
        output_folder=args.output_folder,
        metadata_db=args.metadata_db,
    )

    await fetcher.run()


if __name__ == "__main__":
    asyncio.run(main())
