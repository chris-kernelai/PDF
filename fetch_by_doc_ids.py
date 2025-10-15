#!/usr/bin/env python3
"""
Fetch documents by specific document IDs

Fetches PDFs from the Librarian API by document ID, skipping any that don't exist.

Usage:
    python fetch_by_doc_ids.py 12345 12346 12347
    python fetch_by_doc_ids.py --file doc_ids.txt
"""

import asyncio
import aiohttp
import aiofiles
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "https://api.example.com")  # Update with your API URL
API_KEY = os.getenv("API_KEY")
OUTPUT_DIR = Path("to_process")
TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 250  # API batch limit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def get_download_urls_batch(
    session: aiohttp.ClientSession, document_ids: List[int]
) -> dict:
    """
    Get download URLs for a batch of documents.

    Args:
        session: aiohttp session
        document_ids: List of document IDs (max 250)

    Returns:
        Dictionary mapping document_id to download_url (or None if failed)
    """
    if not document_ids:
        return {}

    url = f"{BASE_URL}/kdocuments/batch/download"
    headers = {"Authorization": f"Bearer {API_KEY}"}

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

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("data", {}).get("results", [])

                    url_map = {}
                    for result in results:
                        doc_id = result.get("document_id")
                        download_url = result.get("download_url")
                        error = result.get("error")

                        if error:
                            logger.warning(f"Document {doc_id}: {error}")
                            url_map[doc_id] = None
                        else:
                            url_map[doc_id] = download_url

                    return url_map
                elif response.status == 429:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"API error: HTTP {response.status}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)

        except asyncio.TimeoutError:
            logger.error(f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)

    return {doc_id: None for doc_id in document_ids}


async def download_pdf(
    session: aiohttp.ClientSession, document_id: int, pdf_url: str
) -> Optional[Path]:
    """
    Download a PDF from presigned URL.

    Args:
        session: aiohttp session
        document_id: Document ID
        pdf_url: Presigned download URL

    Returns:
        Path to downloaded file or None on failure
    """
    if not pdf_url:
        return None

    try:
        async with session.get(pdf_url) as response:
            if response.status == 200:
                pdf_filename = f"doc_{document_id}.pdf"
                pdf_path = OUTPUT_DIR / pdf_filename

                async with aiofiles.open(pdf_path, "wb") as f:
                    await f.write(await response.read())

                logger.info(f"✅ Downloaded: {pdf_filename}")
                return pdf_path
            else:
                logger.error(f"❌ Failed to download doc {document_id}: HTTP {response.status}")
                return None
    except Exception as e:
        logger.error(f"❌ Failed to download doc {document_id}: {e}")
        return None


async def fetch_documents_by_ids(document_ids: List[int]):
    """
    Fetch PDFs for given document IDs.

    Args:
        document_ids: List of document IDs to fetch
    """
    if not API_KEY:
        logger.error("API_KEY environment variable not set")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {len(document_ids)} documents...")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    stats = {
        "total": len(document_ids),
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
    }

    async with aiohttp.ClientSession() as session:
        # Process in batches of 250
        for i in range(0, len(document_ids), BATCH_SIZE):
            batch = document_ids[i : i + BATCH_SIZE]

            logger.info(f"\nProcessing batch {i // BATCH_SIZE + 1} ({len(batch)} documents)...")

            # Get download URLs
            url_map = await get_download_urls_batch(session, batch)

            # Download PDFs
            for doc_id in batch:
                pdf_url = url_map.get(doc_id)

                if not pdf_url:
                    logger.warning(f"⏭️  Skipped doc {doc_id}: No download URL (document may not exist)")
                    stats["skipped"] += 1
                    continue

                pdf_path = await download_pdf(session, doc_id, pdf_url)

                if pdf_path:
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1

                # Rate limiting
                await asyncio.sleep(0.1)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"📄 Total requested: {stats['total']}")
    print(f"✅ Downloaded: {stats['downloaded']}")
    print(f"⏭️  Skipped (not found): {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


def load_doc_ids_from_file(file_path: str) -> List[int]:
    """Load document IDs from file (one per line)."""
    doc_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    doc_ids.append(int(line))
                except ValueError:
                    logger.warning(f"Invalid document ID: {line}")
    return doc_ids


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch documents by document IDs"
    )
    parser.add_argument(
        "doc_ids",
        nargs="*",
        type=int,
        help="Document IDs to fetch",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing document IDs (one per line)",
    )
    parser.add_argument(
        "--range",
        type=str,
        help="Range of document IDs (e.g., '12345-12350' or '12345:12350')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Update output directory
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    # Get document IDs
    doc_ids = []

    if args.file:
        logger.info(f"Loading document IDs from {args.file}...")
        doc_ids = load_doc_ids_from_file(args.file)
    elif args.range:
        # Parse range
        try:
            if '-' in args.range:
                start, end = args.range.split('-')
            elif ':' in args.range:
                start, end = args.range.split(':')
            else:
                logger.error("Invalid range format. Use 'start-end' or 'start:end'")
                sys.exit(1)

            start_id = int(start)
            end_id = int(end)

            if start_id > end_id:
                logger.error("Start ID must be less than or equal to end ID")
                sys.exit(1)

            doc_ids = list(range(start_id, end_id + 1))
            logger.info(f"Generated range: {start_id} to {end_id} ({len(doc_ids)} documents)")
        except ValueError as e:
            logger.error(f"Invalid range format: {e}")
            sys.exit(1)
    elif args.doc_ids:
        doc_ids = args.doc_ids
    else:
        parser.print_help()
        sys.exit(1)

    if not doc_ids:
        logger.error("No document IDs provided")
        sys.exit(1)

    logger.info(f"Loaded {len(doc_ids)} document IDs")

    # Fetch documents
    asyncio.run(fetch_documents_by_ids(doc_ids))


if __name__ == "__main__":
    main()
