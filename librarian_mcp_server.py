#!/usr/bin/env python3
"""
MCP Server for Librarian API

Exposes Librarian API operations as MCP tools and resources.
Designed to work with Claude Desktop and other MCP clients.
"""

import os
import logging
from typing import Any, Optional
from pathlib import Path
import httpx
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Librarian API Server", dependencies=["httpx"])

# Configuration
LIBRARIAN_BASE_URL = "https://librarian.production.primerapp.com/api/v1"
API_KEY = os.getenv("LIBRARIAN_API_KEY", "8fc83f71-23e1-40bd-93a4-4e0a47cdcb44")

# Paths
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


class LibrarianAPIClient:
    """Client for interacting with Librarian API"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def get_companies(
        self,
        countries: Optional[list[str]] = None,
        page: int = 1,
        page_size: int = 100
    ) -> dict[str, Any]:
        """Fetch companies filtered by country"""
        url = f"{self.base_url}/companies/filter"
        payload = {
            "page": page,
            "page_size": page_size
        }
        if countries:
            payload["country"] = countries

        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"Fetching companies: {payload}")
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def search_documents(
        self,
        company_ids: Optional[list[int]] = None,
        document_types: Optional[list[str]] = None,
        filing_date_start: Optional[str] = None,
        filing_date_end: Optional[str] = None,
        page: int = 1,
        page_size: int = 250
    ) -> dict[str, Any]:
        """Search for documents by various criteria"""
        url = f"{self.base_url}/kdocuments/search"
        payload = {
            "page": page,
            "page_size": page_size
        }
        if company_ids:
            payload["company_id"] = company_ids
        if document_types:
            payload["document_type"] = document_types
        if filing_date_start:
            payload["filing_date_start"] = filing_date_start
        if filing_date_end:
            payload["filing_date_end"] = filing_date_end

        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"Searching documents: {payload}")
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def batch_download(
        self,
        document_ids: list[int],
        representation_type: str = "raw"
    ) -> dict[str, Any]:
        """Request batch download URLs for documents"""
        url = f"{self.base_url}/kdocuments/batch/download"

        # Build documents payload
        documents = [
            {
                "document_id": doc_id,
                "representation_type": representation_type,
                "expires_in": 3600
            }
            for doc_id in document_ids
        ]

        payload = {"documents": documents}

        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.info(f"Requesting batch download for {len(document_ids)} documents")
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def get_document_by_id(self, document_id: int) -> dict[str, Any]:
        """Get a specific document by ID"""
        url = f"{self.base_url}/kdocuments/{document_id}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"Fetching document {document_id}")
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()


# Initialize API client
api_client = LibrarianAPIClient(API_KEY, LIBRARIAN_BASE_URL)


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
async def librarian_get_companies(
    countries: Optional[list[str]] = None,
    page: int = 1,
    page_size: int = 100
) -> str:
    """
    Fetch companies from Librarian API, optionally filtered by country.

    Args:
        countries: List of country names to filter by (e.g., ["Canada", "United Kingdom"])
        page: Page number (default: 1)
        page_size: Number of results per page (default: 100, max: 250)

    Returns:
        JSON string containing company data with pagination info

    Example:
        Get all Canadian companies:
        librarian_get_companies(countries=["Canada"])
    """
    try:
        result = await api_client.get_companies(
            countries=countries,
            page=page,
            page_size=page_size
        )

        # Extract useful information
        data = result.get("data", [])
        message = result.get("message", "")

        summary = {
            "message": message,
            "page": page,
            "page_size": page_size,
            "companies_count": len(data),
            "companies": [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "ticker": c.get("ticker"),
                    "country": c.get("country"),
                    "sector": c.get("sector"),
                }
                for c in data
            ]
        }

        import json
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        return f"Error: {str(e)}"


# @mcp.tool()  # DISABLED - Only exposing librarian_get_companies
async def librarian_search_documents(
    company_ids: Optional[list[int]] = None,
    document_types: Optional[list[str]] = None,
    filing_date_start: Optional[str] = None,
    filing_date_end: Optional[str] = None,
    page: int = 1,
    page_size: int = 250
) -> str:
    """
    Search for documents in the Librarian API.

    Args:
        company_ids: List of company IDs to filter by
        document_types: List of document types (e.g., ["filing", "slides"])
        filing_date_start: Start date in YYYY-MM-DD format
        filing_date_end: End date in YYYY-MM-DD format
        page: Page number (default: 1)
        page_size: Number of results per page (default: 250)

    Returns:
        JSON string containing document data with metadata

    Example:
        Search for all filings from specific companies:
        librarian_search_documents(company_ids=[12345, 67890], document_types=["filing"])
    """
    try:
        result = await api_client.search_documents(
            company_ids=company_ids,
            document_types=document_types,
            filing_date_start=filing_date_start,
            filing_date_end=filing_date_end,
            page=page,
            page_size=page_size
        )

        data = result.get("data", [])
        message = result.get("message", "")

        summary = {
            "message": message,
            "page": page,
            "page_size": page_size,
            "documents_count": len(data),
            "documents": [
                {
                    "id": d.get("id"),
                    "company_id": d.get("company_id"),
                    "document_type": d.get("document_type"),
                    "title": d.get("title"),
                    "filing_date": d.get("filing_date"),
                    "pages": d.get("pages"),
                }
                for d in data
            ]
        }

        import json
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error: {str(e)}"


# @mcp.tool()  # DISABLED - Only exposing librarian_get_companies
async def librarian_batch_download(
    document_ids: list[int],
    representation_type: str = "raw"
) -> str:
    """
    Request batch download URLs for multiple documents.

    Args:
        document_ids: List of document IDs to download
        representation_type: Type of representation ("raw", "clean", or "clean_full")

    Returns:
        JSON string containing download URLs for each document

    Example:
        Get download URLs for documents:
        librarian_batch_download(document_ids=[101356, 101357], representation_type="raw")

    Note:
        URLs expire after 1 hour by default
        Maximum 250 documents per request
    """
    try:
        if len(document_ids) > 250:
            return "Error: Maximum 250 documents per request"

        if representation_type not in ["raw", "clean", "clean_full"]:
            return "Error: representation_type must be 'raw', 'clean', or 'clean_full'"

        result = await api_client.batch_download(
            document_ids=document_ids,
            representation_type=representation_type
        )

        results = result.get("data", {}).get("results", [])

        summary = {
            "total_requested": len(document_ids),
            "results": [
                {
                    "document_id": r.get("document_id"),
                    "download_url": r.get("download_url"),
                    "error": r.get("error"),
                    "status": "success" if r.get("download_url") else "failed"
                }
                for r in results
            ]
        }

        import json
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"Error requesting batch download: {e}")
        return f"Error: {str(e)}"


# @mcp.tool()  # DISABLED - Only exposing librarian_get_companies
async def librarian_get_document_metadata(document_id: int) -> str:
    """
    Get detailed metadata for a specific document.

    Args:
        document_id: The ID of the document to retrieve

    Returns:
        JSON string containing full document metadata

    Example:
        Get metadata for a document:
        librarian_get_document_metadata(document_id=101356)
    """
    try:
        result = await api_client.get_document_by_id(document_id)

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error fetching document metadata: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# MCP RESOURCES
# ============================================================================

@mcp.resource("librarian://companies/{country}")
async def get_companies_by_country(country: str) -> str:
    """
    Get all companies for a specific country.

    Args:
        country: Country name (e.g., "Canada", "United Kingdom")

    Returns:
        JSON string containing companies in that country
    """
    try:
        result = await api_client.get_companies(
            countries=[country],
            page=1,
            page_size=250
        )

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error fetching companies for {country}: {e}")
        return f"Error: {str(e)}"


@mcp.resource("librarian://documents/{company_id}")
async def get_documents_by_company(company_id: str) -> str:
    """
    Get all documents for a specific company.

    Args:
        company_id: The company ID

    Returns:
        JSON string containing documents for that company
    """
    try:
        company_id_int = int(company_id)
        result = await api_client.search_documents(
            company_ids=[company_id_int],
            page=1,
            page_size=250
        )

        import json
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error fetching documents for company {company_id}: {e}")
        return f"Error: {str(e)}"


@mcp.resource("librarian://processed/{document_id}.md")
async def get_processed_document(document_id: str) -> str:
    """
    Get the processed markdown content for a document.

    Args:
        document_id: The document ID (without .md extension)

    Returns:
        Markdown content of the processed document
    """
    try:
        # Remove .md if present
        doc_id = document_id.replace(".md", "")

        # Look for the processed file
        processed_file = PROCESSED_DIR / f"doc_{doc_id}.md"

        if not processed_file.exists():
            return f"Error: Processed document not found at {processed_file}"

        with open(processed_file, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    except Exception as e:
        logger.error(f"Error reading processed document {document_id}: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# PROMPTS
# ============================================================================

@mcp.prompt()
def fetch_and_process_documents(
    countries: str = "Canada,United Kingdom",
    document_types: str = "filing,slides",
    max_documents: int = 10
) -> str:
    """
    Generate a prompt to fetch and process documents from specific countries.

    Args:
        countries: Comma-separated list of countries
        document_types: Comma-separated list of document types
        max_documents: Maximum number of documents to process

    Returns:
        A prompt that guides the AI through the document fetching process
    """
    countries_list = [c.strip() for c in countries.split(",")]
    doc_types_list = [d.strip() for d in document_types.split(",")]

    return f"""Please help me fetch and process documents from the Librarian API.

Steps:
1. First, use librarian_get_companies to fetch companies from these countries: {', '.join(countries_list)}
2. Extract the company IDs from the results
3. Use librarian_search_documents to find {', '.join(doc_types_list)} documents for those companies
4. Select up to {max_documents} documents to process
5. Use librarian_batch_download to get download URLs for the selected documents
6. Provide a summary of what documents were found and are ready to download

Please start with step 1."""


# ============================================================================
# HEALTH CHECK (for Render deployment)
# ============================================================================

# @mcp.tool()  # DISABLED - Only exposing librarian_get_companies
async def health_check() -> str:
    """
    Health check endpoint for monitoring.

    Returns:
        JSON string with server status
    """
    import json
    return json.dumps({
        "status": "healthy",
        "service": "Librarian MCP Server",
        "api_base": LIBRARIAN_BASE_URL
    })


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check for command line arguments
    use_http = "--http" in sys.argv or os.getenv("USE_HTTP", "").lower() == "true"

    logger.info("Starting Librarian MCP Server...")
    logger.info(f"API Base URL: {LIBRARIAN_BASE_URL}")
    logger.info(f"Processed files directory: {PROCESSED_DIR}")

    if use_http:
        # HTTP/SSE mode for Render deployment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        logger.info(f"Running in HTTP/SSE mode on {host}:{port}")
        logger.info(f"SSE endpoint: http://{host}:{port}/sse")
        mcp.run(transport="sse", host=host, port=port)
    else:
        # stdio mode for Claude Desktop
        logger.info("Running in stdio mode for Claude Desktop")
        mcp.run()
