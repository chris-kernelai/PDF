#!/usr/bin/env python3
"""
Librarian MCP Proxy - 4 Essential Tools Only

Server-side enforcement: Only exposes 4 document search tools.
This is a wrapper that forwards to the remote API but only exposes selected tools.
"""

import os
import sys
import logging
from fastmcp import FastMCP, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Remote Librarian API URL
REMOTE_URL = "https://librarian.production.primerapp.com/sse"

# Create MCP server with limited tools
mcp = FastMCP("Librarian (4 Tools)")

logger.info("Exposing only 4 tools:")
logger.info("  1. search_companies")
logger.info("  2. search_documents")
logger.info("  3. filter_documents")
logger.info("  4. get_document_content")


# ============================================================================
# TOOL 1: Search Companies
# ============================================================================

@mcp.tool()
async def search_companies(name: str = None, limit: int = 10) -> str:
    """
    Search for companies in Primer's database by name or ticker.

    Args:
        name: Company name or ticker to search for (fuzzy matching)
        limit: Maximum number of results to return (default: 10)

    Returns:
        JSON string with matching companies

    Example:
        search_companies(name="Digital Infrastructure", limit=5)
    """
    logger.info(f"Proxying search_companies: name={name}, limit={limit}")
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool(
            "search_companies",
            {"name": name, "limit": limit}
        )
        return result.content[0].text


# ============================================================================
# TOOL 2: Search Documents
# ============================================================================

@mcp.tool()
async def search_documents(filters: dict) -> str:
    """
    Search kdocuments with comprehensive filtering options and pagination.

    Args:
        filters: Dictionary containing filter criteria:
            - company_ids: List of company IDs (optional)
            - document_types: List of types like ["filing", "slides"] (optional)
            - filing_date_start: Start date "YYYY-MM-DD" (optional)
            - filing_date_end: End date "YYYY-MM-DD" (optional)
            - page: Page number (default: 1)
            - page_size: Results per page (default: 250)

    Returns:
        JSON string with matching documents

    Example:
        search_documents(filters={
            "company_ids": [12345],
            "document_types": ["filing"],
            "filing_date_start": "2024-01-01"
        })
    """
    logger.info(f"Proxying search_documents: filters={filters}")
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool("search_documents", filters)
        return result.content[0].text


# ============================================================================
# TOOL 3: Filter Documents
# ============================================================================

@mcp.tool()
async def filter_documents(
    filters: dict,
    dedupe_methodology: str = "none"
) -> str:
    """
    Filter documents by date range, company identifiers, and other criteria.

    Args:
        filters: Dictionary containing:
            - date_range: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} (optional)
            - company_ids: List of company IDs (optional)
            - document_types: List of types (optional)
        dedupe_methodology: How to handle duplicates:
            "none" (default), "latest", or other options

    Returns:
        JSON string with filtered documents

    Example:
        filter_documents(filters={
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "company_ids": [12345, 67890]
        })
    """
    logger.info(f"Proxying filter_documents: filters={filters}, dedupe={dedupe_methodology}")
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool(
            "filter_documents",
            {"filters": filters, "dedupe_methodology": dedupe_methodology}
        )
        return result.content[0].text


# ============================================================================
# TOOL 4: Get Document Content
# ============================================================================

@mcp.tool()
async def get_document_content(
    document_id: int,
    max_tokens: int = 100000
) -> str:
    """
    Get the plaintext content of a document.

    Note: Only works for filings or call transcripts, not slides.

    Args:
        document_id: The unique identifier of the document
        max_tokens: Maximum number of tokens to return (default: 100000)

    Returns:
        Plaintext content of the document

    Example:
        get_document_content(document_id=101356, max_tokens=50000)
    """
    logger.info(f"Proxying get_document_content: doc_id={document_id}, max_tokens={max_tokens}")
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool(
            "get_document_content",
            {"document_id": document_id, "max_tokens": max_tokens}
        )
        return result.content[0].text


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

if __name__ == "__main__":
    # Check for command line arguments
    use_http = "--http" in sys.argv or os.getenv("USE_HTTP", "").lower() == "true"

    logger.info("Starting Filtered Librarian MCP Proxy (4 Tools Only)")
    logger.info(f"Proxying to: {REMOTE_URL}")
    logger.info("Tools exposed: search_companies, search_documents, filter_documents, get_document_content")
    logger.info("Tools blocked: 18 other tools (financial data, user management, KPIs, etc.)")

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
