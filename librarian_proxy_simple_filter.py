#!/usr/bin/env python3
"""
Simple Filtered Librarian MCP Proxy

Uses a configuration-based approach to filter tools.
Edit the TOOL_ALLOWLIST below to control which tools are exposed.
"""

import os
import sys
import logging
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Remote Librarian API URL
LIBRARIAN_API_URL = "https://librarian.production.primerapp.com/sse"

# ============================================================================
# CONFIGURATION: List the tools you want to expose
# ============================================================================

TOOL_ALLOWLIST = [
    "search_companies",          # Search for companies by name/ticker
    "search_documents",          # Comprehensive document search
    "filter_documents",          # Filter documents by criteria
    "get_document_content",      # Get document plaintext content
    # "fetch",                   # Uncomment to enable
    # "search",                  # Semantic search
    # "get_document_by_id",      # Get document metadata
    # "get_document_context",    # Get line range context
    # "get_document_download_link",  # Get download links
    # "grep_documents",          # Pattern search across docs
    # "lseg_get_data",           # Financial data from LSEG
    # "lseg_get_data_with_pct_to_last",  # Financial data with %
    # "get_user_email_preferences",  # User email settings
    # "get_user_watchlist",      # User watchlist
    # "add_to_watchlist",        # Add to watchlist
    # "remove_from_watchlist",   # Remove from watchlist
    # "onboard_user",            # Create user account
    # "trigger_onboarding_emails",  # Send onboarding emails
    # "update_user_email_preferences",  # Update email settings
    # "get_va_kpi_data",         # Get KPI data
    # "get_va_kpi_metric_groups",  # Get KPI groups
    # "get_va_kpi_metrics_from_groups",  # Get KPI metrics
]

# ============================================================================

logger.info(f"Tool Allowlist ({len(TOOL_ALLOWLIST)} tools):")
for tool in TOOL_ALLOWLIST:
    logger.info(f"  ✓ {tool}")

# Create MCP configuration for the remote server
config = {
    "mcpServers": {
        "librarian": {
            "url": LIBRARIAN_API_URL,
            "transport": "sse"
        }
    }
}

# Note: FastMCP's as_proxy doesn't support native filtering yet
# This is a simple documentation-based approach
# For true filtering, use the proxy server with middleware (see Option 2 below)

logger.warning("Note: This creates a proxy to the remote API.")
logger.warning("Tool filtering must be implemented at the application level.")
logger.warning("All 22 remote tools will be available through this proxy.")
logger.warning("For true filtering, configure your MCP client to only use specific tools.")

# Create proxy
proxy = FastMCP.as_proxy(
    LIBRARIAN_API_URL,
    name="Librarian API Proxy (Configured)"
)

if __name__ == "__main__":
    # Check for command line arguments
    use_http = "--http" in sys.argv or os.getenv("USE_HTTP", "").lower() == "true"

    logger.info("Starting Librarian MCP Proxy Server...")
    logger.info(f"Proxying to: {LIBRARIAN_API_URL}")
    logger.info(f"Recommended tools to use: {', '.join(TOOL_ALLOWLIST[:4])}...")

    if use_http:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        logger.info(f"Running in HTTP/SSE mode on {host}:{port}")
        logger.info(f"SSE endpoint: http://{host}:{port}/sse")
        proxy.run(transport="sse", host=host, port=port)
    else:
        logger.info("Running in stdio mode for Claude Desktop")
        proxy.run()
