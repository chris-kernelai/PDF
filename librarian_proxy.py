#!/usr/bin/env python3
"""
Librarian MCP Proxy Server

A true proxy that forwards all tools from the remote Librarian API.
Exposes all 22 tools from https://librarian.production.primerapp.com/sse
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

# Create proxy server that forwards all tools from remote API
logger.info(f"Creating proxy to remote Librarian API: {LIBRARIAN_API_URL}")

proxy = FastMCP.as_proxy(
    LIBRARIAN_API_URL,
    name="Librarian API Proxy"
)

if __name__ == "__main__":
    # Check for command line arguments
    use_http = "--http" in sys.argv or os.getenv("USE_HTTP", "").lower() == "true"

    logger.info("Starting Librarian MCP Proxy Server...")
    logger.info(f"Proxying to: {LIBRARIAN_API_URL}")

    if use_http:
        # HTTP/SSE mode for Render deployment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        logger.info(f"Running in HTTP/SSE mode on {host}:{port}")
        logger.info(f"SSE endpoint: http://{host}:{port}/sse")
        proxy.run(transport="sse", host=host, port=port)
    else:
        # stdio mode for Claude Desktop
        logger.info("Running in stdio mode for Claude Desktop")
        proxy.run()
