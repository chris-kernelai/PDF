#!/usr/bin/env python3
"""
Filtered Librarian MCP Proxy Server

Forwards only selected tools from the remote Librarian API.
Customize the ALLOWED_TOOLS list to control which tools are exposed.
"""

import os
import sys
import logging
from fastmcp import FastMCP
from mcp.types import Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Remote Librarian API URL
LIBRARIAN_API_URL = "https://librarian.production.primerapp.com/sse"

# ============================================================================
# CONFIGURATION: Choose which tools to expose
# ============================================================================

# Option 1: Whitelist specific tools (recommended)
ALLOWED_TOOLS = [
    "search_companies",
    "search_documents",
    "filter_documents",
    "get_document_content",
    # Add more tools here as needed
]

# Option 2: Blacklist specific tools (expose all except these)
# BLOCKED_TOOLS = [
#     "onboard_user",
#     "trigger_onboarding_emails",
#     "update_user_email_preferences",
# ]

# Set to "whitelist" or "blacklist"
FILTER_MODE = "whitelist"

# ============================================================================


class FilteredProxy:
    """Proxy that filters tools based on configuration"""

    def __init__(self, remote_url: str, allowed_tools: list[str] = None, blocked_tools: list[str] = None, mode: str = "whitelist"):
        self.remote_url = remote_url
        self.allowed_tools = set(allowed_tools or [])
        self.blocked_tools = set(blocked_tools or [])
        self.mode = mode
        self.mcp = FastMCP("Librarian API Proxy (Filtered)")

        logger.info(f"Filter mode: {mode}")
        if mode == "whitelist":
            logger.info(f"Allowed tools: {sorted(self.allowed_tools)}")
        else:
            logger.info(f"Blocked tools: {sorted(self.blocked_tools)}")

    def should_expose_tool(self, tool_name: str) -> bool:
        """Check if a tool should be exposed based on filter settings"""
        if self.mode == "whitelist":
            return tool_name in self.allowed_tools
        else:  # blacklist mode
            return tool_name not in self.blocked_tools

    async def setup(self):
        """Setup the proxy with filtered tools"""
        from fastmcp import Client

        # Connect to remote API to discover tools
        async with Client(self.remote_url) as client:
            remote_tools = await client.list_tools()

            logger.info(f"Remote API has {len(remote_tools)} tools")

            # Filter tools based on configuration
            filtered_count = 0
            exposed_count = 0

            for tool in remote_tools:
                if self.should_expose_tool(tool.name):
                    # Create a wrapper tool that forwards to remote
                    self._create_tool_wrapper(tool, client)
                    exposed_count += 1
                    logger.info(f"✓ Exposing tool: {tool.name}")
                else:
                    filtered_count += 1
                    logger.debug(f"✗ Filtering out tool: {tool.name}")

            logger.info(f"Exposed {exposed_count} tools, filtered {filtered_count} tools")

    def _create_tool_wrapper(self, tool: Tool, client):
        """Create a wrapper function that forwards to the remote tool"""
        tool_name = tool.name

        # Dynamically create a tool function
        async def tool_wrapper(**kwargs):
            """Forward tool call to remote API"""
            from fastmcp import Client
            async with Client(self.remote_url) as remote_client:
                result = await remote_client.call_tool(tool_name, kwargs)
                # Extract text content from result
                if hasattr(result, 'content') and result.content:
                    return result.content[0].text
                return str(result)

        # Set function metadata
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool.description if isinstance(tool.description, str) else str(tool.description)

        # Register as MCP tool
        self.mcp.tool()(tool_wrapper)

    def run(self, **kwargs):
        """Run the proxy server"""
        self.mcp.run(**kwargs)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import asyncio

    # Check for command line arguments
    use_http = "--http" in sys.argv or os.getenv("USE_HTTP", "").lower() == "true"

    logger.info("Starting Filtered Librarian MCP Proxy Server...")
    logger.info(f"Proxying to: {LIBRARIAN_API_URL}")

    # Create filtered proxy
    if FILTER_MODE == "whitelist":
        proxy = FilteredProxy(
            LIBRARIAN_API_URL,
            allowed_tools=ALLOWED_TOOLS,
            mode="whitelist"
        )
    else:
        proxy = FilteredProxy(
            LIBRARIAN_API_URL,
            blocked_tools=BLOCKED_TOOLS if 'BLOCKED_TOOLS' in globals() else [],
            mode="blacklist"
        )

    # Setup filters (load tools from remote)
    asyncio.run(proxy.setup())

    # Run server
    if use_http:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8080"))
        logger.info(f"Running in HTTP/SSE mode on {host}:{port}")
        logger.info(f"SSE endpoint: http://{host}:{port}/sse")
        proxy.run(transport="sse", host=host, port=port)
    else:
        logger.info("Running in stdio mode for Claude Desktop")
        proxy.run()
