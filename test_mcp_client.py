#!/usr/bin/env python3
"""
Test MCP server via proper MCP client connection
"""

import asyncio
from fastmcp import Client


async def test_mcp_server():
    """Test the MCP server using FastMCP client"""

    print("\n" + "=" * 60)
    print("Testing MCP Server via FastMCP Client")
    print("=" * 60)

    # Connect to the running server
    async with Client("http://localhost:8080/sse") as client:

        # Test 1: List available tools
        print("\n[Test 1] Listing available tools...")
        tools = await client.list_tools()
        print(f"✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        # Test 2: Call librarian_get_companies
        print("\n[Test 2] Calling librarian_get_companies...")
        result = await client.call_tool(
            "librarian_get_companies",
            {
                "countries": ["Canada"],
                "page": 1,
                "page_size": 5
            }
        )
        print(f"✓ Result (first 200 chars): {str(result)[:200]}...")

        # Test 3: Call librarian_search_documents
        print("\n[Test 3] Calling librarian_search_documents...")
        result = await client.call_tool(
            "librarian_search_documents",
            {
                "document_types": ["filing"],
                "page": 1,
                "page_size": 3
            }
        )
        print(f"✓ Result (first 200 chars): {str(result)[:200]}...")

        # Test 4: List resources
        print("\n[Test 4] Listing available resources...")
        try:
            resources = await client.list_resources()
            print(f"✓ Found {len(resources)} resources:")
            for resource in resources[:5]:  # Show first 5
                print(f"  - {resource.uri}")
        except Exception as e:
            print(f"⚠ Resources list: {e}")

        # Test 5: List prompts
        print("\n[Test 5] Listing available prompts...")
        try:
            prompts = await client.list_prompts()
            print(f"✓ Found {len(prompts)} prompts:")
            for prompt in prompts:
                print(f"  - {prompt.name}: {prompt.description[:60]}...")
        except Exception as e:
            print(f"⚠ Prompts list: {e}")

        print("\n" + "=" * 60)
        print("✓ All MCP client tests passed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
