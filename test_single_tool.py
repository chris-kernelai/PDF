import asyncio
from fastmcp import Client

async def test():
    print("\n" + "=" * 60)
    print("Testing Restricted MCP Server (Only librarian_get_companies)")
    print("=" * 60)
    
    async with Client("librarian_mcp_server.py") as client:
        # List tools
        tools = await client.list_tools()
        print(f"\n✓ Tools available: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}")
        
        # Test the tool
        print("\n[Test] Calling librarian_get_companies for Canada...")
        result = await client.call_tool(
            "librarian_get_companies",
            {"countries": ["Canada"], "page": 1, "page_size": 3}
        )
        
        import json
        content = result.content[0].text
        data = json.loads(content)
        
        print(f"✓ Success!")
        print(f"  Message: {data['message']}")
        print(f"  Companies found: {data['companies_count']}")
        if data['companies']:
            print(f"  First company: {data['companies'][0]['name']}")
        
        print("\n" + "=" * 60)
        print("✓ Restricted server working correctly!")
        print("=" * 60 + "\n")

asyncio.run(test())
