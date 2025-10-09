import asyncio
from fastmcp import Client

async def test():
    print("\n" + "=" * 80)
    print("TESTING FILTERED LIBRARIAN PROXY")
    print("=" * 80)
    
    async with Client("librarian_proxy_filtered.py") as client:
        print("\n✓ Connected to filtered proxy")
        
        # List tools
        tools = await client.list_tools()
        
        print(f"\n{'=' * 80}")
        print(f"FILTERED TOOLS: {len(tools)} total (expected 4)")
        print(f"{'=' * 80}\n")
        
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
        
        print(f"\n{'=' * 80}")
        
        if len(tools) == 4:
            print("✓ Filter working correctly!")
            print("  Original: 22 tools")
            print("  Filtered: 4 tools")
            print("  Reduction: 82%")
        else:
            print(f"⚠ Expected 4 tools, got {len(tools)}")
        
        print(f"{'=' * 80}\n")
        
        # Test a tool call
        print("Testing tool: search_companies")
        try:
            result = await client.call_tool(
                "search_companies",
                {"name": "Digital", "limit": 1}
            )
            print("✓ Tool call successful!")
        except Exception as e:
            print(f"✗ Tool call failed: {e}")

asyncio.run(test())
