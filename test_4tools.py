import asyncio
from fastmcp import Client

async def test():
    print("\n" + "=" * 80)
    print("TESTING 4-TOOL ENFORCED PROXY")
    print("=" * 80)
    
    async with Client("librarian_proxy_4tools.py") as client:
        print("\n✓ Connected to proxy")
        
        # List tools
        tools = await client.list_tools()
        
        print(f"\nTools Available: {len(tools)}")
        print("=" * 80)
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
        
        print("\n" + "=" * 80)
        
        if len(tools) == 4:
            print("✓ SUCCESS: Enforced filtering working!")
            print("  Original: 22 tools")
            print("  Filtered: 4 tools (82% reduction)")
        else:
            print(f"⚠ Expected 4 tools, got {len(tools)}")
        
        print("=" * 80)
        
        # Test a tool
        print("\nTesting: search_companies(name='Digital', limit=1)")
        result = await client.call_tool(
            "search_companies",
            {"name": "Digital", "limit": 1}
        )
        print("✓ Tool call successful!")
        print(f"  Result preview: {str(result)[:100]}...")

asyncio.run(test())
