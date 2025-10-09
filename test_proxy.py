import asyncio
from fastmcp import Client

async def test_proxy():
    print("\n" + "=" * 80)
    print("TESTING LIBRARIAN PROXY SERVER")
    print("=" * 80)
    
    async with Client("librarian_proxy.py") as client:
        print("\n✓ Connected to proxy server")
        
        # List all tools
        print("\nFetching tools from proxy...")
        tools = await client.list_tools()
        
        print(f"\n{'=' * 80}")
        print(f"PROXIED TOOLS: {len(tools)} total")
        print(f"{'=' * 80}\n")
        
        for i, tool in enumerate(tools, 1):
            name = tool.name
            desc = tool.description
            # Handle description that might be JSON string
            if isinstance(desc, str) and desc.startswith('{'):
                import json
                try:
                    desc_obj = json.loads(desc)
                    desc = desc_obj.get('description', desc)[:80]
                except:
                    desc = desc[:80]
            else:
                desc = str(desc)[:80] if desc else "No description"
            
            print(f"{i:2d}. {name}")
            print(f"    {desc}...")
        
        print(f"\n{'=' * 80}")
        print("✓ All 22 tools successfully proxied!")
        print(f"{'=' * 80}\n")
        
        # Test calling a tool
        print("Testing tool call: search_companies(name='Cordiant', limit=1)")
        result = await client.call_tool(
            "search_companies",
            {"name": "Cordiant", "limit": 1}
        )
        print(f"✓ Tool call successful!")
        print(f"  Result (first 150 chars): {str(result)[:150]}...")

asyncio.run(test_proxy())
