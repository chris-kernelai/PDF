import asyncio
from fastmcp import Client

async def list_all_tools():
    print("\n" + "=" * 80)
    print("LIBRARIAN MCP SERVER - COMPLETE TOOL LIST")
    print("=" * 80)
    
    async with Client("librarian_mcp_server.py") as client:
        tools = await client.list_tools()
        
        print(f"\nTotal Tools: {len(tools)}\n")
        
        for i, tool in enumerate(tools, 1):
            print(f"\n{'─' * 80}")
            print(f"Tool {i}: {tool.name}")
            print(f"{'─' * 80}")
            print(f"\nDescription:")
            print(f"  {tool.description}\n")
            
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if 'properties' in schema:
                    print("Parameters:")
                    for param, details in schema['properties'].items():
                        param_type = details.get('type', 'unknown')
                        param_desc = details.get('description', 'No description')
                        required = param in schema.get('required', [])
                        req_marker = "REQUIRED" if required else "optional"
                        print(f"  • {param} ({param_type}) [{req_marker}]")
                        print(f"    {param_desc}")
        
        print(f"\n{'=' * 80}\n")

asyncio.run(list_all_tools())
