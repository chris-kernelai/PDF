import asyncio
from fastmcp import Client

async def check_remote():
    print("\n" + "=" * 80)
    print("CHECKING REMOTE LIBRARIAN API")
    print("URL: https://librarian.production.primerapp.com/sse")
    print("=" * 80)
    
    try:
        async with Client("https://librarian.production.primerapp.com/sse") as client:
            print("\n✓ Successfully connected to remote server")
            
            # List tools
            print("\nFetching tools...")
            tools = await client.list_tools()
            
            print(f"\n{'=' * 80}")
            print(f"REMOTE SERVER TOOLS: {len(tools)} total")
            print(f"{'=' * 80}\n")
            
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool.name}")
                print(f"   Description: {tool.description[:100]}...")
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    schema = tool.inputSchema
                    if 'properties' in schema:
                        params = list(schema['properties'].keys())
                        print(f"   Parameters: {', '.join(params[:5])}")
                print()
            
            # List resources
            print(f"\n{'=' * 80}")
            print("REMOTE SERVER RESOURCES")
            print(f"{'=' * 80}\n")
            try:
                resources = await client.list_resources()
                print(f"Total resources: {len(resources)}")
                for i, resource in enumerate(resources[:10], 1):
                    print(f"{i}. {resource.uri}")
                    if hasattr(resource, 'name'):
                        print(f"   Name: {resource.name}")
                if len(resources) > 10:
                    print(f"... and {len(resources) - 10} more")
            except Exception as e:
                print(f"Could not list resources: {e}")
            
            # List prompts
            print(f"\n{'=' * 80}")
            print("REMOTE SERVER PROMPTS")
            print(f"{'=' * 80}\n")
            try:
                prompts = await client.list_prompts()
                print(f"Total prompts: {len(prompts)}")
                for i, prompt in enumerate(prompts, 1):
                    print(f"{i}. {prompt.name}")
                    if hasattr(prompt, 'description'):
                        print(f"   Description: {prompt.description[:100]}...")
            except Exception as e:
                print(f"Could not list prompts: {e}")
            
    except Exception as e:
        print(f"\n✗ Error connecting to remote server:")
        print(f"  {type(e).__name__}: {e}")
        print(f"\nThis could mean:")
        print("  - The server is not publicly accessible")
        print("  - Authentication is required")
        print("  - The URL/endpoint is incorrect")
        print("  - Network/firewall issues")

asyncio.run(check_remote())
