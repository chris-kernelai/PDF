# Tool Filtering Options for Librarian MCP Proxy

The remote Librarian API exposes 22 tools. Here are several ways to restrict which tools are available:

## Summary of Options

| Option | Difficulty | Flexibility | When to Use |
|--------|-----------|-------------|-------------|
| **Option 1: Client-side** | Easy | High | Claude Desktop, per-user control |
| **Option 2: Documentation** | Easy | Low | Guide users on which tools to use |
| **Option 3: Wrapper Proxy** | Medium | High | Need server-side enforcement |
| **Option 4: Custom Server** | Hard | Maximum | Need complex logic/auth |

---

## Option 1: Client-Side Filtering (Recommended)

**Best for:** Claude Desktop users, individual control

Claude Desktop and other MCP clients can be configured to only use specific tools, even if the server exposes more.

### Claude Desktop Config with Tool Hints

```json
{
  "mcpServers": {
    "librarian": {
      "command": "python",
      "args": ["/path/to/librarian_proxy.py"],
      "description": "Librarian API - Use search_companies, search_documents, filter_documents, get_document_content"
    }
  }
}
```

### Pros:
- ✅ No server changes needed
- ✅ Different users can have different restrictions
- ✅ Easy to modify
- ✅ All tools still available if needed

### Cons:
- ❌ Not enforced at server level
- ❌ Users could still access other tools directly

---

## Option 2: Documentation-Based (Current Approach)

**Best for:** Providing guidance without enforcement

Use `librarian_proxy_simple_filter.py` which documents recommended tools but doesn't enforce restrictions.

### How it Works:

```python
TOOL_ALLOWLIST = [
    "search_companies",
    "search_documents",
    "filter_documents",
    "get_document_content",
]
```

The server logs which tools are recommended, but all 22 tools are still available.

### Pros:
- ✅ Simple implementation
- ✅ Clear documentation
- ✅ Flexibility for advanced users

### Cons:
- ❌ No actual enforcement
- ❌ Users can still access all tools

---

## Option 3: Wrapper Proxy with Enforcement

**Best for:** True server-side filtering

Create a custom FastMCP server that only registers specific tools from the remote API.

### Implementation:

```python
from fastmcp import FastMCP, Client

mcp = FastMCP("Filtered Librarian")

# Only expose specific tools
ALLOWED_TOOLS = ["search_companies", "search_documents"]

@mcp.tool()
async def search_companies(name: str = None, limit: int = 10):
    """Search for companies (proxied from remote API)"""
    async with Client("https://librarian.production.primerapp.com/sse") as client:
        result = await client.call_tool("search_companies", {"name": name, "limit": limit})
        return result.content[0].text

@mcp.tool()
async def search_documents(filters: dict):
    """Search documents (proxied from remote API)"""
    async with Client("https://librarian.production.primerapp.com/sse") as client:
        result = await client.call_tool("search_documents", filters)
        return result.content[0].text

if __name__ == "__main__":
    mcp.run()
```

### Pros:
- ✅ True server-side enforcement
- ✅ Only selected tools available
- ✅ Full control over parameters

### Cons:
- ❌ More code to maintain
- ❌ Need to manually wrap each tool
- ❌ More complex

---

## Option 4: Custom Server (Your Current librarian_mcp_server.py)

**Best for:** Maximum control, custom logic

You already have this! `librarian_mcp_server.py` creates custom tools that call the REST API directly.

### Current State:

```python
# Only exposing 1 tool (commented out the others)
@mcp.tool()
async def librarian_get_companies(...):
    # Custom implementation
    pass

# @mcp.tool()  # DISABLED
# async def librarian_search_documents(...):
#     pass
```

### Pros:
- ✅ Complete control
- ✅ Can add custom logic/validation
- ✅ Can combine multiple remote calls
- ✅ True enforcement

### Cons:
- ❌ Most code to maintain (500+ lines)
- ❌ Must manually update when API changes
- ❌ Limited to tools you implement

---

## Recommended Approach by Use Case

### For Claude Desktop Users:
**→ Use Option 1 (Client-side filtering)**

Configure Claude Desktop to prefer certain tools:

```json
{
  "mcpServers": {
    "librarian": {
      "command": "python",
      "args": ["/path/to/librarian_proxy.py"],
      "notes": "Recommended tools: search_companies, search_documents, filter_documents"
    }
  }
}
```

Then instruct Claude:
```
"Only use search_companies, search_documents, filter_documents, and get_document_content tools from the Librarian API"
```

### For Multi-User Deployment:
**→ Use Option 3 (Wrapper Proxy)**

Create a server that only exposes approved tools. This provides true enforcement at the server level.

### For Maximum Customization:
**→ Use Option 4 (Custom Server)**

Your existing `librarian_mcp_server.py` with tools enabled/disabled as needed.

---

## Practical Example: 4-Tool Restriction

Want to only expose these 4 tools?
1. search_companies
2. search_documents
3. filter_documents
4. get_document_content

### Quick Solution (Client-side):

```bash
# Use the unrestricted proxy
python librarian_proxy.py

# In Claude Desktop, add system instruction:
"Only use these Librarian tools: search_companies, search_documents,
filter_documents, get_document_content. Do not use any other Librarian tools."
```

### Enforced Solution (Server-side):

Create `librarian_proxy_4tools.py`:

```python
from fastmcp import FastMCP, Client

REMOTE_URL = "https://librarian.production.primerapp.com/sse"
mcp = FastMCP("Librarian (4 Tools)")

@mcp.tool()
async def search_companies(name: str = None, limit: int = 10):
    """Search for companies by name or ticker"""
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool("search_companies",
                                        {"name": name, "limit": limit})
        return result.content[0].text

@mcp.tool()
async def search_documents(filters: dict):
    """Search documents with comprehensive filtering"""
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool("search_documents", filters)
        return result.content[0].text

@mcp.tool()
async def filter_documents(filters: dict, dedupe_methodology: str = "none"):
    """Filter documents by date range and company"""
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool("filter_documents",
                                        {"filters": filters,
                                         "dedupe_methodology": dedupe_methodology})
        return result.content[0].text

@mcp.tool()
async def get_document_content(document_id: int, max_tokens: int = 100000):
    """Get plaintext content of a document"""
    async with Client(REMOTE_URL) as client:
        result = await client.call_tool("get_document_content",
                                        {"document_id": document_id,
                                         "max_tokens": max_tokens})
        return result.content[0].text

if __name__ == "__main__":
    mcp.run()
```

---

## Decision Matrix

**Need enforcement?**
- No → Option 1 or 2
- Yes → Option 3 or 4

**How many tools?**
- All (22) → Option 1 or 2
- Most (10-20) → Option 1 or 2
- Few (1-10) → Option 3 or 4
- One (1) → Option 4 (existing librarian_mcp_server.py)

**Development effort?**
- Minimal → Option 1 or 2
- Medium → Option 3
- High → Option 4

---

## My Recommendation

For your use case, I recommend **Option 1 (Client-side filtering)**:

1. Use the simple proxy: `librarian_proxy.py` (exposes all 22 tools)
2. Configure Claude Desktop with guidance on which tools to use
3. Add system instructions to Claude to only use approved tools

This gives you:
- ✅ Minimal maintenance (40 lines of code)
- ✅ Flexibility (can adjust per user)
- ✅ All tools available if needed
- ✅ Simple deployment

If you need true server-side enforcement later, you can always switch to Option 3.

---

## Current Files Available

1. **`librarian_proxy.py`** - Full proxy (all 22 tools)
2. **`librarian_mcp_server.py`** - Custom server (1 tool currently enabled)
3. **`librarian_proxy_simple_filter.py`** - Proxy with documentation

Choose based on your needs!
