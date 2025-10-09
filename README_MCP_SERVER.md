# Librarian MCP Server

A Model Context Protocol (MCP) server that exposes the Librarian API as tools and resources for AI assistants like Claude.

## Overview

This MCP server provides a seamless interface to the Librarian API, allowing AI assistants to:
- Search and filter companies by country
- Find documents (filings, slides, etc.)
- Request batch download URLs for documents
- Access processed markdown documents
- Use pre-built prompts for common workflows

## Architecture

```
┌─────────────────────────────────────┐
│  AI Client (Claude Desktop, etc.)   │
└──────────────┬──────────────────────┘
               │ MCP Protocol (stdio)
               ↓
┌─────────────────────────────────────┐
│  Librarian MCP Server                │
│  (librarian_mcp_server.py)           │
│                                      │
│  Tools:                              │
│  - librarian_get_companies           │
│  - librarian_search_documents        │
│  - librarian_batch_download          │
│  - librarian_get_document_metadata   │
│                                      │
│  Resources:                          │
│  - librarian://companies/{country}   │
│  - librarian://documents/{id}        │
│  - librarian://processed/{id}.md     │
└──────────────┬──────────────────────┘
               │ HTTPS/REST
               ↓
┌─────────────────────────────────────┐
│  Librarian API                       │
│  librarian.production.primerapp.com  │
└─────────────────────────────────────┘
```

## Features

### MCP Tools

**1. librarian_get_companies**
Fetch companies filtered by country.

```python
# Example usage in Claude
Use the librarian_get_companies tool with:
- countries: ["Canada", "United Kingdom"]
- page: 1
- page_size: 100
```

**2. librarian_search_documents**
Search for documents by company, type, and date range.

```python
# Example usage
Use librarian_search_documents with:
- company_ids: [12345, 67890]
- document_types: ["filing", "slides"]
- filing_date_start: "2024-01-01"
- page: 1
- page_size: 250
```

**3. librarian_batch_download**
Request presigned S3 URLs for downloading multiple documents.

```python
# Example usage
Use librarian_batch_download with:
- document_ids: [101356, 101357, 101358]
- representation_type: "raw"  # or "clean", "clean_full"
```

**4. librarian_get_document_metadata**
Get detailed metadata for a specific document.

```python
# Example usage
Use librarian_get_document_metadata with:
- document_id: 101356
```

### MCP Resources

Resources provide direct URI-based access to data:

**1. librarian://companies/{country}**
Get all companies for a specific country.

```
Example: librarian://companies/Canada
```

**2. librarian://documents/{company_id}**
Get all documents for a specific company.

```
Example: librarian://documents/12345
```

**3. librarian://processed/{document_id}.md**
Access processed markdown content from the local `processed/` directory.

```
Example: librarian://processed/101356.md
```

### MCP Prompts

Pre-built prompts for common workflows:

**fetch_and_process_documents**
Guides the AI through fetching and processing documents from specific countries.

```python
# Parameters:
- countries: "Canada,United Kingdom"
- document_types: "filing,slides"
- max_documents: 10
```

## Installation

### 1. Install Dependencies

```bash
# Ensure you're in the PDF directory
cd /Users/chrismingard/Kernel/PDF

# Activate virtual environment (if using one)
source venv/bin/activate

# Install FastMCP and dependencies
pip install -r requirements.txt
```

### 2. Set API Key

The API key is already configured in the server code, but you can override it via environment variable:

```bash
export LIBRARIAN_API_KEY="your-api-key-here"
```

### 3. Test the Server

```bash
# Run the server directly to test
python librarian_mcp_server.py
```

The server will start and listen for MCP protocol messages via stdio.

## Claude Desktop Integration

To use this MCP server with Claude Desktop, add the configuration to your Claude Desktop config file.

### Configuration File Location

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Configuration

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "librarian": {
      "command": "python",
      "args": [
        "/Users/chrismingard/Kernel/PDF/librarian_mcp_server.py"
      ],
      "env": {
        "LIBRARIAN_API_KEY": "8fc83f71-23e1-40bd-93a4-4e0a47cdcb44"
      }
    }
  }
}
```

**Important:** Update the path in `args` to match your actual installation directory.

### Restart Claude Desktop

After adding the configuration:
1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The Librarian tools should now be available

### Verify Integration

In Claude Desktop, you should see the Librarian tools available. You can test with:

```
Can you show me what Librarian tools are available?
```

Claude should list:
- librarian_get_companies
- librarian_search_documents
- librarian_batch_download
- librarian_get_document_metadata

## Usage Examples

### Example 1: Find Canadian Companies

```
Please use the Librarian API to find all companies in Canada.
```

Claude will:
1. Use `librarian_get_companies` with `countries=["Canada"]`
2. Display the results with company names, IDs, and tickers

### Example 2: Search for Recent Filings

```
Find all filings from the last 6 months for Canadian companies.
```

Claude will:
1. Get Canadian companies
2. Use `librarian_search_documents` with those company IDs
3. Filter by `document_type=["filing"]` and date range
4. Display the results

### Example 3: Download Documents

```
Get download URLs for documents 101356 and 101357.
```

Claude will:
1. Use `librarian_batch_download` with those document IDs
2. Return presigned S3 URLs (valid for 1 hour)
3. Provide instructions for downloading

### Example 4: Access Processed Documents

```
Show me the content of processed document 101356.
```

Claude will:
1. Access resource `librarian://processed/101356.md`
2. Display the markdown content
3. Can analyze or summarize the content

## API Reference

### Tool Schemas

All tools accept JSON parameters and return JSON strings.

**librarian_get_companies**
```typescript
{
  countries?: string[],  // Optional country filter
  page?: number,         // Default: 1
  page_size?: number     // Default: 100, max: 250
}
```

**librarian_search_documents**
```typescript
{
  company_ids?: number[],
  document_types?: string[],     // ["filing", "slides"]
  filing_date_start?: string,    // "YYYY-MM-DD"
  filing_date_end?: string,      // "YYYY-MM-DD"
  page?: number,                 // Default: 1
  page_size?: number             // Default: 250
}
```

**librarian_batch_download**
```typescript
{
  document_ids: number[],        // Required, max 250
  representation_type?: string   // "raw" | "clean" | "clean_full"
}
```

**librarian_get_document_metadata**
```typescript
{
  document_id: number            // Required
}
```

### Response Formats

All tools return formatted JSON strings with:
- Summary information
- Pagination details (where applicable)
- Relevant data arrays
- Error messages (on failure)

## Troubleshooting

### Issue: Claude Desktop doesn't see the tools

1. Check the config file path is correct
2. Verify the Python path in the config matches your installation
3. Restart Claude Desktop completely
4. Check logs in Claude Desktop (Help → Debug)

### Issue: API authentication errors

1. Verify the API key is correct
2. Check the environment variable is set
3. Test the API key directly:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://librarian.production.primerapp.com/api/v1/companies/filter
```

### Issue: "Module not found" errors

Ensure dependencies are installed:

```bash
pip install fastmcp httpx
```

### Issue: Server crashes or timeouts

- Check server logs for detailed error messages
- Verify network connectivity to Librarian API
- Increase timeout values if needed (in the code)

## Development

### Running Tests

```bash
# Run the test script
python test_mcp_server.py
```

### Adding New Tools

To add a new tool:

1. Add a method to `LibrarianAPIClient` class
2. Create a new `@mcp.tool()` decorated function
3. Document the tool in this README
4. Add tests to `test_mcp_server.py`

Example:

```python
@mcp.tool()
async def librarian_new_feature(param: str) -> str:
    """
    Description of new feature.

    Args:
        param: Description

    Returns:
        JSON string with results
    """
    try:
        result = await api_client.new_feature_method(param)
        import json
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}"
```

### Adding New Resources

To add a new resource:

```python
@mcp.resource("librarian://new-resource/{param}")
async def get_new_resource(param: str) -> str:
    """
    Description of resource.

    Args:
        param: Parameter description

    Returns:
        Resource content
    """
    # Implementation
    pass
```

## Advanced Usage

### Using with Other MCP Clients

The server can be used with any MCP-compatible client:

```python
from fastmcp import Client

async def main():
    async with Client("librarian_mcp_server.py") as client:
        # List available tools
        tools = await client.list_tools()
        print(tools)

        # Call a tool
        result = await client.call_tool(
            "librarian_get_companies",
            {"countries": ["Canada"]}
        )
        print(result)

# Run
import asyncio
asyncio.run(main())
```

### Exposing via HTTP/SSE

To make the server accessible remotely:

```python
# Modify the last line in librarian_mcp_server.py:
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8080)
```

Then clients can connect via:
```
http://your-server:8080/sse
```

### Server Composition

Mount this server on another MCP server:

```python
from fastmcp import FastMCP

# Create parent server
parent = FastMCP("Parent Server")

# Mount librarian server
parent.mount("librarian", "librarian_mcp_server.py")

# Now all librarian tools are available with "librarian_" prefix
```

## Security Considerations

1. **API Key Storage**: Keep the API key secure, don't commit it to git
2. **Environment Variables**: Use environment variables in production
3. **Network Access**: The server needs internet access to reach Librarian API
4. **Local Files**: The server reads from `processed/` directory only
5. **Rate Limiting**: Respect Librarian API rate limits

## Performance

- **Latency**: ~100-500ms per API call (network dependent)
- **Concurrency**: Supports multiple concurrent requests
- **Caching**: No caching implemented (could be added for companies list)
- **Rate Limits**: Respects Librarian API rate limits

## Future Enhancements

Potential improvements:

- [ ] Add caching layer for frequently accessed data
- [ ] Implement progress tracking for long operations
- [ ] Add webhook support for document processing notifications
- [ ] Create more specialized prompts for common workflows
- [ ] Add support for streaming large documents
- [ ] Implement request queuing for rate limit management
- [ ] Add metrics and monitoring endpoints

## Related Documentation

- [FastMCP Documentation](https://fastmcp.dev)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io)
- [Claude Desktop Setup](https://docs.anthropic.com/claude/docs)
- [Main Pipeline README](./README.md)

## Support

For issues or questions:
1. Check server logs for detailed error messages
2. Review this documentation
3. Test the API directly using curl
4. Check the main pipeline README for API details

## License

[Your License Here]
