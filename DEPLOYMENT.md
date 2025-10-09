# Librarian MCP Server - Deployment Guide

Complete guide for deploying the Librarian MCP Server on Render and using it with Claude Desktop or other MCP clients.

## Table of Contents

- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Render Deployment](#render-deployment)
- [Claude Desktop Integration](#claude-desktop-integration)
- [Remote Access](#remote-access)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- Render account (for cloud deployment)
- Claude Desktop (for local integration)

---

## Local Development

### 1. Setup Environment

```bash
# Clone repository
cd /Users/chrismingard/Kernel/PDF

# Create virtual environment
python3 -m venv venv-mcp
source venv-mcp/bin/activate  # On Windows: venv-mcp\Scripts\activate

# Install dependencies
pip install -r requirements-mcp.txt
```

### 2. Configure API Key

```bash
# Set environment variable
export LIBRARIAN_API_KEY="8fc83f71-23e1-40bd-93a4-4e0a47cdcb44"

# Or create .env file
echo 'LIBRARIAN_API_KEY=8fc83f71-23e1-40bd-93a4-4e0a47cdcb44' > .env.mcp
source .env.mcp
```

### 3. Test the Server

```bash
# Quick smoke test
python test_mcp_server.py

# Full test suite
python test_mcp_server.py --full
```

Expected output:
```
============================================================
QUICK SMOKE TEST
============================================================

Testing API connectivity...
✓ API connection successful!
  Message: Successfully retrieved paginated companies
  Found companies: 1

✓ Smoke test passed! Server is ready to use.
```

### 4. Run Server Locally

#### stdio mode (for Claude Desktop):
```bash
python librarian_mcp_server.py
```

#### HTTP/SSE mode (for testing):
```bash
python librarian_mcp_server.py --http

# Server will start on http://localhost:8080
# SSE endpoint: http://localhost:8080/sse
```

---

## Render Deployment

Deploy the MCP server to Render for 24/7 availability and remote access.

### Option 1: Deploy via Dashboard (Recommended)

1. **Create New Web Service**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   ```
   Name: librarian-mcp-server
   Region: Oregon (or closest to you)
   Branch: main
   Runtime: Python 3
   Build Command: pip install -r requirements-mcp.txt
   Start Command: python librarian_mcp_server.py --http
   ```

3. **Set Environment Variables**
   ```
   LIBRARIAN_API_KEY = 8fc83f71-23e1-40bd-93a4-4e0a47cdcb44
   PORT = 10000
   HOST = 0.0.0.0
   USE_HTTP = true
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (~2-5 minutes)
   - Note your service URL: `https://librarian-mcp-server.onrender.com`

### Option 2: Deploy via Blueprint (render.yaml)

The repository includes a `render.yaml` file for automated deployment.

1. **Connect Repository**
   - Go to Render Dashboard
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository containing `render.yaml`

2. **Configure Environment Variables**
   - During setup, you'll be prompted for `LIBRARIAN_API_KEY`
   - Enter: `8fc83f71-23e1-40bd-93a4-4e0a47cdcb44`

3. **Deploy**
   - Click "Apply"
   - Service will deploy automatically

### Option 3: Deploy via CLI

```bash
# Install Render CLI
brew install render  # macOS
# or download from https://render.com/docs/cli

# Login
render login

# Deploy from blueprint
render blueprint launch
```

### Verify Deployment

Once deployed, test the service:

```bash
# Replace with your actual Render URL
export RENDER_URL="https://librarian-mcp-server.onrender.com"

# Test health check
curl $RENDER_URL/health

# Expected response:
# {"status":"healthy","service":"Librarian MCP Server","api_base":"https://librarian.production.primerapp.com/api/v1"}
```

### SSE Endpoint

Your MCP server will be available at:
```
https://librarian-mcp-server.onrender.com/sse
```

---

## Claude Desktop Integration

### Local Server (stdio)

1. **Locate Claude Desktop Config**

   **macOS:**
   ```bash
   open ~/Library/Application\ Support/Claude/
   ```

   **Windows:**
   ```
   %APPDATA%\Claude\
   ```

   **Linux:**
   ```bash
   ~/.config/Claude/
   ```

2. **Edit claude_desktop_config.json**

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

   **Important:** Update the path in `args` to match your installation directory.

3. **Restart Claude Desktop**

   - Quit Claude Desktop completely (Cmd+Q on macOS)
   - Reopen Claude Desktop
   - The Librarian tools should now be available

4. **Verify Integration**

   In Claude Desktop, type:
   ```
   What Librarian tools are available?
   ```

   Claude should list:
   - librarian_get_companies
   - librarian_search_documents
   - librarian_batch_download
   - librarian_get_document_metadata
   - health_check

### Remote Server (via Render)

For using the deployed Render server with Claude Desktop:

```json
{
  "mcpServers": {
    "librarian": {
      "url": "https://librarian-mcp-server.onrender.com/sse",
      "transport": "sse"
    }
  }
}
```

**Note:** As of January 2025, Claude Desktop has limited support for remote SSE servers. For best results, use local stdio mode.

---

## Remote Access

### Using with MCP Clients

Connect any MCP-compatible client to your deployed server:

```python
from fastmcp import Client

async def main():
    # Connect to remote server
    async with Client("https://librarian-mcp-server.onrender.com/sse") as client:
        # List available tools
        tools = await client.list_tools()
        print("Available tools:", tools)

        # Call a tool
        result = await client.call_tool(
            "librarian_get_companies",
            {"countries": ["Canada"], "page": 1, "page_size": 5}
        )
        print("Result:", result)

import asyncio
asyncio.run(main())
```

### Using with AI SDKs

#### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic()

# Use MCP tools via Claude API
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[
        {
            "name": "librarian_get_companies",
            "description": "Fetch companies from Librarian API",
            "input_schema": {
                "type": "object",
                "properties": {
                    "countries": {"type": "array", "items": {"type": "string"}},
                    "page": {"type": "integer"},
                    "page_size": {"type": "integer"}
                }
            }
        }
    ],
    messages=[
        {"role": "user", "content": "Get Canadian companies"}
    ]
)
```

#### LangChain Integration

```python
from langchain.tools import Tool
import httpx
import json

async def call_librarian_tool(tool_name: str, **kwargs):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://librarian-mcp-server.onrender.com/tools/call",
            json={"tool": tool_name, "arguments": kwargs}
        )
        return response.json()

librarian_companies_tool = Tool(
    name="librarian_get_companies",
    func=lambda **kwargs: call_librarian_tool("librarian_get_companies", **kwargs),
    description="Fetch companies from Librarian API"
)
```

---

## Troubleshooting

### Issue: "Module not found: fastmcp"

```bash
# Ensure dependencies are installed
pip install -r requirements-mcp.txt

# Or install directly
pip install fastmcp httpx
```

### Issue: API Authentication Failed

```bash
# Verify API key is set
echo $LIBRARIAN_API_KEY

# Test API directly
curl -H "Authorization: Bearer 8fc83f71-23e1-40bd-93a4-4e0a47cdcb44" \
  https://librarian.production.primerapp.com/api/v1/companies/filter \
  -X POST -H "Content-Type: application/json" \
  -d '{"page":1,"page_size":1}'
```

### Issue: Claude Desktop doesn't see tools

1. Check config file syntax (must be valid JSON)
2. Verify file path in config is correct
3. Restart Claude Desktop completely (Cmd+Q, not just close window)
4. Check Claude Desktop logs:
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

### Issue: Render deployment fails

1. **Check build logs** in Render Dashboard
2. **Verify requirements-mcp.txt** exists and has correct dependencies
3. **Check environment variables** are set correctly
4. **Verify start command**: `python librarian_mcp_server.py --http`

### Issue: Server timeout on Render

Render free tier has a 15-minute timeout for inactive services. The server will automatically restart on the next request.

**Solution:** Use a paid plan or implement a keep-alive ping:

```bash
# Ping server every 10 minutes to keep it active
*/10 * * * * curl https://librarian-mcp-server.onrender.com/health
```

### Issue: "Connection refused" when testing

```bash
# Check if server is running
curl http://localhost:8080/health

# If not running, start server
python librarian_mcp_server.py --http
```

### Issue: Tools return "Error: ..."

Check server logs for detailed error messages:

```bash
# Local development
python librarian_mcp_server.py --http
# Watch for error messages in console

# Render deployment
# View logs in Render Dashboard → Your Service → Logs
```

---

## Monitoring & Maintenance

### Health Checks

The server includes a health check tool:

```bash
# Test health
curl https://librarian-mcp-server.onrender.com/health

# Or call via MCP
# In Claude: "Use the health_check tool"
```

### Logs

**Local:**
```bash
# Server logs to console
python librarian_mcp_server.py --http
```

**Render:**
- View logs in Render Dashboard
- Click your service → "Logs" tab
- Real-time log streaming

### Updating the Server

**Local:**
```bash
git pull
pip install -r requirements-mcp.txt --upgrade
python librarian_mcp_server.py
```

**Render:**
- Push changes to GitHub
- Render will automatically redeploy
- Or manually trigger deploy in Dashboard

---

## Performance & Scaling

### Free Tier Limits (Render)

- 750 hours/month of runtime
- 512 MB RAM
- Shared CPU
- 15-minute timeout when inactive

### Paid Tier Benefits

- **Starter ($7/mo):** 512 MB RAM, no timeout
- **Standard ($25/mo):** 2 GB RAM, better performance
- **Pro ($85/mo):** 4 GB RAM, dedicated resources

### Caching Strategy

To improve performance, consider adding caching:

```python
from functools import lru_cache
import time

# Cache companies list for 1 hour
@lru_cache(maxsize=128)
def get_cached_companies(country: str, timestamp: int):
    # timestamp ensures cache expires after 1 hour
    return api_client.get_companies(countries=[country])

# Use in tool
companies = get_cached_companies("Canada", int(time.time() / 3600))
```

---

## Security Best Practices

1. **API Key Management**
   - Use environment variables (not hardcoded)
   - Rotate keys periodically
   - Never commit keys to git

2. **Network Security**
   - Use HTTPS only (Render provides this automatically)
   - Consider adding authentication to your MCP server
   - Limit access by IP if possible

3. **Data Privacy**
   - Don't log sensitive data
   - Clear processed files regularly
   - Monitor API usage

---

## Support & Resources

- **MCP Server Code:** `librarian_mcp_server.py`
- **Test Suite:** `test_mcp_server.py`
- **Server README:** `README_MCP_SERVER.md`
- **FastMCP Docs:** https://fastmcp.dev
- **Render Docs:** https://render.com/docs
- **Claude Desktop:** https://claude.ai/download

---

## Next Steps

1. ✅ Deploy server to Render
2. ✅ Configure Claude Desktop
3. ✅ Test all tools
4. Try example workflows (see README_MCP_SERVER.md)
5. Integrate with your applications
6. Set up monitoring and alerts

---

**Deployment Complete! 🎉**

Your Librarian MCP Server is now ready to use with Claude Desktop and other MCP clients.
