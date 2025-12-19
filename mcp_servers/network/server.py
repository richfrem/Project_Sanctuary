
"""
Sanctuary Network Server
Domain: sanctuary-network
Port: 8002

Provides network fetching capabilities via SSE/JSON-RPC 202-Deferred pattern.
"""
import os
import sys
import logging
import httpx

# Import SSEServer
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from mcp_servers.lib.sse_adaptor import SSEServer
except ImportError:
    # Use generic relative logic
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib.sse_adaptor import SSEServer

# Initialize
server = SSEServer("sanctuary-network")
app = server.app

# Validated Tool Logic (Simple Implementation)
async def fetch_url(url: str) -> str:
    """
    Fetch content from a URL via HTTP GET.
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=10.0, follow_redirects=True)
            return f"Status: {resp.status_code}\nContent:\n{resp.text[:2000]}..." # Truncate for safety
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

async def check_site_status(url: str) -> str:
    """Check if a site is up (HEAD request)."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.head(url, timeout=5.0)
            return f"{url} is UP (Status: {resp.status_code})"
        except Exception as e:
            return f"{url} seems DOWN or unreachable: {str(e)}"

# Register Tools
server.register_tool("fetch_url", fetch_url)
server.register_tool("check_site_status", check_site_status)

if __name__ == "__main__":
    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Legacy Mode)
    import os
    port_env = os.getenv("PORT")
    transport = "sse" if port_env else "stdio"
    port = int(port_env) if port_env else 8002
    
    server.run(port=port, transport=transport)
