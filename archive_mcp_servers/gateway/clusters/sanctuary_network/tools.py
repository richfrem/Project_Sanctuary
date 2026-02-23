"""
Network tools implementation (Business Logic).
Separated from the FastMCP server interface.
"""
import httpx
from fastmcp.exceptions import ToolError
from mcp_servers.lib.logging_utils import setup_mcp_logging

logger = setup_mcp_logging("project_sanctuary.sanctuary_network")

async def fetch_url(url: str) -> str:
    """Fetch content from a URL via HTTP GET."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=10.0, follow_redirects=True)
            # Truncate for safety and context window efficiency
            content = resp.text[:4000]
            return f"Status: {resp.status_code}\nContent:\n{content}{'...' if len(resp.text) > 4000 else ''}"
        except Exception as e:
            logger.error(f"Error in fetch_url for {url}: {e}")
            raise ToolError(f"Fetch failed: {str(e)}")

async def check_site_status(url: str) -> str:
    """Check if a site is up (HEAD request)."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.head(url, timeout=5.0)
            return f"{url} is UP (Status: {resp.status_code})"
        except Exception as e:
            logger.error(f"Error in check_site_status for {url}: {e}")
            raise ToolError(f"Status check failed: {str(e)}")
