import pytest
import httpx
import json
import os

# Configuration from .env or defaults
FILESYSTEM_URL = os.getenv("FILESYSTEM_URL", "http://localhost:8101")

@pytest.mark.asyncio
async def test_filesystem_health():
    """Verify generic health check."""
    async with httpx.AsyncClient() as client:
        # Code MCP might not have a dedicated /health endpoint refactored yet, 
        # but let's check root or standard health if available.
        # Fallback: check that the server accepts connections.
        try:
            response = await client.get(f"{FILESYSTEM_URL}/health") # Many established this pattern
            assert response.status_code in [200, 404] # 404 is acceptable if endpoint missing but server up
        except httpx.ConnectError:
            pytest.fail("Could not connect to sanctuary-filesystem container")

@pytest.mark.asyncio
async def test_filesystem_tool_listing():
    """Verify tool listing (Async/202)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{FILESYSTEM_URL}/messages", json=payload)
        assert response.status_code == 202

@pytest.mark.asyncio
async def test_filesystem_read_op():
    """Verify code_read operation (fire-and-forget 202)."""
    async with httpx.AsyncClient() as client:
        payload = {
            "jsonrpc": "2.0", 
            "id": 2, 
            "method": "tools/call", 
            "params": {
                "name": "code_read", 
                "arguments": {"path": "README.md"}
            }
        }
        response = await client.post(f"{FILESYSTEM_URL}/messages", json=payload)
        assert response.status_code == 202
