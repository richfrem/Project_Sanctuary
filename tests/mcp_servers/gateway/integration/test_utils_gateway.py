import pytest
import httpx
import asyncio
import json
import os

# Configuration (Default to Side-by-Side Port 8100)
UTILS_URL = os.getenv("SANCTUARY_UTILS_URL", "http://localhost:8100")

@pytest.mark.asyncio
async def test_utils_health():
    """Verify sanctuary-utils is healthy and reachable."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{UTILS_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
        except httpx.ConnectError:
            pytest.fail(f"Could not connect to sanctuary-utils at {UTILS_URL}. Is the container running?")

@pytest.mark.asyncio
async def test_utils_sse_endpoint():
    """Verify SSE endpoint exists and accepts connections."""
    async with httpx.AsyncClient() as client:
        # Just checking connection, not consuming stream deeply
        async with client.stream("GET", f"{UTILS_URL}/sse") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_utils_tool_listing():
    """Verify generic tool listing (Async/202 pattern)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    async with httpx.AsyncClient() as client:
        # 1. Start listening to SSE (background) - mocked or just separate connection
        # For this simple health check, we just verify the POST accepted
        response = await client.post(f"{UTILS_URL}/messages", json=payload)
        assert response.status_code == 202
        
        # NOTE: To verify the *content* of list, we would need a proper SSE client.
        # For now, 202 ensures the endpoint is reachable and handling JSON-RPC.

@pytest.mark.asyncio
async def test_tools_time_execution():
    """Verify time tools (fire-and-forget for 202)."""
    # We can't easily assert result without SSE listener, but we verify 202 Accepted
    async with httpx.AsyncClient() as client:
        payload = {
            "jsonrpc": "2.0", "id": 2, 
            "method": "tools/call", 
            "params": {"name": "time.get_current_time", "arguments": {}}
        }
        resp = await client.post(f"{UTILS_URL}/messages", json=payload)
        assert resp.status_code == 202

@pytest.mark.asyncio
async def test_tools_calculator_execution():
    """Verify calculator tools."""
    async with httpx.AsyncClient() as client:
        payload = {
            "jsonrpc": "2.0", "id": 3, 
            "method": "tools/call", 
            "params": {"name": "calculator.add", "arguments": {"a": 10, "b": 5}}
        }
        resp = await client.post(f"{UTILS_URL}/messages", json=payload)
        assert resp.status_code == 202

