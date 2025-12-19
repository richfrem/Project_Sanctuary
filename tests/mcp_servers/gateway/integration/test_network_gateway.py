"""
Gateway Integration Tests: sanctuary-network (Port 8102)
Tests HTTP fetching and network tools via Docker endpoint.
"""
import pytest
import httpx
import os

NETWORK_URL = os.getenv("SANCTUARY_NETWORK_URL", "http://localhost:8102")


@pytest.mark.asyncio
async def test_network_health():
    """Verify sanctuary-network is healthy and reachable."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{NETWORK_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
        except httpx.ConnectError:
            pytest.fail(f"Could not connect to sanctuary-network at {NETWORK_URL}")


@pytest.mark.asyncio
async def test_network_sse_endpoint():
    """Verify SSE endpoint exists and accepts connections."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        async with client.stream("GET", f"{NETWORK_URL}/sse") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_network_tool_listing():
    """Verify tools/list works via 202 pattern."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{NETWORK_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_tools_fetch_url():
    """Verify fetch_url tool execution (202 Accepted)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "fetch_url",
            "arguments": {"url": "https://httpbin.org/get"}
        }
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{NETWORK_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_tools_check_site_status():
    """Verify check_site_status tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "check_site_status",
            "arguments": {"url": "https://example.com"}
        }
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{NETWORK_URL}/messages", json=payload)
        assert response.status_code == 202
