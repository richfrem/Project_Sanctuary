"""
Gateway Integration Tests: sanctuary-cortex (Port 8104)
Tests RAG operations via Docker endpoint.
"""
import pytest
import httpx
import os

CORTEX_URL = os.getenv("SANCTUARY_CORTEX_URL", "http://localhost:8104")


@pytest.mark.asyncio
async def test_cortex_health():
    """Verify sanctuary-cortex is healthy and reachable."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{CORTEX_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
        except httpx.ConnectError:
            pytest.fail(f"Could not connect to sanctuary-cortex at {CORTEX_URL}")


@pytest.mark.asyncio
async def test_cortex_sse_endpoint():
    """Verify SSE endpoint exists and accepts connections."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        async with client.stream("GET", f"{CORTEX_URL}/sse") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_cortex_tool_listing():
    """Verify tools/list works via 202 pattern."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{CORTEX_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_cortex_get_stats():
    """Verify cortex_get_stats tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "cortex_get_stats", "arguments": {}}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{CORTEX_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_cortex_cache_stats():
    """Verify cortex_cache_stats tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "cortex_cache_stats", "arguments": {}}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{CORTEX_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_cortex_query():
    """Verify cortex_query tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "cortex_query",
            "arguments": {"query": "What is Protocol 101?", "max_results": 3}
        }
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{CORTEX_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_cortex_guardian_wakeup():
    """Verify cortex_guardian_wakeup tool execution (P114)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "cortex_guardian_wakeup", "arguments": {}}
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{CORTEX_URL}/messages", json=payload)
        assert response.status_code == 202
