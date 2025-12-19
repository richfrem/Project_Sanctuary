"""
Gateway Integration Tests: sanctuary-git (Port 8103)
Tests Git workflow operations via Docker endpoint.
"""
import pytest
import httpx
import os

GIT_URL = os.getenv("SANCTUARY_GIT_URL", "http://localhost:8103")


@pytest.mark.asyncio
async def test_git_health():
    """Verify sanctuary-git is healthy and reachable."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{GIT_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
        except httpx.ConnectError:
            pytest.fail(f"Could not connect to sanctuary-git at {GIT_URL}")


@pytest.mark.asyncio
async def test_git_sse_endpoint():
    """Verify SSE endpoint exists and accepts connections."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        async with client.stream("GET", f"{GIT_URL}/sse") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_git_tool_listing():
    """Verify tools/list works via 202 pattern."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{GIT_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_git_get_status():
    """Verify git_get_status tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "git_get_status", "arguments": {}}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{GIT_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_git_log():
    """Verify git_log tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "git_log", "arguments": {"max_count": 5}}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{GIT_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_git_diff():
    """Verify git_diff tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "git_diff", "arguments": {}}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{GIT_URL}/messages", json=payload)
        assert response.status_code == 202


@pytest.mark.asyncio
async def test_git_get_safety_rules():
    """Verify git_get_safety_rules tool execution."""
    payload = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "git_get_safety_rules", "arguments": {}}
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{GIT_URL}/messages", json=payload)
        assert response.status_code == 202
