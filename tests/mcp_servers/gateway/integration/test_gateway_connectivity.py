import pytest
import requests
import os
import time

# Verify Gateway Connectivity
# This test assumes the Gateway container (mcp-gateway) is running.
# It attempts to hit the health/status endpoint.

GATEWAY_URL = os.getenv("MCP_GATEWAY_URL", "http://localhost:4444")

@pytest.mark.integration
@pytest.mark.gateway
def test_gateway_connectivity():
    """
    Verifies that the MCP Gateway is reachable and responding.
    This is a basic 'ping' test to ensure the container is up.
    """
    print(f"\\nTesting Gateway connectivity at: {GATEWAY_URL}")
    
    # We'll try hitting the metric endpoint or root, depending on what ContextForge exposes.
    # ContextForge (FastAPI) usually exposes /docs or /health or /
    
    # Attempt 1: Root
    try:
        response = requests.get(f"{GATEWAY_URL}/", timeout=2)
        print(f"Root endpoint response: {response.status_code}")
        # We don't assert 200 here strictly because auth might block it, 
        # but getting a response (even 401/403/404) means it's ALIVE.
        assert response.status_code in [200, 401, 403, 404, 307], f"Gateway reachable but returned unexpected status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        pytest.fail(f"Could not connect to Gateway at {GATEWAY_URL}. Is the container running?")

    # Attempt 2: SSE Endpoint check (usually /sse)
    # Just checking it exists, not establishing a stream here
    try:
        response = requests.get(f"{GATEWAY_URL}/sse", timeout=2)
        # 405 Method Not Allowed (likely only POST/GET stream) or 200 OK
        # If it returns, the server is up.
        print(f"SSE endpoint response: {response.status_code}")
    except requests.exceptions.ConnectionError:
        pass # Already failed above if down
