import pytest
from tests.mcp_servers.gateway.gateway_test_client import GatewayTestClient

#=============================================================================
# TIER 3 (BRIDGE) VERIFICATION: sanctuary_network
#=============================================================================
# This test suite verifies the Gateway-to-Server bridge for network logic.
# It uses the centralized GatewayTestClient for RPC execution.
#=============================================================================

@pytest.fixture(scope="module")
def client():
    """Initialize the Gateway modular test client."""
    return GatewayTestClient()

@pytest.mark.integration
@pytest.mark.gateway
def test_network_gateway_connection(client):
    """Verify the Gateway can reach the sanctuary_network cluster."""
    assert client.health_check(), "Gateway heartbeat failed."
    
    tools_res = client.list_tools(gateway_slug="sanctuary_network")
    assert tools_res["success"], f"Failed to list tools for sanctuary_network: {tools_res.get('error')}"
    assert tools_res["count"] > 0, "No tools discovered for sanctuary_network."

@pytest.mark.integration
@pytest.mark.gateway
@pytest.mark.parametrize("tool, args", [
    ("sanctuary-network-fetch-url", {"url": "https://httpbin.org/get"}),
    ("sanctuary-network-check-site-status", {"url": "https://example.com"}),
])
def test_network_rpc_execution(client, tool, args):
    """
    Verify RPC execution for network tools via the Gateway bridge.
    Target: Tier 3 (Bridge)
    """
    print(f"\n[RPC] Executing {tool}...")
    res = client.call(tool, args)
    
    assert res["success"], f"RPC Failed for {tool}: {res.get('error')}"
    assert "result" in res, f"No result returned for {tool}"
