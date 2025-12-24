import pytest
from tests.mcp_servers.gateway.gateway_test_client import GatewayTestClient

#=============================================================================
# TIER 3 (BRIDGE) VERIFICATION: sanctuary_domain
#=============================================================================
# This test suite verifies the Gateway-to-Server bridge for domain logic.
# It uses the centralized GatewayTestClient for RPC execution.
#=============================================================================

@pytest.fixture(scope="module")
def client():
    """Initialize the Gateway modular test client."""
    return GatewayTestClient()

@pytest.mark.integration
@pytest.mark.gateway
def test_domain_gateway_connection(client):
    """Verify the Gateway can reach the sanctuary_domain cluster."""
    assert client.health_check(), "Gateway heartbeat failed."
    
    tools_res = client.list_tools(gateway_slug="sanctuary_domain")
    assert tools_res["success"], f"Failed to list tools for sanctuary_domain: {tools_res.get('error')}"
    assert tools_res["count"] > 0, "No tools discovered for sanctuary_domain."

@pytest.mark.integration
@pytest.mark.gateway
@pytest.mark.parametrize("tool, args", [
    ("sanctuary-domain-adr-list", {}),
    ("sanctuary-domain-adr-search", {"query": "Protocol"}),
    ("sanctuary-domain-chronicle-list-entries", {"limit": 1}),
    ("sanctuary-domain-protocol-list", {}),
    ("sanctuary-domain-list-tasks", {"status": "in-progress"}),
    ("sanctuary-domain-persona-list-roles", {}),
    ("sanctuary-domain-config-list", {}),
])
def test_domain_rpc_execution(client, tool, args):
    """
    Verify RPC execution for core domain tools via the Gateway bridge.
    Target: Tier 3 (Bridge)
    """
    print(f"\n[RPC] Executing {tool}...")
    res = client.call(tool, args)
    
    assert res["success"], f"RPC Failed for {tool}: {res.get('error')}"
    assert "result" in res, f"No result returned for {tool}"
