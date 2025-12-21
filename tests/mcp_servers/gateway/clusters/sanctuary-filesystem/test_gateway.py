import pytest
from tests.mcp_servers.gateway.gateway_test_client import GatewayTestClient

#=============================================================================
# TIER 3 (BRIDGE) VERIFICATION: sanctuary_filesystem
#=============================================================================
# This test suite verifies the Gateway-to-Server bridge for filesystem logic.
# It uses the centralized GatewayTestClient for RPC execution.
#=============================================================================

@pytest.fixture(scope="module")
def client():
    """Initialize the Gateway modular test client."""
    return GatewayTestClient()

@pytest.mark.integration
@pytest.mark.gateway
def test_filesystem_gateway_connection(client):
    """Verify the Gateway can reach the sanctuary_filesystem cluster."""
    assert client.health_check(), "Gateway heartbeat failed."
    
    tools_res = client.list_tools(gateway_slug="sanctuary_filesystem")
    assert tools_res["success"], f"Failed to list tools for sanctuary_filesystem: {tools_res.get('error')}"
    assert tools_res["count"] > 0, "No tools discovered for sanctuary_filesystem."

@pytest.mark.integration
@pytest.mark.gateway
@pytest.mark.parametrize("tool, args", [
    ("sanctuary_filesystem-code-list-files", {"path": "."}),
    ("sanctuary_filesystem-code-read", {"path": "README.md"}),
    ("sanctuary_filesystem-code-analyze", {"path": "."}),
    ("sanctuary_filesystem-code-check-tools", {}),
])
def test_filesystem_rpc_execution(client, tool, args):
    """
    Verify RPC execution for filesystem tools via the Gateway bridge.
    Target: Tier 3 (Bridge)
    """
    print(f"\n[RPC] Executing {tool}...")
    res = client.call(tool, args)
    
    assert res["success"], f"RPC Failed for {tool}: {res.get('error')}"
    assert "result" in res, f"No result returned for {tool}"
