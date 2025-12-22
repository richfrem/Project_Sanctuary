import pytest
import os
from tests.mcp_servers.gateway.gateway_test_client import GatewayTestClient

#=============================================================================
# TIER 3 (BRIDGE) VERIFICATION: sanctuary_utils
#=============================================================================
# This test suite verifies the Gateway-to-Server bridge for utility tools.
# It uses the centralized GatewayTestClient for RPC execution.
#=============================================================================

@pytest.fixture(scope="module")
def client():
    """Initialize the Gateway modular test client."""
    return GatewayTestClient()

@pytest.mark.integration
@pytest.mark.gateway
def test_utils_gateway_connection(client):
    """Verify the Gateway can reach the sanctuary_utils cluster."""
    assert client.health_check(), "Gateway heartbeat failed."
    
    tools_res = client.list_tools(gateway_slug="sanctuary_utils")
    assert tools_res["success"], f"Failed to list tools for sanctuary_utils: {tools_res.get('error')}"
    assert tools_res["count"] > 0, "No tools discovered for sanctuary_utils."

@pytest.mark.integration
@pytest.mark.gateway
@pytest.mark.parametrize("tool, args", [
    ("sanctuary-utils-calculator-add", {"a": 10, "b": 5}),
    ("sanctuary-utils-time-get-current-time", {}),
    ("sanctuary-utils-uuid-generate-uuid4", {}),
])
def test_utils_rpc_execution(client, tool, args):
    """
    Verify RPC execution for core utility tools via the Gateway bridge.
    Target: Tier 3 (Bridge)
    """
    print(f"\n[RPC] Executing {tool}...")
    res = client.call(tool, args)
    
    assert res["success"], f"RPC Failed for {tool}: {res.get('error')}"
    assert "result" in res, f"No result returned for {tool}"
    
    # Specific result validation for calculator.add
    if "calculator-add" in tool:
        content = res["result"].get("content", [])
        assert any("15" in str(c.get("text", "")) for c in content), f"Math error in {tool}"

@pytest.mark.integration
@pytest.mark.gateway
def test_utils_invalid_tool(client):
    """Verify Gateway handles non-existent tools gracefully."""
    res = client.call("sanctuary-utils-invalid-tool", {})
    assert not res["success"], "Gateway should have rejected invalid tool name."
    assert "error" in res

