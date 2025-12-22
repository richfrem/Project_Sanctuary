import pytest
from tests.mcp_servers.gateway.gateway_test_client import GatewayTestClient

#=============================================================================
# TIER 3 (BRIDGE) VERIFICATION: sanctuary_cortex
#=============================================================================
# This test suite verifies the Gateway-to-Server bridge for cortex logic.
# It uses the centralized GatewayTestClient for RPC execution.
#=============================================================================

@pytest.fixture(scope="module")
def client():
    """Initialize the Gateway modular test client."""
    return GatewayTestClient()

@pytest.mark.integration
@pytest.mark.gateway
def test_cortex_gateway_connection(client):
    """Verify the Gateway can reach the sanctuary_cortex cluster."""
    assert client.health_check(), "Gateway heartbeat failed."
    
    tools_res = client.list_tools(gateway_slug="sanctuary_cortex")
    assert tools_res["success"], f"Failed to list tools for sanctuary_cortex: {tools_res.get('error')}"
    assert tools_res["count"] > 0, "No tools discovered for sanctuary_cortex."

@pytest.mark.integration
@pytest.mark.gateway
@pytest.mark.parametrize("tool, args", [
    ("sanctuary-cortex-cortex-get-stats", {}),
    ("sanctuary-cortex-cortex-cache-stats", {}),
    ("sanctuary-cortex-cortex-query", {"query": "What is Protocol 101?", "max_results": 3}),
    ("sanctuary-cortex-query-sanctuary-model", {"prompt": "What is Project Sanctuary?"}),
])
def test_cortex_rpc_execution(client, tool, args):
    """
    Verify RPC execution for cortex tools (Memory + LLM) via the Gateway bridge.
    Target: Tier 3 (Bridge)
    """
    print(f"\n[RPC] Executing {tool}...")
    res = client.call(tool, args)
    
    assert res["success"], f"RPC Failed for {tool}: {res.get('error')}"
    assert "result" in res, f"No result returned for {tool}"
