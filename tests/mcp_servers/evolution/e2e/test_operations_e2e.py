"""
Evolution MCP E2E Tests - Metric Verification
=============================================

Verifies the self-improvement metrics (Protocol 131) via JSON-RPC.

MCP TOOLS TESTED:
-----------------
| Tool                             | Operation         | Description              |
|----------------------------------|-------------------|--------------------------|
| cortex_evolution_measure_fitness  | calculate_fitness | Depth/Scope Metrics      |
| cortex_evolution_evaluate_depth   | measure_depth     | Depth Metric             |
| cortex_evolution_evaluate_scope   | measure_scope     | Scope Metric             |
"""
import pytest
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

@pytest.mark.e2e
class TestEvolutionE2E(BaseE2ETest):
    SERVER_NAME = "evolution"
    SERVER_MODULE = "mcp_servers.evolution.server"

    def test_evolution_metrics(self, mcp_client):
        """Test Evolution metrics via JSON-RPC."""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        assert "measure_fitness" in names
        
        # 2. Measure Fitness
        test_content = "Technical docs with `code.py`."
        fitness_res = mcp_client.call_tool("measure_fitness", {
            "content": test_content
        })
        # FastMCP returns response["content"] as a list of content items
        assert "content" in fitness_res
        content_item = fitness_res["content"][0]
        assert content_item["type"] == "text"
        
        import json
        metrics = json.loads(content_item["text"])
        assert "depth" in metrics
        assert "scope" in metrics

        # 3. Measure Depth
        depth_res = mcp_client.call_tool("evaluate_depth", {
            "content": test_content
        })
        depth_item = depth_res["content"][0]
        assert float(depth_item["text"]) >= 0
