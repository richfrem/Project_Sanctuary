"""
Shared fixtures for Gateway cluster integration tests.
Tests direct SSE communication without Gateway intermediary.

This is Tier 2 of the test pyramid:
- Tier 1: Unit Tests (pure Python logic)
- Tier 2: Integration Tests (THIS) - Direct SSE cluster calls
- Tier 3: E2E Tests - Full Gateway RPC flow
"""
import pytest
import httpx
import json
from typing import Dict, Any, List


class SSEClusterClient:
    """
    Direct SSE client for cluster testing.
    Bypasses Gateway to test cluster-level functionality.
    """
    
    def __init__(self, base_url: str, name: str = "unknown"):
        self.base_url = base_url
        self.name = name
        self.client = httpx.Client(timeout=30.0, verify=False)
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call tool directly via cluster's SSE endpoint.
        Bypasses Gateway to test cluster-level functionality.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 1
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/rpc",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "result" in result:
                return {
                    "success": True,
                    "result": result["result"]
                }
            elif "error" in result:
                return {
                    "success": False,
                    "error": result["error"]
                }
            else:
                return {"success": False, "error": "Unexpected response format"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if cluster is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def list_tools(self) -> Dict[str, Any]:
        """List available tools from cluster."""
        try:
            response = self.client.get(f"{self.base_url}/tools")
            response.raise_for_status()
            return {"success": True, "tools": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Cluster port mapping (matches docker-compose.yml)
CLUSTER_PORTS = {
    "sanctuary_utils": 8100,
    "sanctuary_filesystem": 8101,
    "sanctuary_network": 8102,
    "sanctuary_git": 8103,
    "sanctuary_cortex": 8104,
    "sanctuary_domain": 8105,
}


@pytest.fixture(scope="session")
def utils_cluster():
    """Direct SSE client for sanctuary_utils cluster."""
    return SSEClusterClient("http://localhost:8100", "sanctuary_utils")


@pytest.fixture(scope="session")
def filesystem_cluster():
    """Direct SSE client for sanctuary_filesystem cluster."""
    return SSEClusterClient("http://localhost:8101", "sanctuary_filesystem")


@pytest.fixture(scope="session")
def network_cluster():
    """Direct SSE client for sanctuary_network cluster."""
    return SSEClusterClient("http://localhost:8102", "sanctuary_network")


@pytest.fixture(scope="session")
def git_cluster():
    """Direct SSE client for sanctuary_git cluster."""
    return SSEClusterClient("http://localhost:8103", "sanctuary_git")


@pytest.fixture(scope="session")
def cortex_cluster():
    """Direct SSE client for sanctuary_cortex cluster."""
    return SSEClusterClient("http://localhost:8104", "sanctuary_cortex")


@pytest.fixture(scope="session")
def domain_cluster():
    """Direct SSE client for sanctuary_domain cluster."""
    return SSEClusterClient("http://localhost:8105", "sanctuary_domain")


@pytest.fixture(scope="session", autouse=True)
def verify_clusters_running(
    utils_cluster, 
    filesystem_cluster, 
    network_cluster, 
    git_cluster, 
    cortex_cluster, 
    domain_cluster
):
    """Verify all clusters are running before integration tests start."""
    clusters = {
        "sanctuary_utils (8100)": utils_cluster,
        "sanctuary_filesystem (8101)": filesystem_cluster,
        "sanctuary_network (8102)": network_cluster,
        "sanctuary_git (8103)": git_cluster,
        "sanctuary_cortex (8104)": cortex_cluster,
        "sanctuary_domain (8105)": domain_cluster,
    }
    
    failed = []
    for name, cluster in clusters.items():
        if not cluster.health_check():
            failed.append(name)
    
    if failed:
        pytest.fail(
            f"❌ Clusters not running: {', '.join(failed)}\n"
            f"Run: podman compose up -d"
        )
    
    print("\n✅ All clusters healthy and ready for integration tests")
