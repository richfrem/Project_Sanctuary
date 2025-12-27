"""
Shared fixtures for Gateway cluster integration tests.
Tests direct SSE communication using the MCP Python SDK.

This is Tier 2 of the test pyramid:
- Tier 1: Unit Tests (pure Python logic)
- Tier 2: Integration Tests (THIS) - Direct SSE cluster calls via MCP SDK
- Tier 3: E2E Tests - Full Gateway RPC flow
"""
import pytest
import httpx
from mcp.client.sse import sse_client
from mcp import ClientSession

class SSEClusterClient:
    """
    Direct SSE client for cluster testing using MCP SDK.
    Bypasses Gateway to test cluster-level functionality.
    """
    
    def __init__(self, base_url: str, name: str = "unknown"):
        self.base_url = base_url.rstrip('/')
        self.sse_url = f"{self.base_url}/sse"
        self.name = name
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call tool directly via MCP SDK sse_client.
        """
        async with sse_client(self.sse_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                # The SDK call_tool returns the result directly or raises
                result = await session.call_tool(tool_name, arguments)
                return result

    async def list_tools(self):
        """List available tools via MCP SDK."""
        async with sse_client(self.sse_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                return await session.list_tools()
    
    async def health_check(self) -> bool:
        """Check if cluster is healthy (standard HTTP)."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except:
                return False

# Cluster port mapping
CLUSTER_PORTS = {
    "sanctuary_utils": 8100,
    "sanctuary_filesystem": 8101,
    "sanctuary_network": 8102,
    "sanctuary_git": 8103,
    "sanctuary_cortex": 8104,
    "sanctuary_domain": 8105,
}

# Fixtures return the client directly
# Tests must be marked with @pytest.mark.asyncio

@pytest.fixture(scope="session")
def utils_cluster():
    return SSEClusterClient("http://localhost:8100", "sanctuary_utils")

@pytest.fixture(scope="session")
def filesystem_cluster():
    return SSEClusterClient("http://localhost:8101", "sanctuary_filesystem")

@pytest.fixture(scope="session")
def network_cluster():
    return SSEClusterClient("http://localhost:8102", "sanctuary_network")

@pytest.fixture(scope="session")
def git_cluster():
    return SSEClusterClient("http://localhost:8103", "sanctuary_git")

@pytest.fixture(scope="session")
def cortex_cluster():
    return SSEClusterClient("http://localhost:8104", "sanctuary_cortex")

@pytest.fixture(scope="session")
def domain_cluster():
    return SSEClusterClient("http://localhost:8105", "sanctuary_domain")

@pytest.fixture(scope="session", autouse=True)
async def verify_clusters_running(
    utils_cluster, 
    filesystem_cluster, 
    network_cluster, 
    git_cluster, 
    cortex_cluster, 
    domain_cluster
):
    """Verify all clusters are running before integration tests start."""
    clusters = {
        "sanctuary_utils": utils_cluster,
        "sanctuary_filesystem": filesystem_cluster,
        "sanctuary_network": network_cluster,
        "sanctuary_git": git_cluster,
        "sanctuary_cortex": cortex_cluster,
        "sanctuary_domain": domain_cluster,
    }
    
    # We can't easily wait for all async checks in a session fixture without an event loop
    # simpler to do synchronous check here or rely on individual tests failing.
    # But let's try to do a quick sync check using httpx.Client just for the health verification
    
    failed = []
    with httpx.Client() as client:
        for name, cluster in clusters.items():
            try:
                resp = client.get(f"{cluster.base_url}/health")
                if resp.status_code != 200:
                    failed.append(f"{name} (status {resp.status_code})")
            except Exception as e:
                failed.append(f"{name} ({str(e)})")

    if failed:
        pytest.fail(
            f"❌ Clusters not running/healthy: {', '.join(failed)}\n"
            f"Run: podman compose up -d"
        )
    
    print("\n✅ All clusters healthy and ready for integration tests")
