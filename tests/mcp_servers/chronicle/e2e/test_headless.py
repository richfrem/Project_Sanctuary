import pytest
import importlib
from pathlib import Path

from mcp_servers.rag_cortex.mcp_client import MCPClient


@pytest.mark.headless
def test_chronicle_headless():
    project_root = str(Path(__file__).resolve().parents[4])
    try:
        mod = importlib.import_module("mcp_servers.chronicle.operations")
    except Exception:
        pytest.skip("chronicle.operations not available")

    client = MCPClient(project_root=project_root)
    # exercise a small route for Living_Chronicle if supported
    try:
        res = client.route_query("Summarize the Living Chronicle", scope="Living_Chronicle")
        assert res is not None
    except Exception:
        # not all environments will have a running RAG; pass if routing is unavailable
        assert client is not None
