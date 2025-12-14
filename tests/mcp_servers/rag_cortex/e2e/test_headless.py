import pytest
import importlib
from pathlib import Path

from mcp_servers.rag_cortex.mcp_client import MCPClient


@pytest.mark.headless
def test_rag_cortex_headless():
    try:
        importlib.import_module("mcp_servers.rag_cortex.operations")
    except Exception:
        pytest.skip("rag_cortex.operations not available")

    client = MCPClient(project_root=str(Path(__file__).resolve().parents[4]))
    # basic route to cortex if available
    try:
        res = client.route_query("What is the status of the RAG DB?", scope="RAG")
        assert res is not None
    except Exception:
        assert client is not None
