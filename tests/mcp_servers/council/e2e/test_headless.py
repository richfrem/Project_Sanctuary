import pytest
import importlib
from pathlib import Path

from mcp_servers.rag_cortex.mcp_client import MCPClient


@pytest.mark.headless
def test_council_headless():
    try:
        importlib.import_module("mcp_servers.council.operations")
    except Exception:
        pytest.skip("council.operations not available")

    client = MCPClient(project_root=str(Path(__file__).resolve().parents[4]))
    assert client is not None
