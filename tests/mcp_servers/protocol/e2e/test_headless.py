import pytest
from pathlib import Path

from mcp_servers.rag_cortex.mcp_client import MCPClient


@pytest.mark.headless
def test_protocol_list_headless():
    project_root = str(Path(__file__).resolve().parents[4])
    client = MCPClient(project_root=project_root)
    res = client.route_query(scope="Protocols", intent="SUMMARIZE", constraints="", query_data={})
    assert isinstance(res, list)
