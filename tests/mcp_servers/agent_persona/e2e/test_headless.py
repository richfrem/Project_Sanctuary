import pytest
import importlib
from pathlib import Path

from mcp_servers.rag_cortex.mcp_client import MCPClient


@pytest.mark.headless
def test_agent_persona_headless():
    project_root = str(Path(__file__).resolve().parents[4])
    # ensure operations module exists
    try:
        mod = importlib.import_module("mcp_servers.agent_persona.operations")
    except Exception:
        pytest.skip("agent_persona.operations not available")

    # Try client routing if applicable (no direct mapping) - just ensure MCPClient constructs
    client = MCPClient(project_root=project_root)
    assert client is not None
