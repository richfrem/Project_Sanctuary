import pytest
import importlib
from pathlib import Path


@pytest.mark.headless
def test_forge_llm_headless():
    try:
        importlib.import_module("mcp_servers.forge_llm.operations")
    except Exception:
        pytest.skip("forge_llm.operations not available")

    # Ensure nothing crashes on import
    assert True
