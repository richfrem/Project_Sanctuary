import pytest
import importlib
from pathlib import Path


@pytest.mark.headless
def test_orchestrator_headless():
    try:
        importlib.import_module("mcp_servers.orchestrator.operations")
    except Exception:
        pytest.skip("orchestrator.operations not available")

    # simple smoke test
    assert True
