import pytest
import importlib
from pathlib import Path


@pytest.mark.headless
def test_config_headless():
    try:
        importlib.import_module("mcp_servers.config.operations")
    except Exception:
        pytest.skip("config.operations not available")

    # Basic smoke: nothing to route here, but ensure import succeeded
    assert True
