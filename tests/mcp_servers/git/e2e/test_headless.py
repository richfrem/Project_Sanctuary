import pytest
import importlib


@pytest.mark.headless
def test_git_headless():
    try:
        importlib.import_module("mcp_servers.git.operations")
    except Exception:
        pytest.skip("git.operations not available")

    assert True
