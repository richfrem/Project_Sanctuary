import pytest
import os

@pytest.fixture
def gateway_url():
    """
    Returns the base URL for the running MCP Gateway container.
    Defaults to http://localhost:4444.
    """
    return os.getenv("MCP_GATEWAY_URL", "http://localhost:4444")
