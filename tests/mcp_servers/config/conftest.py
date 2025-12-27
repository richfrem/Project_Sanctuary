import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def config_root(tmp_path):
    """Create a temporary directory for Config tests."""
    root = tmp_path / "config_test_root"
    root.mkdir()
    
    # Create .agent/config directory structure which Config MCP expects
    config_dir = root / ".agent" / "config"
    config_dir.mkdir(parents=True)
    
    return root

@pytest.fixture
def mock_project_root(config_root):
    """Return the temporary root as the project root."""
    return config_root

@pytest.fixture
def config_ops(config_root):
    """Create ConfigOperations instance."""
    from mcp_servers.config.operations import ConfigOperations
    ops = ConfigOperations(config_root)
    return ops
