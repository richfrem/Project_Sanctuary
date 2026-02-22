import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def protocol_root(tmp_path):
    """Create a temporary directory for protocol tests."""
    root = tmp_path / "protocol_test_root"
    root.mkdir()
    
    # Create required subdirs
    (root / "01_PROTOCOLS").mkdir()
    
    return str(root)

@pytest.fixture
def mock_project_root(protocol_root):
    """Return the temporary root as the project root."""
    return protocol_root
