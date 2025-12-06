import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def adr_root(tmp_path):
    """Create a temporary directory for ADR tests."""
    root = tmp_path / "adr_test_root"
    root.mkdir()
    
    # Create required subdirs
    (root / "docs" / "adr").mkdir(parents=True)
    
    return root

@pytest.fixture
def mock_project_root(adr_root):
    """Return the temporary root as the project root."""
    return adr_root
