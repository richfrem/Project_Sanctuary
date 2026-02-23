import pytest
import shutil
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def chronicle_root(tmp_path):
    """Create a temporary directory for chronicle tests."""
    # Create the structure Chronicle expects
    root = tmp_path / "chronicle_test_root"
    root.mkdir()
    
    # Create required subdirs
    (root / "00_CHRONICLE").mkdir()
    (root / "current_year").mkdir()
    
    return str(root)

@pytest.fixture
def mock_project_root(chronicle_root):
    """Return the temporary root as the project root."""
    return chronicle_root
