import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def code_root(tmp_path):
    """Create a temporary directory for Code tests."""
    root = tmp_path / "code_test_root"
    root.mkdir()
    return root

@pytest.fixture
def mock_project_root(code_root):
    """Return the temporary root as the project root."""
    return code_root

@pytest.fixture
def code_ops(code_root):
    """Create CodeOperations instance."""
    from mcp_servers.code.operations import CodeOperations
    ops = CodeOperations(code_root)
    
    # Create a test Python file
    test_file = code_root / "test.py"
    test_file.write_text("""
def hello():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello()
""")
    return ops
