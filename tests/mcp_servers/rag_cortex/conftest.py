import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.fixture
def temp_project_root():
    """Create a temporary project root for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal directory structure
        root = Path(tmpdir)
        (root / "mcp_servers" / "cognitive" / "cortex" / "scripts").mkdir(parents=True)
        (root / "data" / "cortex").mkdir(parents=True)
        
        yield root

@pytest.fixture
def ops(temp_project_root):
    """Create a CortexOperations instance with mocked dependencies."""
    ops = CortexOperations(str(temp_project_root))
    return ops

@pytest.fixture(autouse=True)
def mock_missing_modules():
    """Mock missing langchain modules to allow patching and avoid torch issues."""
    with patch.dict(sys.modules):
        # Create mock modules
        mock_storage = MagicMock()
        mock_retrievers = MagicMock()
        mock_nomic = MagicMock()
        mock_chroma = MagicMock()
        
        sys.modules["langchain.storage"] = mock_storage
        sys.modules["langchain.retrievers"] = mock_retrievers
        sys.modules["langchain_nomic"] = mock_nomic
        sys.modules["langchain_chroma"] = mock_chroma
        
        yield
