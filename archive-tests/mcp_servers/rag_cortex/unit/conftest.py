"""
Pytest configuration for RAG Cortex MCP tests.
"""
import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import container manager
from mcp_servers.lib.container_manager import ensure_chromadb_running

from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.fixture(scope="session", autouse=True)
def ensure_chromadb():
    """Ensure ChromaDB container is running before tests start."""
    print("\n[Test Setup] Checking ChromaDB service...")
    success, message = ensure_chromadb_running(str(project_root))
    
    if success:
        print(f"[Test Setup] ✓ {message}")
    else:
        print(f"[Test Setup] ✗ {message}")
        pytest.skip("ChromaDB service not available - skipping RAG Cortex tests")
    
    yield
    # Cleanup if needed (container keeps running for now)

@pytest.fixture
def temp_project_root():
    """Create a temporary project root for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal directory structure
        root = Path(tmpdir)
        (root / "mcp_servers" / "rag_cortex" / "scripts").mkdir(parents=True)
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
        mock_huggingface = MagicMock()
        mock_chroma = MagicMock()
        
        sys.modules["langchain.storage"] = mock_storage
        sys.modules["langchain.retrievers"] = mock_retrievers
        sys.modules["langchain_huggingface"] = mock_huggingface
        sys.modules["langchain_chroma"] = mock_chroma
        
        yield
