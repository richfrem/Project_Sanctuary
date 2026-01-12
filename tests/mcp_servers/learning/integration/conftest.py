"""
Pytest configuration for RAG Cortex MCP integration tests.
"""
import pytest
import tempfile
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def temp_project_root():
    """Create a temporary project root for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal directory structure
        root = Path(tmpdir)
        (root / "mcp_servers" / "learning" / "scripts").mkdir(parents=True)
        (root / "data" / "soul").mkdir(parents=True)
        
        yield root
