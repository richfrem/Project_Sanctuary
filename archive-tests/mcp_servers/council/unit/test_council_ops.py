"""
Tests for Council MCP Server
"""

import pytest
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mcp_servers.council.operations import CouncilOperations

@pytest.fixture
def council_ops():
    """Create CouncilOperations instance"""
    return CouncilOperations()

def test_council_ops_initialization(council_ops):
    """Test that CouncilOperations initializes correctly"""
    assert council_ops.project_root.exists()
    # Should have placeholders for lazy initialization
    assert hasattr(council_ops, "_initialized")
    assert council_ops._initialized is False
    assert council_ops.persona_ops is None
    assert council_ops.cortex is None

def test_list_agents(council_ops):
    """Test listing available agents"""
    # Mock the persona_ops to avoid actual MCP calls during unit tests
    # For now we just check the method exists and has correct signature
    assert hasattr(council_ops, "list_agents")

def test_dispatch_task_structure(council_ops):
    """Test that dispatch_task returns correct structure (without actually running)"""
    # This is a structure test only - we won't actually execute the orchestrator
    # in CI/CD to avoid long-running tests
    
    # Verify the method exists and accepts correct parameters
    assert hasattr(council_ops, "dispatch_task")
    
    # Test parameter validation would go here
    # (actual execution tests should be manual or integration tests)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
