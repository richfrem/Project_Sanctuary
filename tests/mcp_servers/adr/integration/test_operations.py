"""
ADR MCP Integration Tests - Operations Testing
===============================================

Tests each ADR MCP operation individually against the REAL ADRs directory.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/adr/integration/test_operations.py -v -s

MCP OPERATIONS:
---------------
| Operation         | Type  | Description              |
|-------------------|-------|--------------------------|
| adr_list          | READ  | List real ADRs           |
| adr_get           | READ  | Get real ADR             |
| adr_search        | READ  | Search real ADRs         |
| adr_create        | WRITE | Create then cleanup      |
| adr_update_status | WRITE | Update then cleanup      |
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.adr.operations import ADROperations

# Use REAL ADRs directory
REAL_ADR_DIR = project_root / "ADRs"


@pytest.fixture
def ops():
    """Create ADROperations with REAL ADRs directory."""
    return ADROperations(str(REAL_ADR_DIR))


# =============================================================================
# READ OPERATIONS
# =============================================================================

def test_adr_list(ops):
    """Test adr_list - list real ADRs."""
    result = ops.list_adrs()
    
    print(f"\n📋 adr_list:")
    print(f"   Total: {len(result)} ADRs")
    
    assert isinstance(result, list)
    assert len(result) > 0
    print("✅ PASSED")


def test_adr_get(ops):
    """Test adr_get - get real ADR by number."""
    adrs = ops.list_adrs()
    adr_number = adrs[0]["number"]
    
    result = ops.get_adr(adr_number)
    
    print(f"\n📄 adr_get:")
    print(f"   Got ADR #{result['number']}: {result['title'][:40]}...")
    
    assert result["number"] == adr_number
    print("✅ PASSED")


def test_adr_search(ops):
    """Test adr_search - search real ADRs."""
    result = ops.search_adrs("protocol")
    
    print(f"\n🔍 adr_search:")
    print(f"   Query 'protocol' found {len(result)} ADRs")
    
    assert isinstance(result, list)
    print("✅ PASSED")


# =============================================================================
# WRITE OPERATIONS (create, verify, cleanup)
# =============================================================================

def test_adr_create(ops):
    """Test adr_create - create then cleanup."""
    result = ops.create_adr(
        title="[TEST] adr_create Test",
        context="Testing adr_create operation.",
        decision="Create test ADR.",
        consequences="Auto-cleaned after test.",
        status="proposed",
        author="Integration Test"
    )
    
    print(f"\n🆕 adr_create:")
    print(f"   Created: #{result['adr_number']}")
    
    assert os.path.exists(result['file_path'])
    
    # Cleanup
    os.remove(result['file_path'])
    print(f"   🧹 Cleaned up")
    print("✅ PASSED")


def test_adr_update_status(ops):
    """Test adr_update_status - create, update, cleanup."""
    created = ops.create_adr(
        title="[TEST] adr_update_status Test",
        context="Testing.", decision="Test.", consequences="Auto-cleaned.",
        status="proposed", author="Integration Test"
    )
    
    result = ops.update_adr_status(
        created["adr_number"], "accepted", "Test verification"
    )
    
    print(f"\n🔄 adr_update_status:")
    print(f"   {result['old_status']} → {result['new_status']}")
    
    assert result['new_status'] == "accepted"
    
    # Cleanup
    os.remove(created['file_path'])
    print(f"   🧹 Cleaned up")
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
