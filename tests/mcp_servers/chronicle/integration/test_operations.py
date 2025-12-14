"""
Chronicle MCP Integration Tests - Operations Testing
=====================================================

Tests each Chronicle MCP operation against the REAL 00_CHRONICLE directory.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/chronicle/integration/test_operations.py -v -s

MCP OPERATIONS:
---------------
| Operation                  | Type  | Description               |
|----------------------------|-------|---------------------------|
| chronicle_list_entries     | READ  | List real entries         |
| chronicle_get_entry        | READ  | Get real entry            |
| chronicle_search           | READ  | Search real entries       |
| chronicle_create_entry     | WRITE | Create then cleanup       |
| chronicle_update_entry     | WRITE | Update then cleanup       |
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.chronicle.operations import ChronicleOperations

# Use REAL Chronicle directory
REAL_CHRONICLE_DIR = project_root / "00_CHRONICLE" / "ENTRIES"


@pytest.fixture
def ops():
    """Create ChronicleOperations with REAL directory."""
    return ChronicleOperations(str(REAL_CHRONICLE_DIR))


# =============================================================================
# READ OPERATIONS
# =============================================================================

def test_chronicle_list_entries(ops):
    """Test chronicle_list_entries - list real entries."""
    result = ops.list_entries()
    
    print(f"\n📋 chronicle_list_entries:")
    print(f"   Total: {len(result)} entries")
    if result:
        print(f"   Latest: #{result[0]['number']}: {result[0]['title'][:40]}...")
    
    assert isinstance(result, list)
    assert len(result) > 0
    print("✅ PASSED")


def test_chronicle_get_entry(ops):
    """Test chronicle_get_entry - get real entry."""
    entries = ops.list_entries()
    entry_number = entries[0]["number"]
    
    result = ops.get_entry(entry_number)
    
    print(f"\n📄 chronicle_get_entry:")
    print(f"   Got #{result['number']}: {result['title'][:40]}...")
    
    assert result["number"] == entry_number
    print("✅ PASSED")


def test_chronicle_search(ops):
    """Test chronicle_search - search real entries."""
    # Try different searches
    searches = ["protocol", "MCP", "task"]
    
    print(f"\n🔍 chronicle_search:")
    for query in searches:
        result = ops.search_entries(query)
        print(f"   '{query}' → {len(result)} entries")
    
    assert isinstance(result, list)
    print("✅ PASSED")


# =============================================================================
# WRITE OPERATIONS (create, verify, cleanup)
# =============================================================================

def test_chronicle_create_entry(ops):
    """Test chronicle_create_entry - create then cleanup."""
    result = ops.create_entry(
        title="[TEST] chronicle_create_entry Test",
        content="Testing chronicle_create_entry operation.",
        author="Integration Test",
        status="draft",
        classification="internal"
    )
    
    print(f"\n🆕 chronicle_create_entry:")
    print(f"   Created: #{result['entry_number']}")
    
    assert os.path.exists(result['file_path'])
    
    # Cleanup
    os.remove(result['file_path'])
    print(f"   🧹 Cleaned up")
    print("✅ PASSED")


def test_chronicle_update_entry(ops):
    """Test chronicle_update_entry - create, update, cleanup."""
    created = ops.create_entry(
        title="[TEST] chronicle_update_entry Test",
        content="Original content.",
        author="Integration Test"
    )
    
    result = ops.update_entry(
        entry_number=created['entry_number'],
        updates={"title": "[TEST] Updated Title"},
        reason="Integration test"
    )
    
    print(f"\n🔄 chronicle_update_entry:")
    print(f"   Updated fields: {result['updated_fields']}")
    
    assert "title" in result['updated_fields']
    
    # Cleanup
    os.remove(created['file_path'])
    print(f"   🧹 Cleaned up")
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
