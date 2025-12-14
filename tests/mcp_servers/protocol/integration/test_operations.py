"""
Protocol MCP Integration Tests - Operations Testing
====================================================

Tests each Protocol MCP operation against the REAL 01_PROTOCOLS directory.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/protocol/integration/test_operations.py -v -s

MCP OPERATIONS:
---------------
| Operation         | Type  | Description               |
|-------------------|-------|---------------------------|
| protocol_list     | READ  | List real protocols       |
| protocol_get      | READ  | Get real protocol         |
| protocol_search   | READ  | Search real protocols     |
| protocol_create   | WRITE | Create then cleanup       |
| protocol_update   | WRITE | Update then cleanup       |
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.protocol.operations import ProtocolOperations

# Use REAL Protocols directory
REAL_PROTOCOL_DIR = project_root / "01_PROTOCOLS"


@pytest.fixture
def ops():
    """Create ProtocolOperations with REAL directory."""
    return ProtocolOperations(str(REAL_PROTOCOL_DIR))


# =============================================================================
# READ OPERATIONS
# =============================================================================

def test_protocol_list(ops):
    """Test protocol_list - list real protocols."""
    result = ops.list_protocols()
    
    print(f"\n📋 protocol_list:")
    print(f"   Total: {len(result)} protocols")
    if result:
        print(f"   Sample: #{result[0]['number']}: {result[0]['title'][:40]}...")
    
    assert isinstance(result, list)
    assert len(result) > 0
    print("✅ PASSED")


def test_protocol_get(ops):
    """Test protocol_get - get real protocol."""
    protocols = ops.list_protocols()
    protocol_number = protocols[0]["number"]
    
    result = ops.get_protocol(protocol_number)
    
    print(f"\n📄 protocol_get:")
    print(f"   Got #{result['number']}: {result['title'][:40]}...")
    
    assert result["number"] == protocol_number
    print("✅ PASSED")


def test_protocol_search(ops):
    """Test protocol_search - search real protocols."""
    searches = ["git", "task", "MCP"]
    
    print(f"\n🔍 protocol_search:")
    for query in searches:
        result = ops.search_protocols(query)
        print(f"   '{query}' → {len(result)} protocols")
    
    assert isinstance(result, list)
    print("✅ PASSED")


# =============================================================================
# WRITE OPERATIONS (create, verify, cleanup)
# =============================================================================

def test_protocol_create(ops):
    """Test protocol_create - create then cleanup."""
    # Use a high number to avoid conflicts
    result = ops.create_protocol(
        number=999,
        title="[TEST] protocol_create Test",
        status="PROPOSED",
        classification="Test Framework",
        version="1.0",
        authority="Integration Test",
        content="Testing protocol_create operation."
    )
    
    print(f"\n🆕 protocol_create:")
    print(f"   Created: #{result['protocol_number']}")
    
    # Cleanup - find and delete the file
    for f in REAL_PROTOCOL_DIR.glob("999_*"):
        f.unlink()
        print(f"   🧹 Cleaned up: {f.name}")
    
    print("✅ PASSED")


def test_protocol_update(ops):
    """Test protocol_update - create, update, cleanup."""
    # Create
    ops.create_protocol(
        number=998,
        title="[TEST] protocol_update Test",
        status="PROPOSED",
        classification="Test",
        version="1.0",
        authority="Integration Test",
        content="Original content."
    )
    
    # Update
    result = ops.update_protocol(
        number=998,
        updates={"status": "CANONICAL"},
        reason="Integration test"
    )
    
    print(f"\n🔄 protocol_update:")
    print(f"   Updated: #{result.get('protocol_number', 998)}")
    
    # Cleanup
    for f in REAL_PROTOCOL_DIR.glob("998_*"):
        f.unlink()
        print(f"   🧹 Cleaned up: {f.name}")
    
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
