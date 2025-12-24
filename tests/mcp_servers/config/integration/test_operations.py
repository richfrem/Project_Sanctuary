"""
Config MCP Integration Tests - Operations Testing
==================================================

Tests each Config MCP operation against REAL .agent/config directory.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/config/integration/test_operations.py -v -s

MCP OPERATIONS:
---------------
| Operation      | Type  | Description               |
|----------------|-------|---------------------------|
| config_list    | READ  | List config files         |
| config_read    | READ  | Read config file          |
| config_write   | WRITE | Write config then cleanup |
| config_delete  | WRITE | Delete config file        |
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.config.operations import ConfigOperations

# Use REAL config directory
REAL_CONFIG_DIR = project_root / ".agent" / "config"


@pytest.fixture
def ops():
    """Create ConfigOperations with REAL directory."""
    return ConfigOperations(str(REAL_CONFIG_DIR))


# =============================================================================
# READ OPERATIONS
# =============================================================================

def test_config_list(ops):
    """Test config_list - list real config files."""
    result = ops.list_configs()
    
    print(f"\n📋 config_list:")
    print(f"   Found: {len(result)} config files")
    if result:
        print(f"   Sample: {result[0]['name']}")
    
    assert isinstance(result, list)
    print("✅ PASSED")


def test_config_read(ops):
    """Test config_read - read real config file."""
    configs = ops.list_configs()
    if not configs:
        pytest.skip("No config files to read")
    
    filename = configs[0]["name"]
    result = ops.read_config(filename)
    
    print(f"\n📄 config_read:")
    print(f"   Read: {filename}")
    print(f"   Type: {type(result).__name__}")
    
    assert result is not None
    print("✅ PASSED")


# =============================================================================
# WRITE OPERATIONS (create, verify, cleanup)
# =============================================================================

def test_config_write(ops):
    """Test config_write - write then cleanup."""
    test_file = "test_integration_config.json"
    test_content = '{"test": true, "source": "integration_test"}'
    
    ops.write_config(test_file, test_content)
    
    print(f"\n📝 config_write:")
    print(f"   Wrote: {test_file}")
    
    # Verify
    full_path = REAL_CONFIG_DIR / test_file
    assert full_path.exists()
    
    # Cleanup
    full_path.unlink()
    print(f"   🧹 Cleaned up")
    print("✅ PASSED")


def test_config_delete(ops):
    """Test config_delete - create and delete."""
    test_file = "test_delete_config.json"
    
    # Create
    ops.write_config(test_file, '{"to_delete": true}')
    full_path = REAL_CONFIG_DIR / test_file
    assert full_path.exists()
    
    # Delete
    ops.delete_config(test_file)
    
    print(f"\n🗑️ config_delete:")
    print(f"   Created and deleted: {test_file}")
    
    assert not full_path.exists()
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
