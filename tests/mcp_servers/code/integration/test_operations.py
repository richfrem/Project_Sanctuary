"""
Code MCP Integration Tests - Operations Testing
================================================

Tests each Code MCP operation against the REAL project codebase.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/code/integration/test_operations.py -v -s

MCP OPERATIONS:
---------------
| Operation           | Type  | Description                    |
|---------------------|-------|--------------------------------|
| code_list_files     | READ  | List files in directory        |
| code_find_file      | READ  | Find file by pattern           |
| code_read           | READ  | Read file contents             |
| code_get_info       | READ  | Get file metadata              |
| code_search_content | READ  | Search content in files        |
| code_analyze        | READ  | Static analysis                |
| code_check_tools    | READ  | Check available tools          |
| code_lint           | READ  | Lint code                      |
| code_format         | READ  | Format check (check_only=True) |
| code_write          | WRITE | Write file then cleanup        |
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.code.code_ops import CodeOperations

# Use project root for real file operations
ops = CodeOperations(str(project_root))


@pytest.fixture
def code_ops():
    """Create CodeOperations with project root."""
    return CodeOperations(str(project_root))


# =============================================================================
# READ OPERATIONS
# =============================================================================

def test_code_list_files(code_ops):
    """Test code_list_files - list real files."""
    result = code_ops.list_files("mcp_servers", pattern="*.py")
    
    print(f"\n📋 code_list_files:")
    print(f"   Found: {len(result)} .py files in mcp_servers/")
    
    assert isinstance(result, list)
    assert len(result) > 0
    print("✅ PASSED")


def test_code_find_file(code_ops):
    """Test code_find_file - find file by pattern."""
    result = code_ops.find_file("server.py", "mcp_servers")
    
    print(f"\n🔍 code_find_file:")
    print(f"   Found: {len(result)} server.py files")
    
    assert isinstance(result, list)
    assert len(result) > 0
    print("✅ PASSED")


def test_code_read(code_ops):
    """Test code_read - read real file."""
    result = code_ops.read_file("README.md")
    
    print(f"\n📄 code_read:")
    print(f"   Read README.md: {len(result)} chars")
    
    assert len(result) > 0
    assert "Sanctuary" in result or "Project" in result
    print("✅ PASSED")


def test_code_get_info(code_ops):
    """Test code_get_info - get file metadata."""
    result = code_ops.get_file_info("README.md")
    
    print(f"\n📊 code_get_info:")
    print(f"   File: README.md")
    print(f"   Info: {result}")
    
    assert result is not None
    print("✅ PASSED")


def test_code_search_content(code_ops):
    """Test code_search_content - search in files."""
    result = code_ops.search_content("import pytest", file_pattern="*.py")
    
    print(f"\n🔍 code_search_content:")
    print(f"   Query 'import pytest' found in {len(result)} files")
    
    assert isinstance(result, list)
    print("✅ PASSED")


def test_code_analyze(code_ops):
    """Test code_analyze - static analysis."""
    result = code_ops.analyze("mcp_servers/code/code_ops.py")
    
    print(f"\n📈 code_analyze:")
    print(f"   Analyzed: mcp_servers/code/code_ops.py")
    
    assert result is not None
    print("✅ PASSED")




def test_code_lint(code_ops):
    """Test code_lint - lint code."""
    result = code_ops.lint("mcp_servers/code/code_ops.py")
    
    print(f"\n🔍 code_lint:")
    print(f"   Linted: mcp_servers/code/code_ops.py")
    
    assert result is not None
    print("✅ PASSED")


def test_code_format(code_ops):
    """Test code_format - format check only (no changes)."""
    result = code_ops.format_code("mcp_servers/code/code_ops.py", check_only=True)
    
    print(f"\n🎨 code_format (check_only):")
    print(f"   Checked: mcp_servers/code/code_ops.py")
    
    assert result is not None
    print("✅ PASSED")


# =============================================================================
# WRITE OPERATIONS (create, verify, cleanup)
# =============================================================================

def test_code_write(code_ops):
    """Test code_write - write then cleanup."""
    test_file = "tests/mcp_servers/code/integration/test_write_temp.txt"
    test_content = "# Test file for code_write operation\nThis will be deleted."
    
    result = code_ops.write_file(test_file, test_content, backup=False)
    
    print(f"\n📝 code_write:")
    print(f"   Wrote: {test_file}")
    
    # Verify file exists
    full_path = project_root / test_file
    assert full_path.exists()
    
    # Cleanup
    full_path.unlink()
    print(f"   🧹 Cleaned up")
    print("✅ PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
