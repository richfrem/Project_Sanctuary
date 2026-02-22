"""
E2E Tests for sanctuary_filesystem cluster (10 tools)

Tools tested:
- File I/O: code-read, code-write, code-get-info
- Discovery: code-list-files, code-find-file, code-search-content
- Code Quality: code-lint, code-format, code-analyze, code-check-tools
"""
import pytest
from pathlib import Path
from tests.mcp_servers.gateway.e2e.conftest import to_container_path


# Test fixtures directory (local)
TEST_FIXTURES_DIR = Path(__file__).parents[5] / "fixtures" / "test_docs"
SAMPLE_DOC = TEST_FIXTURES_DIR / "sample_document.md"

# Container paths for tools that run inside containers
CONTAINER_TEST_FIXTURES_DIR = to_container_path(TEST_FIXTURES_DIR)
CONTAINER_SAMPLE_DOC = to_container_path(SAMPLE_DOC)


# =============================================================================
# FILE I/O TOOLS (3)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestFileIOTools:
    
    def test_code_read(self, logged_call):
        """Test code-read with test fixture file."""
        result = logged_call("sanctuary-filesystem-code-read", {
            "path": CONTAINER_SAMPLE_DOC
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "Test Document" in content or "RAG" in content, f"Expected fixture content, got: {content[:200]}"
    
    def test_code_write(self, logged_call):
        """Test code-write creates a test file."""
        test_file = CONTAINER_TEST_FIXTURES_DIR + "/e2e_write_test.txt"
        
        result = logged_call("sanctuary-filesystem-code-write", {
            "path": test_file,
            "content": "E2E test content written at test time",
            "create_dirs": True
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_code_get_info(self, logged_call):
        """Test code-get-info returns file metadata."""
        result = logged_call("sanctuary-filesystem-code-get-info", {
            "path": CONTAINER_SAMPLE_DOC
        })
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# DISCOVERY TOOLS (3)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestDiscoveryTools:
    
    def test_code_list_files(self, logged_call):
        """Test code-list-files in test fixtures directory."""
        result = logged_call("sanctuary-filesystem-code-list-files", {
            "path": CONTAINER_TEST_FIXTURES_DIR
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        assert "sample" in content.lower() or "document" in content.lower(), f"Expected fixture files listed"
    
    def test_code_find_file(self, logged_call):
        """Test code-find-file with pattern matching."""
        result = logged_call("sanctuary-filesystem-code-find-file", {
            "name_pattern": "*.md",
            "path": CONTAINER_TEST_FIXTURES_DIR
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_code_search_content(self, logged_call):
        """Test code-search-content for text in files."""
        result = logged_call("sanctuary-filesystem-code-search-content", {
            "query": "RAG"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# CODE QUALITY TOOLS (4)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestCodeQualityTools:
    
    def test_code_lint(self, logged_call):
        """Test code-lint on a Python file."""
        # Use conftest.py as test target  
        conftest_path = to_container_path(Path(__file__).parent / "conftest.py")
        result = logged_call("sanctuary-filesystem-code-lint", {
            "path": conftest_path
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_code_format(self, logged_call):
        """Test code-format in check-only mode."""
        conftest_path = to_container_path(Path(__file__).parent / "conftest.py")
        result = logged_call("sanctuary-filesystem-code-format", {
            "path": conftest_path,
            "check_only": True
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_code_analyze(self, logged_call):
        """Test code-analyze performs static analysis."""
        conftest_path = to_container_path(Path(__file__).parent / "conftest.py")
        result = logged_call("sanctuary-filesystem-code-analyze", {
            "path": conftest_path
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_code_check_tools(self, logged_call):
        """Test code-check-tools returns available tools."""
        result = logged_call("sanctuary-filesystem-code-check-tools", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
