import pytest
import shutil

class TestCodeOperationsIntegration:
    
    def test_lint_success(self, code_ops):
        """Test linting a valid Python file."""
        # Only run if ruff is available
        if not shutil.which("ruff"):
            pytest.skip("ruff not installed")
            
        result = code_ops.lint("test.py", tool="ruff")
        assert "path" in result
        assert "tool" in result
        assert result["tool"] == "ruff"

    def test_lint_nonexistent_file(self, code_ops):
        """Test linting a nonexistent file."""
        # Only run if ruff is available
        if not shutil.which("ruff"):
            pytest.skip("ruff not installed")
            
        with pytest.raises(FileNotFoundError):
            code_ops.lint("nonexistent.py")

    def test_format_check_only(self, code_ops):
        """Test format checking without modification."""
        if not shutil.which("ruff"):
            pytest.skip("ruff not installed")
            
        result = code_ops.format_code("test.py", tool="ruff", check_only=True)
        assert "path" in result
        assert "tool" in result
        assert result["tool"] == "ruff"
        assert result["modified"] is False

    def test_analyze(self, code_ops):
        """Test code analysis."""
        result = code_ops.analyze("test.py")
        assert "path" in result
        assert "statistics" in result

    def test_check_tool_available(self, code_ops):
        """Test checking if a tool is available."""
        # 'python3' should always be available
        assert code_ops.check_tool_available("python3")
        # 'nonexistent_tool_xyz' should not be available
        assert not code_ops.check_tool_available("nonexistent_tool_xyz")

    def test_find_file(self, code_ops):
        """Test finding files by pattern."""
        # Find the test.py file
        matches = code_ops.find_file("test.py")
        assert len(matches) == 1
        assert "test.py" in matches[0]

    def test_list_files(self, code_ops):
        """Test listing files in a directory."""
        files = code_ops.list_files(".", "*.py", recursive=False)
        # Should find test.py
        filtered = [f for f in files if f["path"] == "test.py"]
        assert len(filtered) == 1

    def test_search_content(self, code_ops):
        """Test searching for content in files."""
        matches = code_ops.search_content("hello", "*.py")
        assert len(matches) > 0
        assert "test.py" in matches[0]["file"]

    def test_read_file(self, code_ops):
        """Test reading a file."""
        content = code_ops.read_file("test.py")
        assert "def hello" in content
        assert "print" in content

    def test_write_file(self, code_ops):
        """Test writing a file with backup."""
        new_content = "# New content\nprint('test')"
        result = code_ops.write_file("test.py", new_content, backup=True)
        
        assert result["path"] == "test.py"
        assert result["backup"] is not None
        assert not result["created"]
        
        # Verify content was written
        content = code_ops.read_file("test.py")
        assert content == new_content

    def test_write_new_file(self, code_ops):
        """Test creating a new file."""
        new_file = "new_test.py"
        content = "# New file"
        result = code_ops.write_file(new_file, content, backup=True)
        
        assert result["path"] == new_file
        assert result["backup"] is None
        assert result["created"]
        
        # Verify it exists
        assert "new_test.py" in str(code_ops.list_files(".", "new_test.py"))

    def test_get_file_info(self, code_ops):
        """Test getting file metadata."""
        info = code_ops.get_file_info("test.py")
        
        assert info["path"] == "test.py"
        # Language detection might vary, but usually Python for .py
        assert info["language"] == "Python"
        assert info["size"] > 0
        assert info["lines"] > 0
