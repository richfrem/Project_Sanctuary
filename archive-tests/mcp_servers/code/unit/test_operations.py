"""
Unit tests for Code Operations (Business Logic).
Decoupled from Pydantic Models. Mocks external command execution.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
from mcp_servers.code.operations import CodeOperations

class TestCodeOperations:
    @pytest.fixture
    def setup_ops(self):
        """Setup CodeOperations with temp directory."""
        self.test_dir = tempfile.mkdtemp()
        self.project_root = Path(self.test_dir)
        self.ops = CodeOperations(self.test_dir)
        
        yield
        
        # Teardown
        shutil.rmtree(self.test_dir)

    def test_file_io_operations(self, setup_ops):
        """Test read, write, list, find file operations."""
        # Write
        write_res = self.ops.write_file(
            path="test.py",
            content="print('hello')",
            backup=False
        )
        assert write_res["size"] > 0
        assert (self.project_root / "test.py").exists()
        
        # Read
        content = self.ops.read_file("test.py")
        assert content == "print('hello')"
        
        # List
        files = self.ops.list_files()
        assert len(files) == 1
        assert files[0]["path"] == "test.py"
        
        # Find
        found = self.ops.find_file("*.py")
        assert "test.py" in found

    def test_search_content(self, setup_ops):
        """Test grep-like search."""
        self.ops.write_file("main.py", "def foo():\n    pass")
        self.ops.write_file("utils.py", "def bar():\n    pass")
        
        matches = self.ops.search_content("foo")
        assert len(matches) == 1
        assert matches[0]["file"] == "main.py"
        assert "def foo():" in matches[0]["content"]

    def test_get_file_info(self, setup_ops):
        """Test metadata retrieval."""
        self.ops.write_file("data.json", "{}")
        info = self.ops.get_file_info("data.json")
        
        assert info["path"] == "data.json"
        assert info["language"] == "JSON"
        assert info["size"] == 2

    @patch("subprocess.run")
    def test_lint_mocked(self, mock_subprocess, setup_ops):
        """Test linting with mocked subprocess."""
        # Create dummy file to pass valid path check
        self.ops.write_file("bad.py", "import os")
        
        # Setup mock return
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Found errors"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        result = self.ops.lint("bad.py", tool="ruff")
        
        assert not result["success"]
        assert result["output"] == "Found errors"
        assert result["tool"] == "ruff"
        
        # Verify command arguments
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        assert "ruff" in cmd
        assert "check" in cmd

    @patch("subprocess.run")
    def test_format_code_mocked(self, mock_subprocess, setup_ops):
        """Test formatting with mocked subprocess."""
        self.ops.write_file("ugly.py", "x=1")
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_subprocess.return_value = mock_result
        
        result = self.ops.format_code("ugly.py", tool="black")
        
        assert result["success"]
        assert result["tool"] == "black"
        
        # Verify call
        args, _ = mock_subprocess.call_args
        assert "black" in args[0]

    @patch("subprocess.run")
    def test_analyze_mocked(self, mock_subprocess, setup_ops):
        """Test analysis with mock."""
        self.ops.write_file("code.py", "pass")
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Stats: 100%"
        mock_subprocess.return_value = mock_result
        
        result = self.ops.analyze("code.py")
        
        assert result["success"]
        assert "statistics" in result
        assert result["statistics"] == "Stats: 100%"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
