import unittest
import shutil
import tempfile
import os
from pathlib import Path
from mcp_servers.lib.code.code_ops import CodeOperations

class TestCodeOperations(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.ops = CodeOperations(self.test_dir)
        
        # Create a test Python file
        self.test_file = Path(self.test_dir) / "test.py"
        self.test_file.write_text("""
def hello():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello()
""")
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_path_validation(self):
        """Test that path validation blocks traversal attempts."""
        with self.assertRaises(ValueError) as cm:
            self.ops._validate_path("../outside.py")
        self.assertIn("Security Error", str(cm.exception))

    def test_lint_success(self):
        """Test linting a valid Python file."""
        result = self.ops.lint("test.py", tool="ruff")
        self.assertIn("path", result)
        self.assertIn("tool", result)
        self.assertEqual(result["tool"], "ruff")

    def test_lint_nonexistent_file(self):
        """Test linting a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            self.ops.lint("nonexistent.py")

    def test_format_check_only(self):
        """Test format checking without modification."""
        result = self.ops.format_code("test.py", tool="ruff", check_only=True)
        self.assertIn("path", result)
        self.assertIn("tool", result)
        self.assertEqual(result["tool"], "ruff")
        self.assertFalse(result["modified"])

    def test_analyze(self):
        """Test code analysis."""
        result = self.ops.analyze("test.py")
        self.assertIn("path", result)
        self.assertIn("statistics", result)

    def test_check_tool_available(self):
        """Test checking if a tool is available."""
        # 'python3' should always be available
        self.assertTrue(self.ops.check_tool_available("python3"))
        # 'nonexistent_tool_xyz' should not be available
        self.assertFalse(self.ops.check_tool_available("nonexistent_tool_xyz"))

    def test_find_file(self):
        """Test finding files by pattern."""
        # Find the test.py file
        matches = self.ops.find_file("test.py")
        self.assertEqual(len(matches), 1)
        self.assertIn("test.py", matches[0])

    def test_list_files(self):
        """Test listing files in a directory."""
        files = self.ops.list_files(".", "*.py", recursive=False)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], "test.py")

    def test_search_content(self):
        """Test searching for content in files."""
        matches = self.ops.search_content("hello", "*.py")
        self.assertTrue(len(matches) > 0)
        self.assertIn("test.py", matches[0]["file"])

    def test_read_file(self):
        """Test reading a file."""
        content = self.ops.read_file("test.py")
        self.assertIn("def hello", content)
        self.assertIn("print", content)

    def test_write_file(self):
        """Test writing a file with backup."""
        new_content = "# New content\nprint('test')"
        result = self.ops.write_file("test.py", new_content, backup=True)
        
        self.assertEqual(result["path"], "test.py")
        self.assertTrue(result["backup"] is not None)
        self.assertFalse(result["created"])
        
        # Verify content was written
        content = self.ops.read_file("test.py")
        self.assertEqual(content, new_content)

    def test_write_new_file(self):
        """Test creating a new file."""
        new_file = "new_test.py"
        content = "# New file"
        result = self.ops.write_file(new_file, content, backup=True)
        
        self.assertEqual(result["path"], new_file)
        self.assertIsNone(result["backup"])
        self.assertTrue(result["created"])

    def test_get_file_info(self):
        """Test getting file metadata."""
        info = self.ops.get_file_info("test.py")
        
        self.assertEqual(info["path"], "test.py")
        self.assertEqual(info["language"], "Python")
        self.assertGreater(info["size"], 0)
        self.assertGreater(info["lines"], 0)

if __name__ == "__main__":
    unittest.main()
