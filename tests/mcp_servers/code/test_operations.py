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

if __name__ == "__main__":
    unittest.main()
