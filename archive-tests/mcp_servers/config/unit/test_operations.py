"""
Unit tests for Config Operations (Business Logic).
Decoupled from Pydantic Models.
"""
import unittest
import tempfile
import shutil
import os
import json
import time
from pathlib import Path
from mcp_servers.config.operations import ConfigOperations

class TestConfigOperations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = ConfigOperations(self.test_dir)
        self.config_dir = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_write_and_read_json(self):
        """Test writing and reading a JSON config."""
        data = {"key": "value", "number": 123}
        
        # Write
        path = self.ops.write_config("test.json", data)
        self.assertTrue(os.path.exists(path))
        
        # Read
        read_data = self.ops.read_config("test.json")
        self.assertEqual(read_data, data)

    def test_write_and_read_text(self):
        """Test writing and reading a text file."""
        content = "KEY=VALUE\n"
        
        path = self.ops.write_config("env.txt", content)
        read_content = self.ops.read_config("env.txt")
        self.assertEqual(read_content, content)

    def test_backup_creation(self):
        """Test that backup is created when overwriting."""
        # Initial write
        self.ops.write_config("config.json", {"v": 1})
        time.sleep(1) # Ensure timestamp diff
        
        # Overwrite
        self.ops.write_config("config.json", {"v": 2}, backup=True)
        
        # Check files
        files = list(self.config_dir.glob("config.json*"))
        self.assertTrue(len(files) >= 2) # Original + Backup
        
        # Verify content is v2
        current = self.ops.read_config("config.json")
        self.assertEqual(current["v"], 2)

    def test_list_configs(self):
        """Test listing files."""
        self.ops.write_config("a.json", {})
        self.ops.write_config("b.txt", "")
        
        configs = self.ops.list_configs()
        self.assertEqual(len(configs), 2)
        names = [c["name"] for c in configs]
        self.assertIn("a.json", names)
        self.assertIn("b.txt", names)

    def test_delete_config(self):
        """Test deletion."""
        self.ops.write_config("del.json", {})
        self.assertTrue((self.config_dir / "del.json").exists())
        
        self.ops.delete_config("del.json")
        self.assertFalse((self.config_dir / "del.json").exists())

    def test_read_nonexistent(self):
        """Test error on missing file."""
        with self.assertRaises(FileNotFoundError):
            self.ops.read_config("ghost.json")

if __name__ == "__main__":
    unittest.main()
