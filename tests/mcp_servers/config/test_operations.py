import unittest
import shutil
import tempfile
import json
import os
from pathlib import Path
from mcp_servers.lib.config.config_ops import ConfigOperations

class TestConfigOperations(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for config testing
        self.test_dir = tempfile.mkdtemp()
        self.ops = ConfigOperations(self.test_dir)
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_write_and_read_json(self):
        """Test writing and reading a JSON config file."""
        data = {"key": "value", "number": 123}
        filename = "test_config.json"
        
        # Write
        path = self.ops.write_config(filename, data)
        self.assertTrue(os.path.exists(path))
        
        # Read
        read_data = self.ops.read_config(filename)
        self.assertEqual(read_data, data)

    def test_write_and_read_text(self):
        """Test writing and reading a text config file."""
        content = "some configuration text"
        filename = "test.txt"
        
        # Write
        path = self.ops.write_config(filename, content)
        self.assertTrue(os.path.exists(path))
        
        # Read
        read_content = self.ops.read_config(filename)
        self.assertEqual(read_content, content)

    def test_list_configs(self):
        """Test listing configuration files."""
        self.ops.write_config("config1.json", {"a": 1})
        self.ops.write_config("config2.txt", "text")
        
        configs = self.ops.list_configs()
        self.assertEqual(len(configs), 2)
        names = [c["name"] for c in configs]
        self.assertIn("config1.json", names)
        self.assertIn("config2.txt", names)

    def test_delete_config(self):
        """Test deleting a configuration file."""
        filename = "to_delete.json"
        self.ops.write_config(filename, {"data": "temp"})
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, filename)))
        
        self.ops.delete_config(filename)
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, filename)))

    def test_security_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        with self.assertRaises(ValueError) as cm:
            self.ops.read_config("../outside.json")
        self.assertIn("Security Error", str(cm.exception))

    def test_backup_creation(self):
        """Test that backups are created when overwriting."""
        filename = "backup_test.json"
        self.ops.write_config(filename, {"version": 1})
        
        # Wait a moment to ensure timestamp difference if needed, 
        # but usually fast enough. Just overwrite.
        self.ops.write_config(filename, {"version": 2})
        
        # Check for backup file
        files = os.listdir(self.test_dir)
        backups = [f for f in files if f.endswith(".bak")]
        self.assertTrue(len(backups) > 0)

if __name__ == "__main__":
    unittest.main()
