import pytest
import os
import json

class TestConfigOperationsIntegration:
    
    def test_write_and_read_json(self, config_ops):
        """Test writing and reading a JSON config file."""
        data = {"key": "value", "number": 123}
        filename = "test_config.json"
        
        # Write
        path = config_ops.write_config(filename, data)
        assert os.path.exists(path)
        
        # Read
        read_data = config_ops.read_config(filename)
        assert read_data == data

    def test_write_and_read_text(self, config_ops):
        """Test writing and reading a text config file."""
        content = "some configuration text"
        filename = "test.txt"
        
        # Write
        path = config_ops.write_config(filename, content)
        assert os.path.exists(path)
        
        # Read
        read_content = config_ops.read_config(filename)
        assert read_content == content

    def test_list_configs(self, config_ops):
        """Test listing configuration files."""
        config_ops.write_config("config1.json", {"a": 1})
        config_ops.write_config("config2.txt", "text")
        
        configs = config_ops.list_configs()
        assert len(configs) >= 2  # Might have .DS_Store or others? No, in tmp fixture should be clean.
        
        names = [c["name"] for c in configs]
        assert "config1.json" in names
        assert "config2.txt" in names

    def test_delete_config(self, config_ops, config_root):
        """Test deleting a configuration file."""
        filename = "to_delete.json"
        config_ops.write_config(filename, {"data": "temp"})
        assert (config_root / filename).exists()
        
        config_ops.delete_config(filename)
        assert not (config_root / filename).exists()

    def test_backup_creation(self, config_ops, config_root):
        """Test that backups are created when overwriting."""
        filename = "backup_test.json"
        config_ops.write_config(filename, {"version": 1})
        
        # Overwrite
        config_ops.write_config(filename, {"version": 2})
        
        # Check for backup file
        files = list(config_root.iterdir())
        backups = [f for f in files if f.name.endswith(".bak")]
        assert len(backups) > 0
        
        # Verify backup content
        backup_content = json.loads(backups[0].read_text())
        assert backup_content == {"version": 1}
