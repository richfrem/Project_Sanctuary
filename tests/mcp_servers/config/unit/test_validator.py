import pytest
from mcp_servers.config.config_ops import ConfigOperations

class TestConfigValidator:
    def test_security_path_traversal(self, config_root):
        """Test that path traversal attempts are blocked."""
        ops = ConfigOperations(config_root)
        
        with pytest.raises(ValueError) as excinfo:
            ops.read_config("../outside.json")
        
        assert "Security Error" in str(excinfo.value)

    def test_valid_path(self, config_root):
        """Test valid path resolution."""
        ops = ConfigOperations(config_root)
        path = ops._validate_path("valid.json")
        assert path == config_root / "valid.json"
