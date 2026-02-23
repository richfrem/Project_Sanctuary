import pytest
from mcp_servers.code.operations import CodeOperations

class TestCodeValidator:
    def test_path_validation(self, code_root):
        """Test that path validation blocks traversal attempts."""
        ops = CodeOperations(code_root)
        
        with pytest.raises(ValueError) as excinfo:
            ops.validator.validate_path("../outside.py")
        
        assert "Security Error" in str(excinfo.value)

    def test_path_validation_valid(self, code_root):
        """Test valid path validation."""
        ops = CodeOperations(code_root)
        # Should not raise
        path = ops.validator.validate_path("test.py")
        assert path == code_root / "test.py"
