#============================================
# mcp_servers/config/validator.py
# Purpose: Validation logic for Config Operations.
#          Enforces path security and file validity.
# Role: Safety Layer
# Used as: Helper module by operations.py
# LIST OF CLASSES/FUNCTIONS:
#   - ConfigValidator
#     - __init__
#     - validate_path
#============================================

from pathlib import Path

class ConfigValidator:
    """
    Class: ConfigValidator
    Purpose: Enforces security and safety constraints on config operations.
    """
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir

    #============================================
    # Method: validate_path
    # Purpose: Ensure filename resolves to a path within config directory.
    # Args:
    #   filename: Name of the config file
    # Returns: Resolved Path object
    # Throws: ValueError if path violation
    #============================================
    def validate_path(self, filename: str) -> Path:
        """Validate that the file path is within the config directory."""
        # Resolve the full path
        file_path = (self.config_dir / filename).resolve()
        
        # Check if the resolved path starts with the config directory path
        if not str(file_path).startswith(str(self.config_dir)):
            raise ValueError(f"Security Error: Path '{filename}' is outside config directory")
            
        return file_path
