#!/usr/bin/env python3
"""
Config Operations
=====================================

Purpose:
    Core business logic for Config Management.
    Handles safe reading/writing of JSON/YAML configuration files.

Layer: Business Logic

Key Classes:
    - ConfigOperations: Main manager
        - __init__(config_dir)
        - list_configs() -> List[ConfigItem]
        - read_config(filename)
        - write_config(filename, content, backup)
        - delete_config(filename)
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Union

# Use shared logging
from mcp_servers.lib.logging_utils import setup_mcp_logging
from .validator import ConfigValidator
from .models import ConfigItem

logger = setup_mcp_logging(__name__)

class ConfigOperations:
    """
    Class: ConfigOperations
    Purpose: Operations for managing configuration files.
    """

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir).resolve()
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
        self.validator = ConfigValidator(self.config_dir)

    #============================================
    # Method: list_configs
    # Purpose: List all configuration files.
    # Returns: List of ConfigItem dictionaries
    #============================================
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all configuration files."""
        configs = []
        if not self.config_dir.exists():
            return configs

        for file_path in self.config_dir.glob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                configs.append(ConfigItem(
                    name=file_path.name,
                    size=file_path.stat().st_size,
                    modified=time.ctime(file_path.stat().st_mtime)
                ).to_dict())
        return sorted(configs, key=lambda x: x["name"])

    #============================================
    # Method: read_config
    # Purpose: Read and parse a configuration file.
    # Args:
    #   filename: Config file name
    # Returns: Parsed dict or raw string
    #============================================
    def read_config(self, filename: str) -> Union[Dict[str, Any], str]:
        """Read a configuration file."""
        file_path = self.validator.validate_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file '{filename}' not found")

        content = file_path.read_text(encoding="utf-8")
        
        # Try to parse based on extension
        ext = file_path.suffix.lower()
        try:
            if ext == ".json":
                return json.loads(content)
            elif ext in [".yaml", ".yml"]:
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    logger.warning("PyYAML not installed, returning raw content for YAML file")
                    return content
            else:
                return content
        except Exception as e:
            raise ValueError(f"Failed to parse '{filename}': {str(e)}")

    #============================================
    # Method: write_config
    # Purpose: Write config file with backup.
    # Args:
    #   filename: Config file name
    #   content: Data to write
    #   backup: Backup flag (default True)
    # Returns: Path to written file
    #============================================
    def write_config(self, filename: str, content: Union[Dict[str, Any], str], backup: bool = True) -> str:
        """Write a configuration file with optional backup."""
        file_path = self.validator.validate_path(filename)
        
        # Create backup if file exists
        if backup and file_path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f"{file_path.suffix}.{timestamp}.bak")
            shutil.copy2(file_path, backup_path)

        # Write content
        if isinstance(content, (dict, list)):
            ext = file_path.suffix.lower()
            if ext == ".json":
                file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
            elif ext in [".yaml", ".yml"]:
                try:
                    import yaml
                    file_path.write_text(yaml.dump(content, default_flow_style=False), encoding="utf-8")
                except ImportError:
                     raise ImportError("PyYAML not installed, cannot serialize dict to YAML")
            else:
                # Default to JSON for unknown extensions if dict provided
                file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        else:
            file_path.write_text(str(content), encoding="utf-8")
            
        return str(file_path)

    #============================================
    # Method: delete_config
    # Purpose: Delete a configuration file.
    # Args:
    #   filename: Config file name
    # Returns: Success boolean
    #============================================
    def delete_config(self, filename: str) -> bool:
        """Delete a configuration file."""
        file_path = self.validator.validate_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file '{filename}' not found")
            
        file_path.unlink()
        return True
