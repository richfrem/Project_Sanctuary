import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

class ConfigOperations:
    """
    Operations for managing configuration files in the .agent/config directory.
    Enforces safety checks:
    1. Path validation (must be within config dir)
    2. Backups on write
    3. JSON/YAML support
    """

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir).resolve()
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, filename: str) -> Path:
        """Validate that the file path is within the config directory."""
        # Resolve the full path
        # We join with config_dir first, then resolve to handle ../ correctly
        file_path = (self.config_dir / filename).resolve()
        
        # Check if the resolved path starts with the config directory path
        if not str(file_path).startswith(str(self.config_dir)):
            raise ValueError(f"Security Error: Path '{filename}' is outside config directory")
            
        return file_path

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all configuration files."""
        configs = []
        if not self.config_dir.exists():
            return configs

        for file_path in self.config_dir.glob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                configs.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": time.ctime(file_path.stat().st_mtime)
                })
        return sorted(configs, key=lambda x: x["name"])

    def read_config(self, filename: str) -> Union[Dict[str, Any], str]:
        """Read a configuration file."""
        file_path = self._validate_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file '{filename}' not found")

        content = file_path.read_text(encoding="utf-8")
        
        # Try to parse based on extension
        ext = file_path.suffix.lower()
        try:
            if ext == ".json":
                return json.loads(content)
            elif ext in [".yaml", ".yml"]:
                # Basic YAML parsing if pyyaml is not available, or just return string
                # For now, let's return the raw content if it's not JSON, 
                # or we can try to import yaml inside the method
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    return content
            else:
                return content
        except Exception as e:
            raise ValueError(f"Failed to parse '{filename}': {str(e)}")

    def write_config(self, filename: str, content: Union[Dict[str, Any], str], backup: bool = True) -> str:
        """Write a configuration file with optional backup."""
        file_path = self._validate_path(filename)
        
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
                     # Fallback to JSON if YAML lib missing but extension is YAML? 
                     # No, better to error or write as string if provided as string
                     raise ImportError("PyYAML not installed, cannot serialize dict to YAML")
            else:
                # Default to JSON for unknown extensions if dict provided
                file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        else:
            file_path.write_text(str(content), encoding="utf-8")
            
        return str(file_path)

    def delete_config(self, filename: str) -> bool:
        """Delete a configuration file."""
        file_path = self._validate_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file '{filename}' not found")
            
        file_path.unlink()
        return True
