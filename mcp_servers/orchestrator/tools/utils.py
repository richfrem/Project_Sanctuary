import os
import json
from typing import Dict, Any
from pathlib import Path

def write_command_file(command: Dict[str, Any], project_root: str, config: Dict[str, Any]) -> str:
    """Helper to write command.json to the orchestrator directory."""
    orchestrator_config = config.get("orchestrator", {})
    rel_path = orchestrator_config.get("command_file_path", "mcp_servers/orchestrator/command.json")
    
    # Resolve absolute path
    if os.path.isabs(rel_path):
        cmd_path = Path(rel_path)
    else:
        # Default to project_root/mcp_servers/orchestrator/command.json
        cmd_path = Path(project_root) / "mcp_servers" / "orchestrator" / "command.json"

    # Ensure directory exists
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cmd_path, "w") as f:
        json.dump(command, f, indent=2)
        
    return str(cmd_path)
