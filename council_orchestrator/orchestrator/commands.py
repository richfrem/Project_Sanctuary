# council_orchestrator/orchestrator/commands.py
# Command parsing and validation utilities

import json
from typing import Dict, Any, Optional

def determine_command_type(command: Dict[str, Any]) -> str:
    """Determine the type of command based on its structure."""
    if "entry_content" in command and "output_artifact_path" in command:
        return "MECHANICAL_WRITE"
    elif "git_operations" in command:
        return "MECHANICAL_GIT"
    elif "task_description" in command:
        return "COGNITIVE_TASK"
    elif "development_cycle" in command:
        return "DEVELOPMENT_CYCLE"
    else:
        return "UNKNOWN"

def validate_command(command: Dict[str, Any]) -> tuple[bool, str]:
    """Validate that a command has the required fields for its type."""
    command_type = determine_command_type(command)

    if command_type == "MECHANICAL_WRITE":
        required_fields = ["entry_content", "output_artifact_path"]
        for field in required_fields:
            if field not in command:
                return False, f"Missing required field '{field}' for MECHANICAL_WRITE command"

    elif command_type == "MECHANICAL_GIT":
        if "git_operations" not in command:
            return False, "Missing 'git_operations' field for MECHANICAL_GIT command"

    elif command_type == "COGNITIVE_TASK":
        if "task_description" not in command:
            return False, "Missing 'task_description' field for COGNITIVE_TASK command"

    elif command_type == "DEVELOPMENT_CYCLE":
        if "development_cycle" not in command:
            return False, "Missing 'development_cycle' field for DEVELOPMENT_CYCLE command"

    return True, "Command is valid"

def parse_command_from_json(json_content: str) -> tuple[Optional[Dict[str, Any]], str]:
    """Parse a command from JSON string and validate it."""
    try:
        command = json.loads(json_content)
        is_valid, error_msg = validate_command(command)
        if is_valid:
            return command, determine_command_type(command)
        else:
            return None, f"INVALID_JSON: {error_msg}"
    except json.JSONDecodeError as e:
        return None, f"INVALID_JSON: {str(e)}"