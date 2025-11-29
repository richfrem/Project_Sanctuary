import os
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from .safety import SafetyValidator
from .utils import write_command_file

def create_file_write_task(
    content: str,
    output_path: str,
    description: str,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Generate a command.json for writing a file.
    """
    validator = SafetyValidator(project_root)
    res = validator.validate_path(output_path)
    if not res.valid:
        return {"status": "error", "error": res.reason, "risk_level": res.risk_level}

    command = {
        "task_description": description,
        "task_type": "file_write",
        "file_operations": {
            "path": output_path,
            "content": content
        }
    }
    
    cmd_path = write_command_file(command, project_root, config)
    
    return {
        "status": "success",
        "command_file": cmd_path,
        "message": f"File write task for '{output_path}' queued"
    }

def create_git_commit_task(
    files: List[str],
    message: str,
    description: str,
    push: bool = False,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Generate a command.json for a git commit (P101 compliant).
    """
    validator = SafetyValidator(project_root)
    res = validator.validate_git_operation(files, message, push)
    if not res.valid:
        return {"status": "error", "error": res.reason, "risk_level": res.risk_level}

    # Protocol 101: Generate Manifest
    manifest = {}
    for file_path in files:
        abs_path = Path(project_root) / file_path
        if abs_path.exists():
            with open(abs_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            manifest[file_path] = file_hash
        else:
            # If file doesn't exist yet (new file), we can't hash it easily unless content was provided elsewhere
            # For now, we assume files exist or will be created by a previous step.
            # If this is a commit of existing files, they must exist.
            return {"status": "error", "error": f"File not found for hashing: {file_path}", "risk_level": "SAFE"}

    command = {
        "task_description": description,
        "task_type": "git_commit",
        "git_operations": {
            "files_to_add": files,
            "commit_message": message,
            "push_to_origin": push,
            "p101_manifest": manifest
        }
    }
    
    cmd_path = write_command_file(command, project_root, config)
    
    return {
        "status": "success",
        "command_file": cmd_path,
        "message": "Git commit task queued"
    }
