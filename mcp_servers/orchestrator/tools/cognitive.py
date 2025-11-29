import os
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from .safety import SafetyValidator
from .utils import write_command_file

# _write_command_file removed (using utils.write_command_file)

def create_cognitive_task(
    description: str,
    output_path: str,
    max_rounds: int = 5,
    force_engine: Optional[str] = None,
    max_cortex_queries: int = 5,
    input_artifacts: Optional[List[str]] = None,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Generate a command.json for Council deliberation.
    
    Args:
        description: High-level task description
        output_path: Where to save the result
        max_rounds: Maximum deliberation rounds (default: 5)
        force_engine: Force specific engine (gemini/openai/ollama)
        max_cortex_queries: Max RAG queries (default: 5)
        input_artifacts: Optional list of input file paths
        project_root: Root of the project
        config: MCP configuration
    """
    validator = SafetyValidator(project_root)
    
    # Validate output path
    res = validator.validate_cognitive_task(output_path)
    if not res.valid:
        return {"status": "error", "error": res.reason, "risk_level": res.risk_level}
        
    # Validate input artifacts
    if input_artifacts:
        for path in input_artifacts:
            res = validator.validate_path(path)
            if not res.valid:
                 return {"status": "error", "error": f"Invalid input artifact: {res.reason}", "risk_level": res.risk_level}

    command = {
        "task_description": description,
        "output_artifact_path": output_path,
        "config": {
            "max_rounds": max_rounds,
            "max_cortex_queries": max_cortex_queries
        }
    }
    
    if force_engine:
        command["config"]["force_engine"] = force_engine
        
    if input_artifacts:
        command["input_artifacts"] = input_artifacts
        
    cmd_path = write_command_file(command, project_root, config)
    
    return {
        "status": "success",
        "command_file": cmd_path,
        "message": "Cognitive task queued for Council deliberation"
    }

def create_development_cycle(
    description: str,
    project_name: str,
    output_path: str,
    max_rounds: int = 10,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Generate a command.json for a staged development cycle.
    """
    validator = SafetyValidator(project_root)
    res = validator.validate_path(output_path)
    if not res.valid:
        return {"status": "error", "error": res.reason, "risk_level": res.risk_level}

    command = {
        "task_description": description,
        "task_type": "development_cycle",
        "project_name": project_name,
        "output_artifact_path": output_path,
        "config": {
            "max_rounds": max_rounds
        }
    }
    
    cmd_path = write_command_file(command, project_root, config)
    
    return {
        "status": "success",
        "command_file": cmd_path,
        "message": f"Development cycle '{project_name}' queued"
    }

def query_mnemonic_cortex(
    query: str,
    output_path: str,
    max_results: int = 5,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Generate a command.json for a RAG query task.
    """
    validator = SafetyValidator(project_root)
    res = validator.validate_path(output_path)
    if not res.valid:
        return {"status": "error", "error": res.reason, "risk_level": res.risk_level}

    command = {
        "task_description": f"Query Cortex: {query}",
        "task_type": "rag_query",
        "query": query,
        "output_artifact_path": output_path,
        "config": {
            "max_results": max_results
        }
    }
    
    cmd_path = write_command_file(command, project_root, config)
    
    return {
        "status": "success",
        "command_file": cmd_path,
        "message": "RAG query queued"
    }
