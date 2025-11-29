import os
import json
import glob
from typing import List, Optional, Dict, Any
from pathlib import Path

def get_orchestrator_status(
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Check if the orchestrator is running and healthy.
    """
    # In a real implementation, this might check a PID file or health endpoint.
    # For now, we check if the directory structure exists.
    orchestrator_dir = Path(project_root) / "council_orchestrator"
    
    if not orchestrator_dir.exists():
        return {
            "status": "offline",
            "message": "Orchestrator directory not found",
            "healthy": False
        }
        
    return {
        "status": "online", # Assumed for now
        "message": "Orchestrator infrastructure present",
        "healthy": True,
        "directory": str(orchestrator_dir)
    }

def list_recent_tasks(
    limit: int = 10,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> List[Dict[str, Any]]:
    """
    List recent tasks from the orchestrator logs/results.
    """
    orchestrator_config = config.get("orchestrator", {})
    results_dir_rel = orchestrator_config.get("results_directory", "../council_orchestrator/command_results/")
    
    # Resolve path
    if os.path.isabs(results_dir_rel):
        results_dir = Path(results_dir_rel)
    else:
        results_dir = Path(project_root) / "council_orchestrator" / "command_results"
        
    if not results_dir.exists():
        return []
        
    # Find JSON result files
    files = sorted(results_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    recent_files = files[:limit]
    
    tasks = []
    for f in recent_files:
        try:
            with open(f, "r") as json_file:
                data = json.load(json_file)
                tasks.append({
                    "task_id": f.stem,
                    "timestamp": os.path.getmtime(f),
                    "summary": data.get("summary", "No summary"),
                    "status": data.get("status", "unknown")
                })
        except Exception:
            continue
            
    return tasks

def get_task_result(
    task_id: str,
    project_root: str = ".",
    config: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """
    Retrieve the result of a specific task.
    """
    orchestrator_config = config.get("orchestrator", {})
    results_dir_rel = orchestrator_config.get("results_directory", "../council_orchestrator/command_results/")
    
    if os.path.isabs(results_dir_rel):
        results_dir = Path(results_dir_rel)
    else:
        results_dir = Path(project_root) / "council_orchestrator" / "command_results"
        
    # Try to find the file
    # task_id might be the filename without extension or with
    if task_id.endswith(".json"):
        file_path = results_dir / task_id
    else:
        file_path = results_dir / f"{task_id}.json"
        
    if not file_path.exists():
        return {"status": "error", "error": "Task result not found"}
        
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "error": str(e)}
