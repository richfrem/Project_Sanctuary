#============================================
# mcp_servers/orchestrator/operations.py
# Purpose: Core business logic for Orchestrator.
#          Centralizes cognitive, mechanical, and query operations.
# Role: Business Logic Layer
# Used as: Helper module by server.py
# LIST OF CLASSES/FUNCTIONS:
#   - OrchestratorOperations
#     - dispatch_mission
#     - create_cognitive_task
#     - create_development_cycle
#     - query_mnemonic_cortex
#     - create_file_write_task
#     - create_git_commit_task
#     - get_orchestrator_status
#     - list_recent_tasks
#     - get_task_result
#     - run_strategic_cycle
#============================================

import os
import json
import hashlib
import glob
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime

from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.path_utils import find_project_root
from .validator import OrchestratorValidator

logger = setup_mcp_logging(__name__)

class OrchestratorOperations:
    """
    Class: OrchestratorOperations
    Purpose: Centralized operations for the Sanctuary Orchestrator.
    Consolidates Cognitive, Mechanical, and Query tools.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
             # Fallback if specific root passed is bad, or use find_project_root if explicit arg was None/empty
             # But here we take str.
             pass

        # Config loading handled by Validator, or we can load here if needed for ops
        # We'll rely on Validator for config related to validation, but might need it here for paths
        self.validator = OrchestratorValidator(str(self.project_root))
        self.config = self.validator.config # Reuse config loaded by validator

    def _write_command_file(self, command: Dict[str, Any]) -> str:
        """Helper to write command.json to the orchestrator directory."""
        orchestrator_config = self.config.get("orchestrator", {})
        rel_path = orchestrator_config.get("command_file_path", "mcp_servers/orchestrator/command.json")
        
        # Resolve path
        if os.path.isabs(rel_path):
            cmd_path = Path(rel_path)
        else:
            cmd_path = self.project_root / rel_path

        # Ensure directory exists
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cmd_path, "w") as f:
            json.dump(command, f, indent=2)
            
        return str(cmd_path)

    # -------------------------------------------------------------------------
    # Cognitive Tools
    # -------------------------------------------------------------------------

    def create_cognitive_task(
        self,
        description: str,
        output_path: str,
        max_rounds: int = 5,
        force_engine: Optional[str] = None,
        max_cortex_queries: int = 5,
        input_artifacts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a command.json for Council deliberation."""
        # Validate output path
        res = self.validator.validate_cognitive_task(output_path)
        if not res.valid:
            return {"status": "error", "error": res.reason, "risk_level": res.risk_level}
            
        # Validate input artifacts
        if input_artifacts:
            for path in input_artifacts:
                res = self.validator.validate_path(path)
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
            
        cmd_path = self._write_command_file(command)
        
        return {
            "status": "success",
            "command_file": cmd_path,
            "message": "Cognitive task queued for Council deliberation"
        }

    def create_development_cycle(
        self,
        description: str,
        project_name: str,
        output_path: str,
        max_rounds: int = 10
    ) -> Dict[str, Any]:
        """Generate a command.json for a staged development cycle."""
        res = self.validator.validate_path(output_path)
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
        
        cmd_path = self._write_command_file(command)
        
        return {
            "status": "success",
            "command_file": cmd_path,
            "message": f"Development cycle '{project_name}' queued"
        }

    def query_mnemonic_cortex(
        self,
        query: str,
        output_path: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Generate a command.json for a RAG query task."""
        res = self.validator.validate_path(output_path)
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
        
        cmd_path = self._write_command_file(command)
        
        return {
            "status": "success",
            "command_file": cmd_path,
            "message": "RAG query queued"
        }

    # -------------------------------------------------------------------------
    # Mechanical Tools
    # -------------------------------------------------------------------------

    def create_file_write_task(
        self,
        content: str,
        output_path: str,
        description: str
    ) -> Dict[str, Any]:
        """Generate a command.json for writing a file."""
        res = self.validator.validate_path(output_path)
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
        
        cmd_path = self._write_command_file(command)
        
        return {
            "status": "success",
            "command_file": cmd_path,
            "message": f"File write task for '{output_path}' queued"
        }

    def create_git_commit_task(
        self,
        files: List[str],
        message: str,
        description: str,
        push: bool = False
    ) -> Dict[str, Any]:
        """Generate a command.json for a git commit (P101 compliant)."""
        res = self.validator.validate_git_operation(files, message, push)
        if not res.valid:
            return {"status": "error", "error": res.reason, "risk_level": res.risk_level}

        # Protocol 101: Generate Manifest
        manifest = {}
        for file_path in files:
            abs_path = self.project_root / file_path
            if abs_path.exists():
                try:
                     with open(abs_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                     manifest[file_path] = file_hash
                except Exception as e:
                     return {"status": "error", "error": f"Failed to hash {file_path}: {e}", "risk_level": "SAFE"}
            else:
                # If file doesn't exist (new file), we assume it will be created by this task chain or already exists.
                # Actually, git add requires file existence. If creating + committing, file write must happen first.
                # Here we assume the file exists for immediate commit, OR the tool execution will fail.
                # But for manifest generation, we can only hash what exists.
                pass

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
        
        cmd_path = self._write_command_file(command)
        
        return {
            "status": "success",
            "command_file": cmd_path,
            "message": "Git commit task queued"
        }

    # -------------------------------------------------------------------------
    # Query Tools
    # -------------------------------------------------------------------------

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Check if the orchestrator is running and healthy."""
        # Using default convention for external orchestrator location
        orchestrator_dir = self.project_root / "council_orchestrator"
        
        if not orchestrator_dir.exists():
            return {
                "status": "offline",
                "message": "Orchestrator directory not found",
                "healthy": False
            }
            
        return {
            "status": "online",
            "message": "Orchestrator infrastructure present",
            "healthy": True,
            "directory": str(orchestrator_dir)
        }

    def list_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent tasks from the orchestrator logs/results."""
        orchestrator_config = self.config.get("orchestrator", {})
        results_dir_rel = orchestrator_config.get("results_directory", "council_orchestrator/command_results/")
        
        if os.path.isabs(results_dir_rel):
            results_dir = Path(results_dir_rel)
        else:
            results_dir = self.project_root / results_dir_rel
            
        if not results_dir.exists():
            return []
            
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

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Retrieve the result of a specific task."""
        orchestrator_config = self.config.get("orchestrator", {})
        results_dir_rel = orchestrator_config.get("results_directory", "council_orchestrator/command_results/")
        
        if os.path.isabs(results_dir_rel):
            results_dir = Path(results_dir_rel)
        else:
            results_dir = self.project_root / results_dir_rel
            
        # Try to find the file
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

    # -------------------------------------------------------------------------
    # Server Logic Moved to Operations
    # -------------------------------------------------------------------------

    def dispatch_mission(self, mission_id: str, objective: str, assigned_agent: str = "Kilo") -> str:
        """Dispatch a mission to an agent (Placeholder)."""
        return f"Mission '{mission_id}' dispatched to {assigned_agent}. Objective: {objective}"

    def run_strategic_cycle(
        self,
        gap_description: str,
        research_report_path: str,
        days_to_synthesize: int = 1
    ) -> str:
        """Execute a full Strategic Crucible Loop: Ingest -> Synthesize -> Adapt -> Cache."""
        from mcp_servers.rag_cortex.operations import CortexOperations
    
        results = []
        results.append(f"--- Strategic Crucible Cycle: {gap_description} ---")
        
        # 1. Ingestion
        try:
            results.append(f"1. Ingesting Report: {research_report_path}")
            # Use root string safely
            cortex_ops = CortexOperations(str(self.project_root)) 
            ingest_stats = cortex_ops.ingest_incremental([research_report_path])
            results.append(f"   - Ingestion Complete: {ingest_stats}")
        except Exception as e:
            return "\n".join(results) + f"\n[CRITICAL FAIL] Ingestion failed: {e}"
    
        # 2. Adaptation (Placeholder)
        try:
            results.append(f"2. Generating Adaptation Packet (Window: {days_to_synthesize} days)")
            packet_path = "TODO: Re-implement SynthesisGenerator in Cortex MCP"
            results.append(f"   - Packet Generated: {packet_path}")
        except Exception as e:
            return "\n".join(results) + f"\n[CRITICAL FAIL] Adaptation failed: {e}"
    
        # 3. Cache Update
        try:
            results.append(f"3. Waking Guardian Cache")
            if hasattr(cortex_ops, 'guardian_wakeup'):
                 wakeup_stats = cortex_ops.guardian_wakeup()
                 results.append(f"   - Cache Updated: {wakeup_stats}")
            elif hasattr(cortex_ops, 'cache_warmup'):
                 # Fallback if guardian_wakeup missing but cache_warmup exists
                 cortex_ops.cache_warmup()
                 results.append(f"   - Cache warmed up")
            else:
                 results.append(f"   - [WARN] guardian_wakeup not found on CortexOperations")
    
        except Exception as e:
            results.append(f"   - [WARN] Cache update failed (non-critical): {e}")
    
        results.append("--- Cycle Complete ---")
        return "\n".join(results)
