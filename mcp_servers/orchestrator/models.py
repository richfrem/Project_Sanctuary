#!/usr/bin/env python3
"""
Orchestrator Models
=====================================

Purpose:
    Data definition layer for Orchestrator Server.
    Defines schemas for task requests and validation results.

Layer: Data (DTOs)

Key Models:
    # Internal
    - ValidationResult: Safety check outcome

    # MCP Requests
    - OrchestratorDispatchMissionRequest
    - OrchestratorStrategicCycleRequest
    - CreateCognitiveTaskRequest
    - CreateDevelopmentCycleRequest
    - QueryMnemonicCortexRequest
    - CreateFileWriteTaskRequest
    - CreateGitCommitTaskRequest
    - ListRecenttasksRequest
    - GetTaskResultRequest
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    reason: str = ""
    risk_level: str = "SAFE"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "risk_level": self.risk_level
        }

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field
from typing import List, Optional

class OrchestratorDispatchMissionRequest(BaseModel):
    mission_id: str = Field(..., description="Unique mission identifier")
    objective: str = Field(..., description="Core objective of the mission")
    assigned_agent: str = Field("Kilo", description="Agent assigned to the mission (e.g., Kilo, Lima)")

class OrchestratorStrategicCycleRequest(BaseModel):
    gap_description: str = Field(..., description="Description of the strategic gap to be analyzed")
    research_report_path: str = Field(..., description="Path to the research report markdown file")
    days_to_synthesize: int = Field(1, description="Number of days to synthesize")

class CreateCognitiveTaskRequest(BaseModel):
    description: str = Field(..., description="Task description and requirements")
    output_path: str = Field(..., description="Target file path for the command.json")
    max_rounds: int = Field(5, description="Maximum number of deliberation rounds")
    force_engine: Optional[str] = Field(None, description="Force a specific AI engine")
    max_cortex_queries: int = Field(5, description="Maximum allowed Cortex/RAG queries")
    input_artifacts: Optional[List[str]] = Field(None, description="List of relevant artifact file paths")

class CreateDevelopmentCycleRequest(BaseModel):
    description: str = Field(..., description="What to build or refactor")
    project_name: str = Field(..., description="Name of the sub-project or component")
    output_path: str = Field(..., description="Target file path for the command.json")
    max_rounds: int = Field(10, description="Maximum number of rounds for the cycle")

class QueryMnemonicCortexRequest(BaseModel):
    query: str = Field(..., description="The RAG query to execute")
    output_path: str = Field(..., description="Target file path for results")
    max_results: int = Field(5, description="Maximum number of context fragments to retrieve")

class CreateFileWriteTaskRequest(BaseModel):
    content: str = Field(..., description="Full content to be written")
    output_path: str = Field(..., description="Target file path")
    description: str = Field(..., description="Reason for the file write")

class CreateGitCommitTaskRequest(BaseModel):
    files: List[str] = Field(..., description="List of staged files to commit")
    message: str = Field(..., description="Commit message following project standards")
    description: str = Field(..., description="Reason for the commit")
    push: bool = Field(False, description="Whether to push to origin after commit")

class ListRecenttasksRequest(BaseModel):
    limit: int = Field(10, description="Number of recent tasks to return")

class GetTaskResultRequest(BaseModel):
    task_id: str = Field(..., description="The unique ID of the task result to retrieve")
