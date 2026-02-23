#!/usr/bin/env python3
"""
Council Models
=====================================

Purpose:
    Data definitions for Council Server.
    Defines agents, task results, and MCP request packets.

Layer: Data (DTOs)

Key Models:
    # Internal
    - CouncilAgent: Agent metadata and status
    - CouncilTaskResult: Outcome of deliberation session

    # MCP Requests
    - CouncilDispatchRequest: Task parameters
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class CouncilAgent:
    """Represents a Council Agent."""
    name: str
    status: str
    persona: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "persona": self.persona
        }

@dataclass
class CouncilTaskResult:
    """Result of a Council deliberation task."""
    session_id: str
    status: str
    rounds: int
    agents: List[str]
    packets: List[Dict[str, Any]]
    final_synthesis: str
    error: Optional[str] = None
    output_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "status": self.status,
            "rounds": self.rounds,
            "agents": self.agents,
            "packets": self.packets,
            "final_synthesis": self.final_synthesis
        }
        if self.error:
            result["error"] = self.error
        if self.output_path:
            result["output_path"] = self.output_path
        return result


# ============================================================================
# FastMCP Request Models
# ============================================================================
from pydantic import BaseModel, Field

class CouncilDispatchRequest(BaseModel):
    task_description: str = Field(..., description="The task description for the council to deliberate on")
    agent: Optional[str] = Field(None, description="Optional specific agent ('coordinator', 'strategist', 'auditor')")
    max_rounds: int = Field(3, description="Maximum number of deliberation rounds")
    force_engine: Optional[str] = Field(None, description="Force specific engine ('gemini', 'openai', 'ollama')")
    model_preference: Optional[str] = Field(None, description="Model preference ('OLLAMA', 'GEMINI', 'GPT')")
    output_path: Optional[str] = Field(None, description="Optional relative output file path")
