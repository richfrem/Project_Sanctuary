#!/usr/bin/env python3
"""
Agent Persona Models
=====================================

Purpose:
    Data definitions for Agent Persona Server.
    Defines schemas for agent responses, roles, and lifecycle management.

Layer: Data (DTOs)

Key Models:
    # Internal
    - PersonaRole: Enum for standard/custom roles
    - AgentResponse: Dispatch execution result
    - PersonaConstants: Paths and built-ins

    # MCP Requests
    - PersonaDispatchParams
    - PersonaRoleParams
    - PersonaCreateCustomParams
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
from pathlib import Path

#============================================
# Enum: PersonaRole
# Purpose: Define standard and custom agent roles.
#============================================
class PersonaRole(Enum):
    """Enumeration of agent roles."""
    COORDINATOR = "coordinator"
    STRATEGIST = "strategist"
    AUDITOR = "auditor"
    CUSTOM = "custom"

#============================================
# DataClass: AgentResponse
# Purpose: Structured response from an agent dispatch.
#============================================
@dataclass
class AgentResponse:
    """Agent response data model."""
    role: str
    response: str
    reasoning_type: str
    session_id: str
    state_preserved: bool
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "role": self.role,
            "response": self.response,
            "reasoning_type": self.reasoning_type,
            "session_id": self.session_id,
            "state_preserved": self.state_preserved,
            "status": self.status
        }
        if self.error:
            result["error"] = self.error
        return result

#============================================
# Class: PersonaConstants
# Purpose: System-wide constants.
#============================================
class PersonaConstants:
    """Constants for Agent Persona."""
    DEFAULT_PERSONA_DIR = Path("mcp_servers/agent_persona/personas")
    DEFAULT_STATE_DIR = Path("mcp_servers/agent_persona/state")
    BUILT_IN_ROLES = [role.value for role in PersonaRole if role != PersonaRole.CUSTOM]

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field

class PersonaDispatchParams(BaseModel):
    role: str = Field(..., description="Persona role (coordinator, strategist, auditor, or custom)")
    task: str = Field(..., description="The task or message for the agent")
    context: Optional[dict] = Field(None, description="Optional supporting context")
    maintain_state: bool = Field(True, description="Whether to persist and use conversation history")
    engine: Optional[str] = Field(None, description="AI engine provider (gemini, openai, ollama)")
    model_name: Optional[str] = Field(None, description="Specific model variant to use")
    custom_persona_file: Optional[str] = Field(None, description="Path to a custom persona definition file")

class PersonaRoleParams(BaseModel):
    role: str = Field(..., description="The unique name/ID of the persona role")

class PersonaCreateCustomParams(BaseModel):
    role: str = Field(..., description="Unique role identifier for the custom persona")
    persona_definition: str = Field(..., description="Full system prompt or behavior definition")
    description: str = Field(..., description="Brief description of the persona's purpose")
