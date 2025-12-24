#============================================
# mcp_servers/agent_persona/operations.py
# Purpose: Core business logic for Agent Persona Operations.
#          Orchestrates agent instantiation, dispatch, and state persistence.
# Role: Business Logic Layer
# Used as: Helper module by server.py
# Calling example:
#   ops = PersonaOperations(project_root)
#   ops.dispatch(role="coordinator", task="...")
# LIST OF CLASSES/FUNCTIONS:
#   - PersonaOperations
#     - __init__
#     - list_roles
#     - dispatch
#     - get_state
#     - reset_state
#     - create_custom
#     - _create_agent
#     - _classify_response
#============================================

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Import local clean implementations
from .agent import Agent
from .llm_client import get_llm_client
from .models import PersonaConstants, AgentResponse
from .validator import PersonaValidator

from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.path_utils import find_project_root

logger = setup_mcp_logging(__name__)

class PersonaOperations:
    """
    Class: PersonaOperations
    Purpose: Orchestrates Agent interactions and lifecycle.
    """
    
    #============================================
    # Method: __init__
    # Purpose: Initialize operations and directories.
    # Args:
    #   project_root: Optional override for root
    #============================================
    def __init__(self, project_root: Path | None = None):
        """Initialize Agent Persona Operations."""
        self.project_root = project_root or Path(find_project_root())
        self.persona_dir = self.project_root / "mcp_servers" / "agent_persona" / "personas"
        self.state_dir = self.project_root / "mcp_servers" / "agent_persona" / "state"
        
        # Initialize Validator
        self.validator = PersonaValidator(self.persona_dir)

        # Ensure directories exist
        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Map of active agents (role -> Agent instance)
        self._active_agents: Dict[str, Agent] = {}
        
        setup_mcp_logging(__name__)
        logger.info(f"[Agent Persona MCP] Initialized with persona_dir: {self.persona_dir}")
    
    # _find_project_root removed in favor of mcp_servers.lib.path_utils.find_project_root
    
    #============================================
    # Method: list_roles
    # Purpose: List all available built-in and custom personas.
    # Returns: Dict summary of roles
    #============================================
    def list_roles(self) -> Dict[str, Any]:
        """List all available persona roles."""
        built_in = PersonaConstants.BUILT_IN_ROLES
        
        # Find custom personas
        custom = []
        if self.persona_dir.exists():
            for persona_file in self.persona_dir.glob("*.txt"):
                role_name = persona_file.stem
                if role_name not in built_in:
                    custom.append(role_name)
        
        return {
            "built_in": built_in,
            "custom": custom,
            "total": len(built_in) + len(custom),
            "persona_dir": str(self.persona_dir)
        }
    
    #============================================
    # Method: dispatch
    # Purpose: Execute a task using a specific persona agent.
    # Args:
    #   role: Target persona role
    #   task: Instructions for the agent
    #   context: Optional data context
    #   maintain_state: Persist conversation flag
    #   engine/model_*: LLM configuration
    #   custom_persona_file: Path for custom loading
    # Returns: AgentResponse dictionary
    #============================================
    def dispatch(
        self,
        role: str,
        task: str,
        context: Dict[str, Any] | None = None,
        maintain_state: bool = True,
        engine: str | None = None,
        model_name: str | None = None,
        model_preference: str | None = None,
        custom_persona_file: str | None = None
    ) -> Dict[str, Any]:
        """Dispatch a task to a specific persona agent."""
        try:
            # Validate and Normalize Role
            role_normalized = self.validator.validate_role_name(role)
            
            # Get or create agent
            if maintain_state and role_normalized in self._active_agents:
                agent = self._active_agents[role_normalized]
                logger.info(f"[Agent Persona] Reusing existing {role_normalized} agent")
            else:
                logger.info(f"[Agent Persona] Creating new {role_normalized} agent...")
                agent = self._create_agent(role_normalized, engine, model_name, model_preference, custom_persona_file)
                logger.info(f"[Agent Persona] Agent created successfully")
                if maintain_state:
                    self._active_agents[role_normalized] = agent
            
            # Build prompt with context
            prompt = task
            if context:
                context_str = json.dumps(context, indent=2)
                prompt = f"Context:\n{context_str}\n\nTask:\n{task}"
            
            logger.info(f"[Agent Persona] Dispatching to {role_normalized}")
            
            # Execute query
            response = agent.query(prompt)
            
            # Save state if maintaining
            if maintain_state:
                agent.save_history()
            
            # Create standardized response
            result = AgentResponse(
                role=role_normalized,
                response=response,
                reasoning_type=self._classify_response(response, role_normalized),
                session_id=f"persona_{role_normalized}_{id(agent)}",
                state_preserved=maintain_state,
                status="success"
            )
            return result.to_dict()
        
        except Exception as e:
            logger.error(f"[Agent Persona] Dispatch failed for {role}: {e}")
            return AgentResponse(
                role=role,
                response="",
                reasoning_type="error",
                session_id="",
                state_preserved=False,
                status="error",
                error=str(e)
            ).to_dict()
    
    def _create_agent(
        self,
        role: str,
        engine: str | None = None,
        model_name: str | None = None,
        model_preference: str | None = None,
        custom_persona_file: str | None = None
    ) -> Agent:
        """Internal helper to instantiate an Agent."""
        # Determine persona file
        if custom_persona_file:
            persona_file = self.validator.validate_file_path(custom_persona_file)
        else:
            # Use persona from agent_persona directory
            persona_file = self.persona_dir / f"{role}.txt"
        
        if not persona_file.exists():
            available_roles = self.list_roles()
            raise FileNotFoundError(
                f"Persona file not found: {persona_file}. "
                f"Available roles: {available_roles['built_in'] + available_roles['custom']}"
            )
        
        # Determine state file
        state_file = self.state_dir / f"{role}_session.json"
        
        # Create LLM client
        client = get_llm_client(provider=engine, model_name=model_name, ollama_host=None)
        
        # Create agent
        agent = Agent(
            client=client,
            persona_file=persona_file,
            state_file=state_file
        )
        
        return agent
    
    def _classify_response(self, response: str, role: str) -> str:
        """Classify response type based on keywords."""
        response_lower = response.lower()
        
        if role == "coordinator":
            if any(word in response_lower for word in ["plan", "strategy", "coordinate"]):
                return "strategy"
            elif any(word in response_lower for word in ["analysis", "evaluate"]):
                return "analysis"
        elif role == "strategist":
            if any(word in response_lower for word in ["propose", "suggest", "recommend"]):
                return "proposal"
            elif any(word in response_lower for word in ["design", "architecture"]):
                return "design"
        elif role == "auditor":
            if any(word in response_lower for word in ["review", "audit", "validate"]):
                return "critique"
            elif any(word in response_lower for word in ["risk", "concern", "issue"]):
                return "analysis"
        
        return "discussion"
    
    #============================================
    # Method: get_state
    # Purpose: Retrieve conversation history.
    # Args:
    #   role: Persona role identifier
    # Returns: State dictionary
    #============================================
    def get_state(self, role: str) -> Dict[str, Any]:
        """Get conversation state for a specific role."""
        try:
            role_normalized = self.validator.validate_role_name(role)
            state_file = self.state_dir / f"{role_normalized}_session.json"
            
            if not state_file.exists():
                return {
                    "role": role_normalized,
                    "state": "no_history",
                    "messages": []
                }
            
            messages = json.loads(state_file.read_text())
            return {
                "role": role_normalized,
                "state": "active",
                "messages": messages,
                "message_count": len(messages)
            }
        except Exception as e:
            logger.error(f"[Agent Persona] Failed to load state for {role}: {e}")
            return {"role": role, "state": "error", "error": str(e)}
    
    #============================================
    # Method: reset_state
    # Purpose: Clear conversation history.
    # Args:
    #   role: Persona role identifier
    # Returns: Status dictionary
    #============================================
    def reset_state(self, role: str) -> Dict[str, Any]:
        """Reset conversation state for a specific role."""
        try:
            role_normalized = self.validator.validate_role_name(role)
            state_file = self.state_dir / f"{role_normalized}_session.json"
            
            if state_file.exists():
                state_file.unlink()
            
            # Remove from active agents
            if role_normalized in self._active_agents:
                del self._active_agents[role_normalized]
            
            return {
                "role": role_normalized,
                "status": "reset",
                "message": f"State cleared for {role_normalized}"
            }
        except Exception as e:
            logger.error(f"[Agent Persona] Failed to reset state for {role}: {e}")
            return {"role": role, "status": "error", "error": str(e)}
    
    #============================================
    # Method: create_custom
    # Purpose: Generate a new custom persona file.
    # Args:
    #   role: New role name
    #   persona_definition: Prompt content
    #   description: Metadata description
    # Returns: Status dictionary
    #============================================
    def create_custom(
        self,
        role: str,
        persona_definition: str,
        description: str
    ) -> Dict[str, Any]:
        """Create a new custom persona."""
        try:
            role_normalized = self.validator.validate_role_name(role)
            
            # Validation logic
            self.validator.validate_custom_persona_creation(role_normalized, persona_definition)
            
            persona_file = self.persona_dir / f"{role_normalized}.txt"
            persona_file.write_text(persona_definition)
            
            logger.info(f"[Agent Persona] Created custom persona: {role_normalized}")
            
            return {
                "role": role_normalized,
                "file_path": str(persona_file),
                "description": description,
                "status": "created"
            }
        except Exception as e:
            logger.error(f"[Agent Persona] Failed to create custom persona {role}: {e}")
            return {"role": role, "status": "error", "error": str(e)}
