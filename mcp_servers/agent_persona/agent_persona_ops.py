"""
Agent Persona Operations Library

Provides operations for managing and executing AI agent personas.
Uses the clean Agent and LLMClient implementations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Import local clean implementations
from .agent import Agent
from .llm_client import get_llm_client

from mcp_servers.lib.logging_utils import setup_mcp_logging

logger = setup_mcp_logging(__name__)

# Constants
COORDINATOR = "coordinator"
STRATEGIST = "strategist"
AUDITOR = "auditor"

class AgentPersonaOperations:
    """Operations for managing AI agent personas"""
    
    def __init__(self, project_root: Path | None = None):
        """
        Initialize Agent Persona Operations
        
        Args:
            project_root: Project root directory (auto-detected if None)
        """
        self.project_root = project_root or self._find_project_root()
        self.persona_dir = self.project_root / "mcp_servers" / "agent_persona" / "personas"
        self.state_dir = self.project_root / "mcp_servers" / "agent_persona" / "state"
        
        # Ensure directories exist
        self.persona_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Map of active agents (role -> Agent instance)
        self._active_agents: Dict[str, Agent] = {}
        
        # Ensure logger is configured
        setup_mcp_logging(__name__)
        
        logger.info(f"[Agent Persona MCP] Initialized with persona_dir: {self.persona_dir}")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root"""
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "mcp_servers").exists():
                return current
            current = current.parent
        raise RuntimeError("Could not find project root")
    
    def list_roles(self) -> Dict[str, Any]:
        """
        List all available persona roles
        
        Returns:
            Dictionary with built-in and custom roles
        """
        built_in = [COORDINATOR, STRATEGIST, AUDITOR]
        
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
    
    def dispatch(
        self,
        role: str,
        task: str,
        context: Dict[str, Any] | None = None,
        maintain_state: bool = True,
        engine: str | None = None,
        model_name: str | None = None,
        custom_persona_file: str | None = None
    ) -> Dict[str, Any]:
        """
        Dispatch a task to a specific persona agent
        
        Args:
            role: Persona role (coordinator, strategist, auditor, or custom)
            task: Task for the agent
            context: Optional context dictionary
            maintain_state: Whether to persist conversation history
            engine: AI engine to use (gemini, openai, ollama)
            model_name: Specific model variant
            custom_persona_file: Path to custom persona file
        
        Returns:
            Dictionary with agent response and metadata
        """
        try:
            role_normalized = role.lower()
            
            # Get or create agent
            if maintain_state and role_normalized in self._active_agents:
                agent = self._active_agents[role_normalized]
                logger.info(f"[Agent Persona] Reusing existing {role_normalized} agent")
            else:
                logger.info(f"[Agent Persona] Creating new {role_normalized} agent...")
                agent = self._create_agent(role_normalized, engine, model_name, custom_persona_file)
                logger.info(f"[Agent Persona] Agent created successfully")
                if maintain_state:
                    self._active_agents[role_normalized] = agent
            
            # Build prompt with context
            prompt = task
            if context:
                context_str = json.dumps(context, indent=2)
                prompt = f"Context:\n{context_str}\n\nTask:\n{task}"
            
            logger.info(f"[Agent Persona] Dispatching to {role_normalized} (maintain_state={maintain_state})")
            logger.info(f"[Agent Persona] Task preview: {task[:100]}...")
            
            # Execute query
            logger.info(f"[Agent Persona] About to call agent.query() with prompt length: {len(prompt)}")
            response = agent.query(prompt)
            logger.info(f"[Agent Persona] agent.query() returned successfully")
            
            # Save state if maintaining
            if maintain_state:
                agent.save_history()
            
            return {
                "role": role_normalized,
                "response": response,
                "reasoning_type": self._classify_response(response, role_normalized),
                "session_id": f"persona_{role_normalized}_{id(agent)}",
                "state_preserved": maintain_state,
                "status": "success"
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"[Agent Persona] Dispatch failed for {role}: {e}")
            return {
                "role": role,
                "status": "error",
                "error": str(e)
            }
    
    def _create_agent(
        self,
        role: str,
        engine: str | None = None,
        model_name: str | None = None,
        custom_persona_file: str | None = None
    ) -> Agent:
        """
        Create a new Agent instance
        
        Args:
            role: Persona role
            engine: AI engine to use (gemini, openai, ollama)
            model_name: Specific model variant
            custom_persona_file: Path to custom persona file
        
        Returns:
            Agent instance
        """
        # Determine persona file
        if custom_persona_file:
            persona_file = Path(custom_persona_file)
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
        client = get_llm_client(provider=engine, model_name=model_name)
        
        # Create agent
        agent = Agent(
            client=client,
            persona_file=persona_file,
            state_file=state_file
        )
        
        return agent
    
    def _classify_response(self, response: str, role: str) -> str:
        """Classify response type based on content and role"""
        response_lower = response.lower()
        
        if role == COORDINATOR:
            if any(word in response_lower for word in ["plan", "strategy", "coordinate"]):
                return "strategy"
            elif any(word in response_lower for word in ["analysis", "evaluate"]):
                return "analysis"
        elif role == STRATEGIST:
            if any(word in response_lower for word in ["propose", "suggest", "recommend"]):
                return "proposal"
            elif any(word in response_lower for word in ["design", "architecture"]):
                return "design"
        elif role == AUDITOR:
            if any(word in response_lower for word in ["review", "audit", "validate"]):
                return "critique"
            elif any(word in response_lower for word in ["risk", "concern", "issue"]):
                return "analysis"
        
        return "discussion"
    
    def get_state(self, role: str) -> Dict[str, Any]:
        """
        Get conversation state for a specific role
        
        Args:
            role: Persona role
        
        Returns:
            Dictionary with conversation history
        """
        role_normalized = role.lower()
        state_file = self.state_dir / f"{role_normalized}_session.json"
        
        if not state_file.exists():
            return {
                "role": role_normalized,
                "state": "no_history",
                "messages": []
            }
        
        try:
            messages = json.loads(state_file.read_text())
            return {
                "role": role_normalized,
                "state": "active",
                "messages": messages,
                "message_count": len(messages)
            }
        except Exception as e:
            logger.error(f"[Agent Persona] Failed to load state for {role}: {e}")
            return {
                "role": role_normalized,
                "state": "error",
                "error": str(e)
            }
    
    def reset_state(self, role: str) -> Dict[str, Any]:
        """
        Reset conversation state for a specific role
        
        Args:
            role: Persona role
        
        Returns:
            Status dictionary
        """
        role_normalized = role.lower()
        state_file = self.state_dir / f"{role_normalized}_session.json"
        
        try:
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
            return {
                "role": role_normalized,
                "status": "error",
                "error": str(e)
            }
    
    def create_custom(
        self,
        role: str,
        persona_definition: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Create a new custom persona
        
        Args:
            role: Unique role identifier
            persona_definition: Full persona instruction text
            description: Brief description of the role
        
        Returns:
            Status dictionary with file path
        """
        try:
            role_normalized = role.lower().replace(" ", "_")
            persona_file = self.persona_dir / f"{role_normalized}.txt"
            
            if persona_file.exists():
                return {
                    "role": role_normalized,
                    "status": "error",
                    "error": f"Persona '{role_normalized}' already exists"
                }
            
            # Write persona file
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
            return {
                "role": role,
                "status": "error",
                "error": str(e)
            }
