
"""
Agent Implementation for Agent Persona MCP

Represents an AI agent with a specific persona and conversation state.
Replaces the legacy 'PersonaAgent' class.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class Agent:
    """
    An AI agent with a specific persona (system prompt) and conversation history.
    """
    
    def __init__(self, client: LLMClient, persona_file: Path, state_file: Path = None):
        """
        Initialize an Agent.
        
        Args:
            client: The LLM client to use for generation
            persona_file: Path to the persona definition file
            state_file: Path to the conversation state file (optional)
        """
        self.client = client
        self.persona_file = persona_file
        self.state_file = state_file
        self.role = self._extract_role_from_filename(persona_file.name)
        self.messages: List[Dict[str, str]] = []
        
        self._load_history()
        
        logger.info(f"Initialized {self.role} agent with {type(client).__name__}")

    def _extract_role_from_filename(self, filename: str) -> str:
        """Extract role name from filename, handling legacy and new formats"""
        try:
            # Try legacy format: core_essence_{ROLE}_awakening_seed.txt
            if "core_essence_" in filename and "_awakening_seed.txt" in filename:
                return filename.split('core_essence_')[1].split('_awakening_seed.txt')[0].upper()
            
            # Fallback to simple format: {role}.txt
            return Path(filename).stem.upper()
        except Exception:
            return "UNKNOWN"

    def _load_history(self):
        """Load conversation history from state file"""
        if self.state_file and self.state_file.exists():
            try:
                content = self.state_file.read_text(encoding="utf-8")
                self.messages = json.loads(content)
                logger.info(f"Loaded history for {self.role} ({len(self.messages)} messages)")
            except Exception as e:
                logger.error(f"Failed to load history for {self.role}: {e}")
                self.messages = []
        
        # Initialize with system prompt if empty
        if not self.messages:
            try:
                system_prompt = self.persona_file.read_text(encoding="utf-8")
                self.messages.append({"role": "system", "content": system_prompt})
            except Exception as e:
                logger.error(f"Failed to read persona file {self.persona_file}: {e}")
                self.messages.append({"role": "system", "content": f"You are the {self.role}."})

    def save_history(self):
        """Save conversation history to state file"""
        if self.state_file:
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                self.state_file.write_text(json.dumps(self.messages, indent=2), encoding="utf-8")
                logger.info(f"Saved history for {self.role}")
            except Exception as e:
                logger.error(f"Failed to save history for {self.role}: {e}")

    def query(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: The user message
            
        Returns:
            The agent's response
        """
        logger.info(f"[{self.role}] query() called with message length: {len(message)}")
        
        # Add user message
        self.messages.append({"role": "user", "content": message})
        logger.info(f"[{self.role}] User message appended, total messages: {len(self.messages)}")
        
        try:
            logger.info(f"[{self.role}] Querying LLM with {len(self.messages)} messages...")
            import time
            start_time = time.time()
            
            # Execute turn using LLM client
            response = self.client.execute_turn(self.messages)
            
            elapsed = time.time() - start_time
            logger.info(f"[{self.role}] LLM response received in {elapsed:.2f}s ({len(response)} chars)")
            
            # Add assistant response
            self.messages.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(f"[{self.role}] {error_msg}")
            # Don't append error to history to avoid poisoning state
            raise RuntimeError(error_msg)
