#!/usr/bin/env python3
"""
Council Operations
=====================================

Purpose:
    Core business logic for Council Operations.
    Orchestrates multi-agent deliberation and synthesis.

Layer: Business Logic

Key Classes:
    - CouncilOperations: Main manager
        - __init__(project_root)
        - dispatch_task(task_description, agent, ...)
        - list_agents()
        - _ensure_initialized()
        - _format_rag_context()
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.council.packets.schema import CouncilRoundPacket, seed_for
from .validator import CouncilValidator
from .models import CouncilAgent, CouncilTaskResult

logger = setup_mcp_logging(__name__)

class CouncilOperations:
    """
    Class: CouncilOperations
    Purpose: Interface to the Council Orchestrator - Refactored for MCP Architecture.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else Path(find_project_root())
        self.validator = CouncilValidator()
        self._initialized = False
        self.persona_ops = None
        self.cortex = None

    #============================================
    # Method: _ensure_initialized
    # Purpose: Lazy initialization of dependencies.
    #============================================
    def _ensure_initialized(self):
        """Lazy initialization with cache warmup."""
        if not self._initialized:
            # Deferred imports to avoid circular deps and startup cost
            from mcp_servers.agent_persona.operations import AgentPersonaOperations
            from mcp_servers.rag_cortex.operations import CortexOperations
            
            # AgentPersonaOperations expects project_root to be a Path or str depending on impl.
            # We updated AgentPersona to use models/validator/operations structure.
            # Its ops init takes `project_root: Path | None`.
            self.persona_ops = AgentPersonaOperations(project_root=self.project_root)
            self.cortex = CortexOperations(project_root=str(self.project_root))
            
            # Warm up cache if needed
            try:
                stats = self.cortex.get_cache_stats()
                if stats.get("hot_cache_size", 0) == 0:
                    logger.info("Cache empty, warming up...")
                    self.cortex.cache_warmup()
            except Exception as e:
                logger.warning(f"Cache warmup failed: {e}")
            
            self._initialized = True

    #============================================
    # Method: dispatch_task
    # Purpose: Dispatch a task to the Council for multi-agent deliberation.
    # Args:
    #   task_description: Task content
    #   agent: Optional single agent
    #   max_rounds: Deliberation rounds
    #   force_engine: Override AI engine
    #   model_name: Specific model
    #   model_preference: Routing preference
    #   output_path: Optional save path
    # Returns: Dict with execution results
    #============================================
    def dispatch_task(
        self,
        task_description: str,
        agent: Optional[str] = None,
        max_rounds: int = 3,
        force_engine: Optional[str] = None,
        model_name: Optional[str] = None,
        model_preference: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dispatch a task to the Council."""
        # Validate inputs
        self.validator.validate_task(task_description)
        self.validator.validate_agent(agent)
        
        self._ensure_initialized()
        
        session_id = str(uuid.uuid4())[:8]
        
        # Determine agents
        agents = [agent] if agent else ["coordinator", "strategist", "auditor"]
        
        # Query knowledge base for context
        try:
            rag_results = self.cortex.query(task_description, max_results=3)
            context = self._format_rag_context(rag_results)
            
            if hasattr(rag_results, "results"):
                result_count = len(rag_results.results)
            else:
                result_count = len(rag_results.get("results", []))
                
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            rag_results = {}
            context = ""
            result_count = 0
        
        # Multi-round deliberation
        packets = []
        final_decision = ""
        
        try:
            for round_num in range(max_rounds):
                for agent_role in agents:
                    # Dispatch to agent via Agent Persona MCP
                    response = self.persona_ops.dispatch(
                        role=agent_role,
                        task=task_description,
                        context=context,
                        model_name=model_name or "Sanctuary-Qwen2-7B:latest",
                        engine=force_engine,
                        model_preference=model_preference
                    )
                    
                    # Create round packet
                    # Note: Using existing CouncilRoundPacket schema which is outside this module refactor scope for now
                    packet = CouncilRoundPacket(
                        timestamp=datetime.now().isoformat(),
                        session_id=session_id,
                        round_id=round_num,
                        member_id=agent_role,
                        engine=response.get("engine", "ollama"),
                        seed=seed_for(session_id, round_num, agent_role),
                        prompt_hash="",
                        inputs={"task": task_description},
                        decision=response.get("response", ""),
                        rationale=response.get("reasoning", ""),
                        confidence=0.8,
                        citations=[],
                        rag={"results": result_count},
                        cag={}
                    )
                    packets.append(packet)
                    
                    # Update context with agent's response
                    agent_response = response.get("response", "")
                    context += f"\n\n**{agent_role.capitalize()}:** {agent_response}"
                    final_decision = agent_response
            
            result = CouncilTaskResult(
                session_id=session_id,
                status="success",
                rounds=max_rounds,
                agents=agents,
                packets=[asdict(p) for p in packets],
                final_synthesis=final_decision,
                output_path=output_path
            )
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"[Council MCP] Deliberation failed: {e}")
            return {
                "status": "error",
                "session_id": session_id,
                "error": str(e)
            }

    #============================================
    # Method: _format_rag_context
    # Purpose: Format RAG results into context string.
    # Args:
    #   rag_results: Raw results from Cortex
    # Returns: Formatted string
    #============================================
    def _format_rag_context(self, rag_results: Any) -> str:
        """Format RAG results into context string."""
        context = "## Relevant Context\n\n"
        
        # Handle QueryResponse object or dict
        if hasattr(rag_results, "results"):
            results = rag_results.results
        else:
            results = rag_results.get("results", [])
            
        if not results:
            return ""
            
        for i, result in enumerate(results, 1):
            if hasattr(result, "content"):
                content = result.content[:500]  # Truncate
                metadata = result.metadata
            else:
                content = result.get("content", "")[:500]
                metadata = result.get("metadata", {})
                
            source = metadata.get("source", "unknown")
            context += f"**Source {i} ({source}):**\n{content}...\n\n"
            
        return context

    #============================================
    # Method: list_agents
    # Purpose: List available Council agents.
    # Returns: List of CouncilAgent dicts
    #============================================
    def list_agents(self) -> List[Dict[str, str]]:
        """List available Council agents."""
        self._ensure_initialized()
        
        roles = self.persona_ops.list_roles()
        agents = []
        
        # Filter for built-in council roles
        for role in ["coordinator", "strategist", "auditor"]:
            if role in roles.get("built_in", []):
                agents.append(CouncilAgent(
                    name=role,
                    status="available",
                    persona=f"Built-in {role.capitalize()} persona"
                ).to_dict())
        
        return agents
