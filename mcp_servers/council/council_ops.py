"""
Council Operations Library

⚠️ DEPRECATED - Pending Refactoring (Task 60268594)

This module currently depends on the legacy council_orchestrator which has been
archived to ARCHIVE/council_orchestrator_legacy/. 

This needs to be refactored to:
1. Use Agent Persona MCP for individual agent execution
2. Implement orchestration logic directly in this module
3. Use the packets/ system already migrated to mcp_servers/lib/council/packets/

DO NOT USE until refactoring is complete.

Provides interface to the Sanctuary Council Orchestrator for MCP tools.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from dataclasses import asdict
from mcp_servers.council.packets.schema import CouncilRoundPacket, seed_for

from mcp_servers.lib.logging_utils import setup_mcp_logging

logger = setup_mcp_logging(__name__)

class CouncilOperations:
    """Interface to the Council Orchestrator - Refactored for MCP Architecture"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize Council Operations
        
        Args:
            project_root: Path to Project Sanctuary root (auto-detected if None)
        """
        if project_root is None:
            # Auto-detect project root
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / "mcp_servers").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                raise RuntimeError("Could not find Project Sanctuary root")
        
        self.project_root = Path(project_root)
        self._initialized = False
        self.persona_ops = None
        self.cortex = None
        
        # Ensure logger is configured
        setup_mcp_logging(__name__)
    
    def _ensure_initialized(self):
        """Lazy initialization with cache warmup"""
        if not self._initialized:
            from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations
            from mcp_servers.rag_cortex.operations import CortexOperations
            
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
    
    def dispatch_task(
        self,
        task_description: str,
        agent: Optional[str] = None,
        max_rounds: int = 3,
        force_engine: Optional[str] = None,
        model_name: Optional[str] = None,
        output_path: Optional[str] = None,
        update_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Dispatch a task to the Council for multi-agent deliberation
        
        Args:
            task_description: Task for the council to execute
            agent: Specific agent ("coordinator", "strategist", "auditor") or None for full council
            max_rounds: Maximum deliberation rounds
            force_engine: Force specific engine (e.g. "ollama")
            model_name: Specific model to use
            output_path: Optional path to save result
            update_rag: Whether to update RAG after execution
            
        Returns:
            Dictionary with execution results
        """
        self._ensure_initialized()
        
        # Generate unique session ID
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Determine agents for deliberation
        if agent:
            agents = [agent]
        else:
            agents = ["coordinator", "strategist", "auditor"]
        
        # Query knowledge base for context
        try:
            rag_results = self.cortex.query(task_description, max_results=3)
            context = self._format_rag_context(rag_results)
            
            # Count results for telemetry
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
                        engine=force_engine
                    )
                    
                    # Create round packet
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
            
            return {
                "session_id": session_id,
                "status": "success",
                "rounds": max_rounds,
                "agents": agents,
                "packets": [asdict(p) for p in packets],
                "final_synthesis": final_decision
            }
            
        except Exception as e:
            logger.error(f"[Council MCP] Deliberation failed: {e}")
            return {
                "status": "error",
                "session_id": session_id,
                "error": str(e)
            }

    def _format_rag_context(self, rag_results: Any) -> str:
        """Format RAG results into context string"""
        context = "## Relevant Context\n\n"
        
        # Handle QueryResponse object or dict
        if hasattr(rag_results, "results"):
            results = rag_results.results
        else:
            results = rag_results.get("results", [])
            
        if not results:
            return ""
            
        for i, result in enumerate(results, 1):
            # Handle QueryResult object or dict
            if hasattr(result, "content"):
                content = result.content[:500]  # Truncate
                metadata = result.metadata
            else:
                content = result.get("content", "")[:500]
                metadata = result.get("metadata", {})
                
            source = metadata.get("source", "unknown")
            context += f"**Source {i} ({source}):**\n{content}...\n\n"
            
        return context
    
    def list_agents(self) -> List[Dict[str, str]]:
        """
        List available Council agents
        
        Returns:
            List of agent info dicts with name, status, persona
        """
        self._ensure_initialized()
        
        roles = self.persona_ops.list_roles()
        agents = []
        
        # Filter for built-in council roles
        for role in ["coordinator", "strategist", "auditor"]:
            if role in roles.get("built_in", []):
                agents.append({
                    "name": role,
                    "status": "available",
                    "persona": f"Built-in {role.capitalize()} persona"
                })
        
        return agents
    

