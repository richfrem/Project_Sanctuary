"""
Council MCP Server

Exposes Sanctuary Council Orchestrator capabilities via Model Context Protocol.

DESIGN PRINCIPLE: Separation of Concerns
- This MCP focuses ONLY on multi-agent deliberation (the Council's unique capability)
- File operations → Use Code MCP (code_write, code_read)
- Memory queries → Use Cortex MCP (cortex_query)
- Git operations → Use Git MCP (git_add, git_smart_commit)
- Protocol docs → Use Protocol MCP (protocol_create)
- Task management → Use Task MCP (create_task)

The Council MCP is a HIGH-LEVEL ORCHESTRATOR, not a duplicate of existing services.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.council.council_ops import CouncilOperations

# Initialize FastMCP server
mcp = FastMCP("Council Orchestrator")

# Initialize operations
council_ops = CouncilOperations()

@mcp.tool()
def council_dispatch(
    task_description: str,
    agent: str | None = None,
    max_rounds: int = 3,
    force_engine: str | None = None,
    output_path: str | None = None
) -> dict:
    """
    Dispatch a task to the Sanctuary Council for multi-agent deliberation.
    
    This is the CORE capability of the Council MCP - multi-agent cognitive processing.
    For other operations, use the appropriate specialized MCP server:
    - File I/O: Code MCP (code_write, code_read)
    - Memory: Cortex MCP (cortex_query, cortex_ingest_incremental)
    - Git: Git MCP (git_add, git_smart_commit, git_push_feature)
    - Protocols: Protocol MCP (protocol_create, protocol_get)
    - Tasks: Task MCP (create_task, update_task_status)
    
    Args:
        task_description: The task for the council to deliberate on
        agent: Optional specific agent ("coordinator", "strategist", "auditor"). 
               If None, full council deliberation is used.
        max_rounds: Maximum number of deliberation rounds (default: 3)
        force_engine: Force specific AI engine ("gemini", "openai", "ollama")
        output_path: Optional output file path (relative to project root)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - decision: Council's deliberation output
        - session_id: Unique session identifier
        - output_path: Path to output file (if generated)
    
    Example - Full Council Deliberation:
        result = council_dispatch(
            task_description="Review the architecture for Protocol 115 and provide recommendations",
            max_rounds=3
        )
    
    Example - Single Agent Consultation:
        result = council_dispatch(
            task_description="Audit the test coverage for the Git MCP server",
            agent="auditor",
            max_rounds=2
        )
    
    Example - Workflow Composition:
        # 1. Council deliberates
        decision = council_dispatch(
            task_description="Design a new protocol for MCP composition patterns",
            output_path="WORK_IN_PROGRESS/protocol_design.md"
        )
        
        # 2. Save to protocol (use Protocol MCP)
        protocol_create(
            number=120,
            title="MCP Composition Patterns",
            content=decision["decision"]
        )
        
        # 3. Commit (use Git MCP)
        git_add(files=["01_PROTOCOLS/120_mcp_composition_patterns.md"])
        git_smart_commit(message="feat(protocol): Add Protocol 120 - MCP Composition")
    """
    return council_ops.dispatch_task(
        task_description=task_description,
        agent=agent,
        max_rounds=max_rounds,
        force_engine=force_engine,
        output_path=output_path
    )

@mcp.tool()
def council_list_agents() -> list[dict]:
    """
    List all available Council agents and their current status.
    
    Returns:
        List of agent dictionaries with:
        - name: Agent identifier
        - status: Current availability status
        - persona: Agent's role and specialty
    
    Example:
        agents = council_list_agents()
        # Returns: [
        #   {"name": "coordinator", "status": "available", "persona": "Task planning and execution oversight"},
        #   {"name": "strategist", "status": "available", "persona": "Long-term planning and risk assessment"},
        #   {"name": "auditor", "status": "available", "persona": "Quality assurance and compliance verification"}
        # ]
    """
    return council_ops.list_agents()

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
