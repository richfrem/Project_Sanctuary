"""
Agent Persona MCP Server

Provides configurable AI agent personas via Model Context Protocol.
Supports built-in personas (Coordinator, Strategist, Auditor) and custom personas.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import sys

from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations

# Initialize FastMCP server
mcp = FastMCP("Agent Persona")

# Initialize operations
persona_ops = AgentPersonaOperations()

@mcp.tool()
def persona_dispatch(
    role: str,
    task: str,
    context: dict | None = None,
    maintain_state: bool = True,
    engine: str | None = None,
    model_name: str | None = None,
    custom_persona_file: str | None = None
) -> dict:
    """
    Dispatch a task to a specific persona agent.
    
    This tool executes a task using a configurable AI agent persona. The agent
    can be one of the built-in personas (coordinator, strategist, auditor) or
    a custom persona.
    
    Args:
        role: Persona role (coordinator, strategist, auditor, or custom role name)
        task: Task for the agent to execute
        context: Optional context dictionary (e.g., from Cortex MCP query results)
        maintain_state: Whether to persist conversation history (default: True)
        engine: AI engine to use - "gemini", "openai", or "ollama" (default: auto-select)
        model_name: Specific model variant (e.g., 'Sanctuary-Qwen2-7B:latest', 'gpt-4', 'gemini-2.5-pro')
        custom_persona_file: Path to custom persona definition file (optional)
    
    Returns:
        Dictionary containing:
        - role: The persona role used
        - response: Agent's response to the task
        - reasoning_type: Type of reasoning (strategy, analysis, proposal, critique, discussion)
        - session_id: Unique session identifier
        - state_preserved: Whether conversation state was maintained
        - status: "success" or "error"
    
    Example - Built-in Persona:
        result = persona_dispatch(
            role="auditor",
            task="Review the test coverage for the Git MCP server"
        )
    
    Example - With Context:
        # First, get context from Cortex MCP
        context = cortex_query(query="Previous security audits", max_results=3)
        
        # Then dispatch to security reviewer with context
        result = persona_dispatch(
            role="security_reviewer",
            task="Audit the authentication flow",
            context=context
        )
    
    Example - Custom Persona:
        result = persona_dispatch(
            role="performance_analyst",
            task="Analyze the database query performance",
            custom_persona_file="/path/to/performance_analyst.txt"
        )
    """
    return persona_ops.dispatch(
        role=role,
        task=task,
        context=context,
        maintain_state=maintain_state,
        engine=engine,
        model_name=model_name,
        custom_persona_file=custom_persona_file
    )

@mcp.tool()
def persona_list_roles() -> dict:
    """
    List all available persona roles (built-in and custom).
    
    Returns:
        Dictionary containing:
        - built_in: List of built-in persona roles (coordinator, strategist, auditor)
        - custom: List of custom persona roles
        - total: Total number of available personas
        - persona_dir: Directory where custom personas are stored
    
    Example:
        roles = persona_list_roles()
        # Returns: {
        #   "built_in": ["coordinator", "strategist", "auditor"],
        #   "custom": ["security_reviewer", "performance_analyst"],
        #   "total": 5,
        #   "persona_dir": "/path/to/personas"
        # }
    """
    return persona_ops.list_roles()

@mcp.tool()
def persona_get_state(role: str) -> dict:
    """
    Get conversation state for a specific persona role.
    
    Retrieves the conversation history for a persona agent, useful for
    understanding context or debugging agent behavior.
    
    Args:
        role: Persona role to get state for
    
    Returns:
        Dictionary containing:
        - role: The persona role
        - state: "active", "no_history", or "error"
        - messages: List of conversation messages (if available)
        - message_count: Number of messages in history
    
    Example:
        state = persona_get_state(role="coordinator")
        print(f"Coordinator has {state['message_count']} messages in history")
    """
    return persona_ops.get_state(role=role)

@mcp.tool()
def persona_reset_state(role: str) -> dict:
    """
    Reset conversation state for a specific persona role.
    
    Clears the conversation history for a persona agent, useful for
    starting fresh or clearing context.
    
    Args:
        role: Persona role to reset
    
    Returns:
        Dictionary containing:
        - role: The persona role
        - status: "reset" or "error"
        - message: Status message
    
    Example:
        result = persona_reset_state(role="auditor")
        # Auditor's conversation history is now cleared
    """
    return persona_ops.reset_state(role=role)

@mcp.tool()
def persona_create_custom(
    role: str,
    persona_definition: str,
    description: str
) -> dict:
    """
    Create a new custom persona.
    
    Creates a new persona definition file that can be used with persona_dispatch.
    This enables creating specialized agents for specific domains or tasks.
    
    Args:
        role: Unique role identifier (e.g., "security_reviewer", "performance_analyst")
        persona_definition: Full persona instruction text defining the agent's behavior
        description: Brief description of the persona's purpose
    
    Returns:
        Dictionary containing:
        - role: The created role identifier
        - file_path: Path to the created persona file
        - description: Description of the persona
        - status: "created" or "error"
    
    Example:
        result = persona_create_custom(
            role="security_reviewer",
            persona_definition='''You are a Security Reviewer for Project Sanctuary.
            
            Your role is to:
            - Identify security vulnerabilities and risks
            - Review code for security best practices
            - Assess authentication and authorization mechanisms
            - Evaluate data protection and encryption
            - Check for common security anti-patterns (SQL injection, XSS, etc.)
            
            Be thorough, critical, and provide actionable recommendations.''',
            description="Security audit and vulnerability assessment"
        )
    """
    return persona_ops.create_custom(
        role=role,
        persona_definition=persona_definition,
        description=description
    )

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
