"""
Sanctuary Domain Logic Server (Container #6)
Functions as the "Brain" of the project, hosting core business logic and Python development tools.

Aggregates:
- Chronicle MCP (Business Logic)
- Protocol MCP (Business Logic)
- Task MCP (Business Logic)
- ADR MCP (Business Logic)
- Python Dev Tools (lint, format, analyze) from Code MCP

This unified server solves the "Router vs Runtime" problem by providing a dedicated 
Python runtime for logic that cannot run on the Gateway.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Add project root to python path to allow imports from other modules
# We are likely running from PROJECT_ROOT in the container, but let's be safe
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastmcp import FastMCP

# Import Operations Classes
from mcp_servers.chronicle.operations import ChronicleOperations
from mcp_servers.protocol.operations import ProtocolOperations
from mcp_servers.task.operations import TaskOperations
from mcp_servers.adr.operations import ADROperations
from mcp_servers.code.code_ops import CodeOperations
from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations
from mcp_servers.config.config_ops import ConfigOperations
from mcp_servers.workflow.operations import WorkflowOperations

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.domain")

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/app")  # Default to /app in Docker

# Setup Directories
CHRONICLE_DIR = os.path.join(PROJECT_ROOT, "00_CHRONICLE/ENTRIES")
PROTOCOL_DIR = os.path.join(PROJECT_ROOT, "01_PROTOCOLS")
ADRS_DIR = os.path.join(PROJECT_ROOT, "ADRs")

# Initialize Operations
chronicle_ops = ChronicleOperations(CHRONICLE_DIR)
protocol_ops = ProtocolOperations(PROTOCOL_DIR)
task_ops = TaskOperations(Path(PROJECT_ROOT))
adr_ops = ADROperations(ADRS_DIR)
code_ops = CodeOperations(PROJECT_ROOT)
persona_ops = AgentPersonaOperations()
CONFIG_DIR = os.path.join(PROJECT_ROOT, ".agent/config")
CONFIG_DIR = os.path.join(PROJECT_ROOT, ".agent/config")
config_ops = ConfigOperations(CONFIG_DIR)
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, ".agent/workflows")
workflow_ops = WorkflowOperations(Path(WORKFLOW_DIR))

# =============================================================================
# CHRONICLE TOOLS
# =============================================================================

@mcp.tool()
def chronicle_create_entry(
    title: str,
    content: str,
    author: str,
    date: Optional[str] = None,
    status: str = "draft",
    classification: str = "internal"
) -> str:
    """Create a new chronicle entry."""
    try:
        result = chronicle_ops.create_entry(title, content, author, date, status, classification)
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    except Exception as e:
        return f"Error creating entry: {str(e)}"

@mcp.tool()
def chronicle_append_entry(
    title: str,
    content: str,
    author: str,
    date: Optional[str] = None,
    status: str = "draft",
    classification: str = "internal"
) -> str:
    """Append a new entry to the Chronicle (Alias for create_entry)."""
    try:
        result = chronicle_ops.create_entry(title, content, author, date, status, classification)
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    except Exception as e:
        return f"Error creating entry: {str(e)}"

@mcp.tool()
def chronicle_update_entry(
    entry_number: int,
    updates: Dict[str, Any],
    reason: str,
    override_approval_id: Optional[str] = None
) -> str:
    """Update an existing chronicle entry."""
    try:
        result = chronicle_ops.update_entry(entry_number, updates, reason, override_approval_id)
        return f"Updated Chronicle Entry {result['entry_number']}. Fields: {', '.join(result['updated_fields'])}"
    except Exception as e:
        return f"Error updating entry: {str(e)}"

@mcp.tool()
def chronicle_get_entry(entry_number: int) -> str:
    """Retrieve a specific chronicle entry."""
    try:
        entry = chronicle_ops.get_entry(entry_number)
        return f"""Entry {entry['number']}: {entry['title']}
Date: {entry['date']}
Author: {entry['author']}
Status: {entry['status']}
Classification: {entry['classification']}

{entry['content']}"""
    except Exception as e:
        return f"Error retrieving entry: {str(e)}"

@mcp.tool()
def chronicle_list_entries(limit: int = 10) -> str:
    """List recent chronicle entries."""
    try:
        entries = chronicle_ops.list_entries(limit)
        if not entries:
            return "No entries found."
            
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing entries: {str(e)}"

@mcp.tool()
def chronicle_read_latest_entries(limit: int = 10) -> str:
    """Read the latest entries from the Chronicle (Alias for list_entries)."""
    try:
        entries = chronicle_ops.list_entries(limit)
        if not entries:
            return "No entries found."
            
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing entries: {str(e)}"

@mcp.tool()
def chronicle_search(query: str) -> str:
    """Search chronicle entries by content."""
    try:
        results = chronicle_ops.search_entries(query)
        if not results:
            return f"No entries found matching '{query}'"
            
        output = [f"Found {len(results)} entries matching '{query}':"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching entries: {str(e)}"

# =============================================================================
# PROTOCOL TOOLS
# =============================================================================

@mcp.tool()
def protocol_create(
    number: int,
    title: str,
    status: str,
    classification: str,
    version: str,
    authority: str,
    content: str,
    linked_protocols: Optional[str] = None
) -> str:
    """Create a new protocol."""
    try:
        result = protocol_ops.create_protocol(
            number, title, status, classification, version, authority, content, linked_protocols
        )
        return f"Created Protocol {result['protocol_number']}: {result['file_path']}"
    except Exception as e:
        return f"Error creating protocol: {str(e)}"

@mcp.tool()
def protocol_update(
    number: int,
    updates: Dict[str, Any],
    reason: str
) -> str:
    """Update an existing protocol."""
    try:
        result = protocol_ops.update_protocol(number, updates, reason)
        return f"Updated Protocol {result['protocol_number']}. Fields: {', '.join(result['updated_fields'])}"
    except Exception as e:
        return f"Error updating protocol: {str(e)}"

@mcp.tool()
def protocol_get(number: int) -> str:
    """Retrieve a specific protocol."""
    try:
        protocol = protocol_ops.get_protocol(number)
        return f"""Protocol {protocol['number']}: {protocol['title']}
Status: {protocol['status']}
Classification: {protocol['classification']}
Version: {protocol['version']}
Authority: {protocol['authority']}
Linked Protocols: {protocol.get('linked_protocols', 'None')}

{protocol['content']}"""
    except Exception as e:
        return f"Error retrieving protocol: {str(e)}"

@mcp.tool()
def protocol_list(status: Optional[str] = None) -> str:
    """List protocols."""
    try:
        protocols = protocol_ops.list_protocols(status)
        if not protocols:
            return "No protocols found."
            
        output = [f"Found {len(protocols)} protocol(s):"]
        for p in protocols:
            output.append(f"- {p['number']:03d}: {p['title']} [{p['status']}] v{p['version']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing protocols: {str(e)}"

@mcp.tool()
def protocol_search(query: str) -> str:
    """Search protocols by content."""
    try:
        results = protocol_ops.search_protocols(query)
        if not results:
            return f"No protocols found matching '{query}'"
            
        output = [f"Found {len(results)} protocol(s) matching '{query}':"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching protocols: {str(e)}"

# =============================================================================
# TASK TOOLS
# =============================================================================

@mcp.tool()
def create_task(
    title: str,
    objective: str,
    deliverables: List[str],
    acceptance_criteria: List[str],
    priority: str = "Medium",
    status: str = "backlog",
    lead: str = "Unassigned",
    dependencies: Optional[str] = None,
    related_documents: Optional[str] = None,
    notes: Optional[str] = None,
    task_number: Optional[int] = None
) -> str:
    """Create a new task file in TASKS/ directory."""
    try:
        from mcp_servers.task.models import TaskStatus, TaskPriority
        
        result = task_ops.create_task(
            title=title,
            objective=objective,
            deliverables=deliverables,
            acceptance_criteria=acceptance_criteria,
            priority=TaskPriority(priority),
            status=TaskStatus(status),
            lead=lead,
            dependencies=dependencies,
            related_documents=related_documents,
            notes=notes,
            task_number=task_number
        )
        
        return f"Created Task {result.task_number:03d}: {result.file_path}"
    except Exception as e:
        return f"Error creating task: {str(e)}"

@mcp.tool()
def update_task(
    task_number: int,
    updates: Dict[str, Any]
) -> str:
    """Update an existing task's metadata or content."""
    try:
        result = task_ops.update_task(task_number, updates)
        updated_fields = list(updates.keys())
        return f"Updated Task {result.task_number:03d}. Fields: {', '.join(updated_fields)}"
    except Exception as e:
        return f"Error updating task: {str(e)}"

@mcp.tool()
def update_task_status(
    task_number: int,
    new_status: str,
    notes: Optional[str] = None
) -> str:
    """Change task status (moves file between directories)."""
    try:
        from mcp_servers.task.models import TaskStatus
        result = task_ops.update_task_status(
            task_number,
            TaskStatus(new_status),
            notes
        )
        return f"Updated Task {result.task_number:03d} status to {new_status}"
    except Exception as e:
        return f"Error updating task status: {str(e)}"

@mcp.tool()
def get_task(task_number: int) -> str:
    """Retrieve a specific task by number."""
    try:
        task = task_ops.get_task(task_number)
        
        if task is None:
            return f"Task #{task_number:03d} not found"
        
        output = []
        output.append(f"Task {task.get('number', task_number):03d}: {task.get('title', 'Untitled')}")
        output.append(f"Status: {task.get('status', 'unknown')}")
        output.append(f"Priority: {task.get('priority', 'unknown')}")
        output.append(f"Lead: {task.get('lead', 'Unassigned')}")
        
        if 'objective' in task and task['objective']:
            output.append(f"\nObjective:\n{task['objective']}")
        
        if 'deliverables' in task and task['deliverables']:
            output.append(f"\nDeliverables:")
            for d in task['deliverables']:
                output.append(f"- {d}")
        
        if 'acceptance_criteria' in task and task['acceptance_criteria']:
            output.append(f"\nAcceptance Criteria:")
            for c in task['acceptance_criteria']:
                output.append(f"- {c}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error retrieving task: {str(e)}"

@mcp.tool()
def list_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None
) -> str:
    """List tasks with optional filters."""
    try:
        from mcp_servers.task.models import TaskStatus, TaskPriority
        status_filter = TaskStatus(status) if status else None
        priority_filter = TaskPriority(priority) if priority else None
        
        tasks = task_ops.list_tasks(status_filter, priority_filter)
        
        if not tasks:
            filter_desc = []
            if status:
                filter_desc.append(f"status={status}")
            if priority:
                filter_desc.append(f"priority={priority}")
            filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
            return f"No tasks found{filter_str}"
        
        output = [f"Found {len(tasks)} task(s):"]
        for task in tasks:
            output.append(f"- {task['number']:03d}: {task['title']} [{task['status']}] ({task['priority']})")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error listing tasks: {str(e)}"

@mcp.tool()
def search_tasks(query: str) -> str:
    """Search tasks by content (full-text search)."""
    try:
        results = task_ops.search_tasks(query)
        if not results:
            return f"No tasks found matching '{query}'"
        
        output = [f"Found {len(results)} task(s) matching '{query}':"]
        for task in results:
            output.append(f"- {task['number']:03d}: {task['title']} [{task['status']}]")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error searching tasks: {str(e)}"

# =============================================================================
# ADR TOOLS
# =============================================================================

@mcp.tool()
def adr_create(
    title: str,
    context: str,
    decision: str,
    consequences: str,
    date: Optional[str] = None,
    status: str = "proposed",
    author: str = "AI Assistant",
    supersedes: Optional[int] = None
) -> str:
    """Create a new ADR with automatic sequential numbering."""
    try:
        result = adr_ops.create_adr(
            title=title,
            context=context,
            decision=decision,
            consequences=consequences,
            date=date,
            status=status,
            author=author,
            supersedes=supersedes
        )
        return f"Created ADR {result['adr_number']:03d}: {result['file_path']}"
    except Exception as e:
        return f"Error creating ADR: {str(e)}"

@mcp.tool()
def adr_update_status(number: int, new_status: str, reason: str) -> str:
    """Update the status of an existing ADR."""
    try:
        result = adr_ops.update_adr_status(number, new_status, reason)
        return (
            f"Updated ADR {result['adr_number']:03d}: "
            f"{result['old_status']} → {result['new_status']} "
            f"(Reason: {reason})"
        )
    except Exception as e:
        return f"Error updating ADR status: {str(e)}"

@mcp.tool()
def adr_get(number: int) -> str:
    """Retrieve a specific ADR by number."""
    try:
        adr = adr_ops.get_adr(number)
        return (
            f"ADR {adr['number']:03d}: {adr['title']}\n"
            f"Status: {adr['status']}\n"
            f"Date: {adr['date']}\n"
            f"Author: {adr['author']}\n\n"
            f"Context:\n{adr['context']}\n\n"
            f"Decision:\n{adr['decision']}\n\n"
            f"Consequences:\n{adr['consequences']}"
        )
    except Exception as e:
        return f"Error retrieving ADR: {str(e)}"

@mcp.tool()
def adr_list(status: Optional[str] = None) -> str:
    """List all ADRs with optional status filter."""
    try:
        adrs = adr_ops.list_adrs(status)
        if not adrs:
            return "No ADRs found" + (f" with status '{status}'" if status else "")
        
        result = f"Found {len(adrs)} ADR(s)" + (f" with status '{status}'" if status else "") + ":\n\n"
        for adr in adrs:
            result += f"ADR {adr['number']:03d}: {adr['title']} [{adr['status']}] ({adr['date']})\n"
        
        return result
    except Exception as e:
        return f"Error listing ADRs: {str(e)}"

@mcp.tool()
def adr_search(query: str) -> str:
    """Full-text search across all ADRs."""
    try:
        results = adr_ops.search_adrs(query)
        if not results:
            return f"No ADRs found matching '{query}'"
        
        output = f"Found {len(results)} ADR(s) matching '{query}':\n\n"
        for result in results:
            output += f"ADR {result['number']:03d}: {result['title']}\n"
            for match in result['matches']:
                output += f"  - {match}\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"Error searching ADRs: {str(e)}"

# =============================================================================
# PERSONA TOOLS
# =============================================================================

@mcp.tool()
def persona_dispatch(
    role: str,
    task: str,
    context: Optional[dict] = None,
    maintain_state: bool = True,
    engine: Optional[str] = None,
    model_name: Optional[str] = None,
    custom_persona_file: Optional[str] = None
) -> dict:
    """Dispatch a task to a specific persona agent."""
    return persona_ops.dispatch(
        role=role, task=task, context=context, maintain_state=maintain_state,
        engine=engine, model_name=model_name, custom_persona_file=custom_persona_file
    )

@mcp.tool()
def persona_list_roles() -> dict:
    """List all available persona roles (built-in and custom)."""
    return persona_ops.list_roles()

@mcp.tool()
def persona_get_state(role: str) -> dict:
    """Get conversation state for a specific persona role."""
    return persona_ops.get_state(role=role)

@mcp.tool()
def persona_reset_state(role: str) -> dict:
    """Reset conversation state for a specific persona role."""
    return persona_ops.reset_state(role=role)

@mcp.tool()
def persona_create_custom(role: str, persona_definition: str, description: str) -> dict:
    """Create a new custom persona."""
    return persona_ops.create_custom(
        role=role, persona_definition=persona_definition, description=description
    )

# =============================================================================
# CONFIG TOOLS
# =============================================================================

@mcp.tool()
def config_list() -> str:
    """List all configuration files in the .agent/config directory."""
    try:
        configs = config_ops.list_configs()
        if not configs: return "No configuration files found."
        output = [f"Found {len(configs)} configuration files:"]
        for c in configs:
            output.append(f"- {c['name']} ({c['size']} bytes, {c['modified']})")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing configs: {str(e)}"

@mcp.tool()
def config_read(filename: str) -> str:
    """Read a configuration file."""
    try:
        content = config_ops.read_config(filename)
        if isinstance(content, (dict, list)):
            import json
            return json.dumps(content, indent=2)
        return str(content)
    except Exception as e:
        return f"Error reading config '{filename}': {str(e)}"

@mcp.tool()
def config_write(filename: str, content: str) -> str:
    """Write a configuration file."""
    try:
        import json
        if filename.endswith('.json'):
            try:
                data = json.loads(content)
                path = config_ops.write_config(filename, data)
            except json.JSONDecodeError:
                path = config_ops.write_config(filename, content)
        else:
            path = config_ops.write_config(filename, content)
        return f"Successfully wrote config to {path}"
    except Exception as e:
        return f"Error writing config '{filename}': {str(e)}"

@mcp.tool()
def config_delete(filename: str) -> str:
    """Delete a configuration file."""
    try:
        config_ops.delete_config(filename)
        return f"Successfully deleted config '{filename}'"
    except Exception as e:
        return f"Error deleting config '{filename}': {str(e)}"

# =============================================================================
# WORKFLOW TOOLS
# =============================================================================

@mcp.tool()
def get_available_workflows() -> str:
    """List all available workflows in the .agent/workflows directory."""
    try:
        workflows = workflow_ops.list_workflows()
        if not workflows:
            return "No workflows found in .agent/workflows."
        
        output = [f"Found {len(workflows)} available workflow(s):"]
        for wf in workflows:
            turbo = " [TURBO]" if wf.get('turbo_mode') else ""
            output.append(f"- {wf['filename']}{turbo}: {wf['description']}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error listing workflows: {str(e)}"

@mcp.tool()
def read_workflow(filename: str) -> str:
    """Read the content of a specific workflow file."""
    try:
        content = workflow_ops.get_workflow_content(filename)
        if content is None:
            return f"Workflow '{filename}' not found."
        return content
    except Exception as e:
        return f"Error reading workflow: {str(e)}"

# =============================================================================
# PYTHON DEV TOOLS (from Code MCP)
# =============================================================================

@mcp.tool()
async def code_lint(path: str, tool: str = "ruff") -> str:
    """Run linting on a file or directory."""
    try:
        result = code_ops.lint(path, tool)
        output = [f"Linting {result['path']} with {result['tool']}:", ""]
        if result['issues_found']:
            output.append("❌ Issues found:")
            output.append(result['output'])
        else:
            output.append("✅ No issues found")
        return "\n".join(output)
    except Exception as e:
        return f"Error linting '{path}': {str(e)}"

@mcp.tool()
async def code_format(path: str, tool: str = "ruff", check_only: bool = False) -> str:
    """Format code in a file or directory."""
    try:
        result = code_ops.format_code(path, tool, check_only)
        output = [f"Formatting {result['path']} with {result['tool']}:", ""]
        if check_only:
            if result['success']:
                output.append("✅ Code is properly formatted")
            else:
                output.append("❌ Code needs formatting")
                output.append(result['output'])
        else:
            if result['modified']:
                output.append("✅ Code formatted successfully")
            else:
                output.append("❌ Formatting failed")
                output.append(result['output'])
        return "\n".join(output)
    except Exception as e:
        return f"Error formatting '{path}': {str(e)}"

@mcp.tool()
async def code_analyze(path: str) -> str:
    """Perform static analysis on code."""
    try:
        result = code_ops.analyze(path)
        output = [f"Analyzing {result['path']}:", "", result['statistics']]
        return "\n".join(output)
    except Exception as e:
        return f"Error analyzing '{path}': {str(e)}"

@mcp.tool()
async def code_check_tools() -> str:
    """Check which code quality tools are available."""
    tools = ["ruff", "black", "pylint", "flake8", "mypy"]
    available = []
    unavailable = []
    for tool in tools:
        if code_ops.check_tool_available(tool):
            available.append(f"✅ {tool}")
        else:
            unavailable.append(f"❌ {tool}")
    output = ["Available code tools:", ""]
    output.extend(available)
    if unavailable:
        output.append("")
        output.append("Unavailable:")
        output.extend(unavailable)
    return "\n".join(output)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Robust Entry Point:
    # 1. Check Environment Variables (Docker Override)
    # 2. Default to stdio for local piping
    
    port_env = os.environ.get("DOMAIN_SERVER_PORT")
    # If running in Docker (port set), default to SSE. Else stdio.
    
    if port_env:
        # Docker Mode: Listen on 0.0.0.0
        print(f"🚀 Starting Domain Server on port {port_env} (Transport: SSE)")
        mcp.run(transport="sse", port=int(port_env), host="0.0.0.0")
    else:
        # Local Mode: Stdio
        mcp.run()
