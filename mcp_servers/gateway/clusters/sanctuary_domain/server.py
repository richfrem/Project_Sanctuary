#============================================
# mcp_servers/gateway/clusters/sanctuary_domain/server.py
# Purpose: Sanctuary Domain Logic Cluster - Dual-Transport Entry Point
# Role: Interface Layer (Aggregator Node)
# Status: ADR-066 v1.3 Compliant (SSEServer for Gateway, FastMCP for STDIO)
# Used by: Gateway Fleet (SSE) and Claude Desktop (STDIO)
#============================================

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging

# Setup Logging
logger = setup_mcp_logging("project_sanctuary.sanctuary_domain")

# Configuration
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)

# Lazy Instance Store
_ops = {}

def get_op(op_type):
    if op_type not in _ops:
        if op_type == "chronicle":
            from mcp_servers.chronicle.operations import ChronicleOperations
            _ops[op_type] = ChronicleOperations(os.path.join(PROJECT_ROOT, "00_CHRONICLE/ENTRIES"))
        elif op_type == "protocol":
            from mcp_servers.protocol.operations import ProtocolOperations
            _ops[op_type] = ProtocolOperations(os.path.join(PROJECT_ROOT, "01_PROTOCOLS"))
        elif op_type == "task":
            from mcp_servers.task.operations import TaskOperations
            _ops[op_type] = TaskOperations(PROJECT_ROOT_PATH)
        elif op_type == "adr":
            from mcp_servers.adr.operations import ADROperations
            _ops[op_type] = ADROperations(os.path.join(PROJECT_ROOT, "ADRs"))
        elif op_type == "persona":
            from mcp_servers.agent_persona.operations import PersonaOperations
            _ops[op_type] = PersonaOperations()
        elif op_type == "config":
            from mcp_servers.config.operations import ConfigOperations
            _ops[op_type] = ConfigOperations(os.path.join(PROJECT_ROOT, ".agent/config"))
        elif op_type == "workflow":
            from mcp_servers.workflow.operations import WorkflowOperations
            _ops[op_type] = WorkflowOperations(PROJECT_ROOT_PATH / ".agent/workflows")
    return _ops[op_type]


#============================================
# Tool Schema Definitions (for SSEServer)
#============================================
# Chronicle Schemas
CHRONICLE_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "content": {"type": "string"},
        "author": {"type": "string"},
        "date": {"type": "string"},
        "status": {"type": "string"},
        "classification": {"type": "string"}
    },
    "required": ["title", "content"]
}

CHRONICLE_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_number": {"type": "integer"},
        "updates": {"type": "object"},
        "reason": {"type": "string"},
        "override_approval_id": {"type": "string"}
    },
    "required": ["entry_number", "updates"]
}

CHRONICLE_GET_SCHEMA = {"type": "object", "properties": {"entry_number": {"type": "integer"}}, "required": ["entry_number"]}
CHRONICLE_LIST_SCHEMA = {"type": "object", "properties": {"limit": {"type": "integer"}}}
CHRONICLE_SEARCH_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

# Protocol Schemas
PROTOCOL_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {"type": "integer"},
        "title": {"type": "string"},
        "status": {"type": "string"},
        "classification": {"type": "string"},
        "version": {"type": "string"},
        "authority": {"type": "string"},
        "content": {"type": "string"},
        "linked_protocols": {"type": "array", "items": {"type": "integer"}}
    },
    "required": ["title", "content"]
}

PROTOCOL_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {"number": {"type": "integer"}, "updates": {"type": "object"}, "reason": {"type": "string"}},
    "required": ["number", "updates"]
}

PROTOCOL_GET_SCHEMA = {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}
PROTOCOL_LIST_SCHEMA = {"type": "object", "properties": {"status": {"type": "string"}}}
PROTOCOL_SEARCH_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

# Task Schemas
TASK_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "objective": {"type": "string"},
        "deliverables": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "string"}, "status": {"type": "string"},
        "lead": {"type": "string"}, "dependencies": {"type": "array"},
        "related_documents": {"type": "array"}, "notes": {"type": "string"},
        "task_number": {"type": "integer"}
    },
    "required": ["title", "objective"]
}

TASK_UPDATE_SCHEMA = {"type": "object", "properties": {"task_number": {"type": "integer"}, "updates": {"type": "object"}}, "required": ["task_number", "updates"]}
TASK_UPDATE_STATUS_SCHEMA = {"type": "object", "properties": {"task_number": {"type": "integer"}, "new_status": {"type": "string"}, "notes": {"type": "string"}}, "required": ["task_number", "new_status"]}
TASK_GET_SCHEMA = {"type": "object", "properties": {"task_number": {"type": "integer"}}, "required": ["task_number"]}
TASK_LIST_SCHEMA = {"type": "object", "properties": {"status": {"type": "string"}, "priority": {"type": "string"}}}
TASK_SEARCH_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

# ADR Schemas
ADR_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"}, "context": {"type": "string"},
        "decision": {"type": "string"}, "consequences": {"type": "string"},
        "date": {"type": "string"}, "status": {"type": "string"},
        "author": {"type": "string"}, "supersedes": {"type": "integer"}
    },
    "required": ["title", "context", "decision", "consequences"]
}

ADR_UPDATE_STATUS_SCHEMA = {"type": "object", "properties": {"number": {"type": "integer"}, "new_status": {"type": "string"}, "reason": {"type": "string"}}, "required": ["number", "new_status", "reason"]}
ADR_GET_SCHEMA = {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}
ADR_LIST_SCHEMA = {"type": "object", "properties": {"status": {"type": "string"}}}
ADR_SEARCH_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

# Persona Schemas
PERSONA_DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string"}, "task": {"type": "string"}, "context": {"type": "string"},
        "maintain_state": {"type": "boolean"}, "engine": {"type": "string"},
        "model_name": {"type": "string"}, "custom_persona_file": {"type": "string"}
    },
    "required": ["role", "task"]
}

PERSONA_ROLE_SCHEMA = {"type": "object", "properties": {"role": {"type": "string"}}, "required": ["role"]}
PERSONA_CREATE_CUSTOM_SCHEMA = {"type": "object", "properties": {"role": {"type": "string"}, "persona_definition": {"type": "string"}, "description": {"type": "string"}}, "required": ["role", "persona_definition"]}

# Config Schemas
CONFIG_READ_SCHEMA = {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}
CONFIG_WRITE_SCHEMA = {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]}
CONFIG_DELETE_SCHEMA = {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}

# Workflow Schemas
WORKFLOW_READ_SCHEMA = {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}

EMPTY_SCHEMA = {"type": "object", "properties": {}}


#============================================
# SSE Transport Implementation (Gateway Mode)
# Migrated to @sse_tool decorator pattern per ADR-076
#============================================
def run_sse_server(port: int):
    """Run using SSEServer for Gateway compatibility (ADR-066 v1.3)."""
    from mcp_servers.lib.sse_adaptor import SSEServer, sse_tool
    from mcp_servers.task.models import TaskStatus, TaskPriority
    
    server = SSEServer("sanctuary_domain", version="1.0.0")
    
    # =============================================================================
    # CHRONICLE TOOLS
    # =============================================================================
    @sse_tool(name="chronicle_create_entry", description="Create a new chronicle entry.", schema=CHRONICLE_CREATE_SCHEMA)
    def chronicle_create_entry(title: str, content: str, author: str = "Agent", date: str = None, status: str = "draft", classification: str = "internal"):
        result = get_op("chronicle").create_entry(title=title, content=content, author=author, date=date, status=status, classification=classification)
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    
    @sse_tool(name="chronicle_append_entry", description="Append a new entry to the Chronicle (Alias for create_entry).", schema=CHRONICLE_CREATE_SCHEMA)
    def chronicle_append_entry(title: str, content: str, author: str = "Agent", date: str = None, status: str = "draft", classification: str = "internal"):
        result = get_op("chronicle").create_entry(title=title, content=content, author=author, date=date, status=status, classification=classification)
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    
    @sse_tool(name="chronicle_update_entry", description="Update an existing chronicle entry.", schema=CHRONICLE_UPDATE_SCHEMA)
    def chronicle_update_entry(entry_number: int, updates: dict, reason: str = None, override_approval_id: str = None):
        result = get_op("chronicle").update_entry(entry_number=entry_number, updates=updates, reason=reason, override_approval_id=override_approval_id)
        return f"Updated Chronicle Entry {result['entry_number']}. Fields: {', '.join(result['updated_fields'])}"
    
    @sse_tool(name="chronicle_get_entry", description="Retrieve a specific chronicle entry.", schema=CHRONICLE_GET_SCHEMA)
    def chronicle_get_entry(entry_number: int):
        entry = get_op("chronicle").get_entry(entry_number)
        return f"Entry {entry['number']}: {entry['title']}\nDate: {entry['date']}\nAuthor: {entry['author']}\nStatus: {entry['status']}\n\n{entry['content']}"
    
    @sse_tool(name="chronicle_list_entries", description="List recent chronicle entries.", schema=CHRONICLE_LIST_SCHEMA)
    def chronicle_list_entries(limit: int = 10):
        entries = get_op("chronicle").list_entries(limit)
        if not entries: return "No entries found."
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    
    @sse_tool(name="chronicle_read_latest_entries", description="Read the latest entries from the Chronicle.", schema=CHRONICLE_LIST_SCHEMA)
    def chronicle_read_latest_entries(limit: int = 10):
        return chronicle_list_entries(limit)
    
    @sse_tool(name="chronicle_search", description="Search chronicle entries by content.", schema=CHRONICLE_SEARCH_SCHEMA)
    def chronicle_search(query: str):
        results = get_op("chronicle").search_entries(query)
        if not results: return f"No entries found matching '{query}'"
        output = [f"Found {len(results)} entries:"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    
    # =============================================================================
    # PROTOCOL TOOLS
    # =============================================================================
    @sse_tool(name="protocol_create", description="Create a new protocol.", schema=PROTOCOL_CREATE_SCHEMA)
    def protocol_create(title: str, content: str, number: int = None, status: str = "proposed", classification: str = "internal", version: str = "1.0.0", authority: str = "System", linked_protocols: list = None):
        result = get_op("protocol").create_protocol(number=number, title=title, status=status, classification=classification, version=version, authority=authority, content=content, linked_protocols=linked_protocols)
        return f"Created Protocol {result['protocol_number']}: {result['file_path']}"
    
    @sse_tool(name="protocol_update", description="Update an existing protocol.", schema=PROTOCOL_UPDATE_SCHEMA)
    def protocol_update(number: int, updates: dict, reason: str = None):
        result = get_op("protocol").update_protocol(number=number, updates=updates, reason=reason)
        return f"Updated Protocol {result['protocol_number']}. Fields: {', '.join(result['updated_fields'])}"
    
    @sse_tool(name="protocol_get", description="Retrieve a specific protocol.", schema=PROTOCOL_GET_SCHEMA)
    def protocol_get(number: int):
        protocol = get_op("protocol").get_protocol(number)
        return f"Protocol {protocol['number']}: {protocol['title']}\nStatus: {protocol['status']}\nVersion: {protocol['version']}\n\n{protocol['content']}"
    
    @sse_tool(name="protocol_list", description="List protocols.", schema=PROTOCOL_LIST_SCHEMA)
    def protocol_list(status: str = None):
        protocols = get_op("protocol").list_protocols(status)
        if not protocols: return "No protocols found."
        output = [f"Found {len(protocols)} protocol(s):"]
        for p in protocols:
            output.append(f"- {p['number']:03d}: {p['title']} [{p['status']}] v{p['version']}")
        return "\n".join(output)
    
    @sse_tool(name="protocol_search", description="Search protocols by content.", schema=PROTOCOL_SEARCH_SCHEMA)
    def protocol_search(query: str):
        results = get_op("protocol").search_protocols(query)
        if not results: return f"No protocols found matching '{query}'"
        output = [f"Found {len(results)} protocol(s):"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    
    # =============================================================================
    # TASK TOOLS
    # =============================================================================
    @sse_tool(name="create_task", description="Create a new task file in TASKS/ directory.", schema=TASK_CREATE_SCHEMA)
    def create_task(title: str, objective: str, deliverables: list = None, acceptance_criteria: list = None, priority: str = "medium", status: str = "backlog", lead: str = None, dependencies: list = None, related_documents: list = None, notes: str = None, task_number: int = None):
        result = get_op("task").create_task(title=title, objective=objective, deliverables=deliverables, acceptance_criteria=acceptance_criteria, priority=TaskPriority(priority), status=TaskStatus(status), lead=lead, dependencies=dependencies, related_documents=related_documents, notes=notes, task_number=task_number)
        return f"Created Task {result.task_number:03d}: {result.file_path}"
    
    @sse_tool(name="update_task", description="Update an existing task's metadata or content.", schema=TASK_UPDATE_SCHEMA)
    def update_task(task_number: int, updates: dict):
        result = get_op("task").update_task(task_number, updates)
        return f"Updated Task {result.task_number:03d}. Fields: {', '.join(updates.keys())}"
    
    @sse_tool(name="update_task_status", description="Change task status (moves file between directories).", schema=TASK_UPDATE_STATUS_SCHEMA)
    def update_task_status(task_number: int, new_status: str, notes: str = None):
        result = get_op("task").update_task_status(task_number, TaskStatus(new_status), notes)
        return f"Updated Task {result.task_number:03d} status to {new_status}"
    
    @sse_tool(name="get_task", description="Retrieve a specific task by number.", schema=TASK_GET_SCHEMA)
    def get_task(task_number: int):
        task = get_op("task").get_task(task_number)
        if not task: return f"Task #{task_number:03d} not found"
        return f"Task {task.get('number', task_number):03d}: {task.get('title', 'Untitled')}\nStatus: {task.get('status')}\nPriority: {task.get('priority')}\n\nObjective:\n{task.get('objective', '')}"
    
    @sse_tool(name="list_tasks", description="List tasks with optional filters.", schema=TASK_LIST_SCHEMA)
    def list_tasks(status: str = None, priority: str = None):
        status_filter = TaskStatus(status) if status else None
        priority_filter = TaskPriority(priority) if priority else None
        tasks = get_op("task").list_tasks(status_filter, priority_filter)
        if not tasks: return "No tasks found."
        output = [f"Found {len(tasks)} task(s):"]
        for t in tasks:
            output.append(f"- {t['number']:03d}: {t['title']} [{t['status']}] ({t['priority']})")
        return "\n".join(output)
    
    @sse_tool(name="search_tasks", description="Search tasks by content (full-text search).", schema=TASK_SEARCH_SCHEMA)
    def search_tasks(query: str):
        results = get_op("task").search_tasks(query)
        if not results: return f"No tasks found matching '{query}'"
        output = [f"Found {len(results)} task(s):"]
        for t in results:
            output.append(f"- {t['number']:03d}: {t['title']} [{t['status']}]")
        return "\n".join(output)
    
    # =============================================================================
    # ADR TOOLS
    # =============================================================================
    @sse_tool(name="adr_create", description="Create a new ADR with automatic sequential numbering.", schema=ADR_CREATE_SCHEMA)
    def adr_create(title: str, context: str, decision: str, consequences: str, date: str = None, status: str = "proposed", author: str = "Agent", supersedes: int = None):
        result = get_op("adr").create_adr(title=title, context=context, decision=decision, consequences=consequences, date=date, status=status, author=author, supersedes=supersedes)
        return f"Created ADR {result['adr_number']:03d}: {result['file_path']}"
    
    @sse_tool(name="adr_update_status", description="Update the status of an existing ADR.", schema=ADR_UPDATE_STATUS_SCHEMA)
    def adr_update_status(number: int, new_status: str, reason: str):
        result = get_op("adr").update_adr_status(number, new_status, reason)
        return f"Updated ADR {result['adr_number']:03d}: {result['old_status']} → {result['new_status']}"
    
    @sse_tool(name="adr_get", description="Retrieve a specific ADR by number.", schema=ADR_GET_SCHEMA)
    def adr_get(number: int):
        adr = get_op("adr").get_adr(number)
        return f"ADR {adr['number']:03d}: {adr['title']}\nStatus: {adr['status']}\nDate: {adr['date']}\n\nContext:\n{adr['context']}\n\nDecision:\n{adr['decision']}"
    
    @sse_tool(name="adr_list", description="List all ADRs with optional status filter.", schema=ADR_LIST_SCHEMA)
    def adr_list(status: str = None):
        adrs = get_op("adr").list_adrs(status)
        if not adrs: return "No ADRs found"
        result = f"Found {len(adrs)} ADR(s):\n\n"
        for adr in adrs:
            result += f"ADR {adr['number']:03d}: {adr['title']} [{adr['status']}]\n"
        return result
    
    @sse_tool(name="adr_search", description="Full-text search across all ADRs.", schema=ADR_SEARCH_SCHEMA)
    def adr_search(query: str):
        results = get_op("adr").search_adrs(query)
        if not results: return f"No ADRs found matching '{query}'"
        output = f"Found {len(results)} ADR(s):\n\n"
        for result in results:
            output += f"ADR {result['number']:03d}: {result['title']}\n"
        return output
    
    # =============================================================================
    # PERSONA TOOLS
    # =============================================================================
    @sse_tool(name="persona_dispatch", description="Dispatch a task to a specific persona agent.", schema=PERSONA_DISPATCH_SCHEMA)
    def persona_dispatch(role: str, task: str, context: str = None, maintain_state: bool = True, engine: str = None, model_name: str = None, custom_persona_file: str = None):
        result = get_op("persona").dispatch(role=role, task=task, context=context, maintain_state=maintain_state, engine=engine, model_name=model_name, custom_persona_file=custom_persona_file)
        return json.dumps(result, indent=2)
    
    @sse_tool(name="persona_list_roles", description="List all available persona roles.", schema=EMPTY_SCHEMA)
    def persona_list_roles():
        return json.dumps(get_op("persona").list_roles(), indent=2)
    
    @sse_tool(name="persona_get_state", description="Get conversation state for a specific persona role.", schema=PERSONA_ROLE_SCHEMA)
    def persona_get_state(role: str):
        return json.dumps(get_op("persona").get_state(role=role), indent=2)
    
    @sse_tool(name="persona_reset_state", description="Reset conversation state for a specific persona role.", schema=PERSONA_ROLE_SCHEMA)
    def persona_reset_state(role: str):
        return json.dumps(get_op("persona").reset_state(role=role), indent=2)
    
    @sse_tool(name="persona_create_custom", description="Create a new custom persona.", schema=PERSONA_CREATE_CUSTOM_SCHEMA)
    def persona_create_custom(role: str, persona_definition: str, description: str = None):
        return json.dumps(get_op("persona").create_custom(role=role, persona_definition=persona_definition, description=description), indent=2)
    
    # =============================================================================
    # CONFIG TOOLS
    # =============================================================================
    @sse_tool(name="config_list", description="List all configuration files in the .agent/config directory.", schema=EMPTY_SCHEMA)
    def config_list():
        configs = get_op("config").list_configs()
        if not configs: return "No configuration files found."
        output = [f"Found {len(configs)} configuration files:"]
        for c in configs:
            output.append(f"- {c['name']} ({c['size']} bytes)")
        return "\n".join(output)
    
    @sse_tool(name="config_read", description="Read a configuration file.", schema=CONFIG_READ_SCHEMA)
    def config_read(filename: str):
        content = get_op("config").read_config(filename)
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2)
        return str(content)
    
    @sse_tool(name="config_write", description="Write a configuration file.", schema=CONFIG_WRITE_SCHEMA)
    def config_write(filename: str, content: str):
        path = get_op("config").write_config(filename, content)
        return f"Successfully wrote config to {path}"
    
    @sse_tool(name="config_delete", description="Delete a configuration file.", schema=CONFIG_DELETE_SCHEMA)
    def config_delete(filename: str):
        get_op("config").delete_config(filename)
        return f"Successfully deleted config '{filename}'"
    
    # =============================================================================
    # WORKFLOW TOOLS
    # =============================================================================
    @sse_tool(name="get_available_workflows", description="List all available workflows in the .agent/workflows directory.", schema=EMPTY_SCHEMA)
    def get_available_workflows():
        workflows = get_op("workflow").list_workflows()
        if not workflows: return "No workflows found."
        output = [f"Found {len(workflows)} available workflow(s):"]
        for wf in workflows:
            turbo = " [TURBO]" if wf.get('turbo_mode') else ""
            output.append(f"- {wf['filename']}{turbo}: {wf['description']}")
        return "\n".join(output)
    
    @sse_tool(name="read_workflow", description="Read the content of a specific workflow file.", schema=WORKFLOW_READ_SCHEMA)
    def read_workflow(filename: str):
        content = get_op("workflow").get_workflow_content(filename)
        if content is None:
            return f"Workflow '{filename}' not found."
        return content
    
    # Auto-register all decorated tools (ADR-076)
    server.register_decorated_tools(locals())
    
    logger.info(f"Starting SSEServer on port {port} (Gateway Mode) with 34 tools")
    server.run(port=port, transport="sse")


#============================================
# STDIO Transport Implementation (Local Mode)
#============================================
def run_stdio_server():
    """Run using FastMCP for local development (Claude Desktop)."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from mcp_servers.chronicle.models import ChronicleCreateRequest, ChronicleUpdateRequest, ChronicleGetRequest, ChronicleListRequest, ChronicleSearchRequest
    from mcp_servers.protocol.models import ProtocolCreateRequest, ProtocolUpdateRequest, ProtocolGetRequest, ProtocolListRequest, ProtocolSearchRequest
    from mcp_servers.task.models import TaskCreateRequest, TaskUpdateRequest, TaskUpdateStatusRequest, TaskGetRequest, TaskListRequest, TaskSearchRequest, TaskStatus, TaskPriority
    from mcp_servers.adr.models import ADRCreateRequest, ADRUpdateStatusRequest, ADRGetRequest, ADRListRequest, ADRSearchRequest
    from mcp_servers.agent_persona.models import PersonaDispatchParams, PersonaRoleParams, PersonaCreateCustomParams
    from mcp_servers.config.models import ConfigReadRequest, ConfigWriteRequest, ConfigDeleteRequest
    from mcp_servers.workflow.models import WorkflowReadRequest
    
    mcp = FastMCP(
        "sanctuary_domain",
        instructions="""
        Sanctuary Domain Cluster Aggregator.
        - Aggregates Chronicle, Protocol, Task, ADR, Persona, Config, and Workflow tools.
        - Acts as the primary domain logic interface for Project Sanctuary.
        """
    )
    
    # Chronicle Tools
    @mcp.tool()
    async def chronicle_create_entry(request: ChronicleCreateRequest) -> str:
        """Create a new chronicle entry."""
        try:
            result = get_op("chronicle").create_entry(title=request.title, content=request.content, author=request.author, date=request.date, status=request.status, classification=request.classification)
            return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
        except Exception as e:
            raise ToolError(f"Creation failed: {str(e)}")
    
    @mcp.tool()
    async def chronicle_append_entry(request: ChronicleCreateRequest) -> str:
        """Append a new entry to the Chronicle (Alias for create_entry)."""
        return await chronicle_create_entry(request)
    
    @mcp.tool()
    async def chronicle_update_entry(request: ChronicleUpdateRequest) -> str:
        """Update an existing chronicle entry."""
        try:
            result = get_op("chronicle").update_entry(entry_number=request.entry_number, updates=request.updates, reason=request.reason, override_approval_id=request.override_approval_id)
            return f"Updated Chronicle Entry {result['entry_number']}. Fields: {', '.join(result['updated_fields'])}"
        except Exception as e:
            raise ToolError(f"Update failed: {str(e)}")
    
    @mcp.tool()
    async def chronicle_get_entry(request: ChronicleGetRequest) -> str:
        """Retrieve a specific chronicle entry."""
        try:
            entry = get_op("chronicle").get_entry(request.entry_number)
            return f"Entry {entry['number']}: {entry['title']}\nDate: {entry['date']}\nAuthor: {entry['author']}\nStatus: {entry['status']}\n\n{entry['content']}"
        except Exception as e:
            raise ToolError(f"Retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def chronicle_list_entries(request: Optional[ChronicleListRequest] = None) -> str:
        """List recent chronicle entries."""
        request = request or ChronicleListRequest()
        try:
            entries = get_op("chronicle").list_entries(request.limit)
            if not entries: return "No entries found."
            output = [f"Found {len(entries)} recent entries:"]
            for e in entries:
                output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"List failed: {str(e)}")
    
    @mcp.tool()
    async def chronicle_read_latest_entries(request: Optional[ChronicleListRequest] = None) -> str:
        """Read the latest entries from the Chronicle."""
        return await chronicle_list_entries(request)
    
    @mcp.tool()
    async def chronicle_search(request: ChronicleSearchRequest) -> str:
        """Search chronicle entries by content."""
        try:
            results = get_op("chronicle").search_entries(request.query)
            if not results: return f"No entries found matching '{request.query}'"
            output = [f"Found {len(results)} entries:"]
            for r in results:
                output.append(f"- {r['number']:03d}: {r['title']}")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Search failed: {str(e)}")
    
    # Protocol Tools
    @mcp.tool()
    async def protocol_create(request: ProtocolCreateRequest) -> str:
        """Create a new protocol."""
        try:
            result = get_op("protocol").create_protocol(number=request.number, title=request.title, status=request.status, classification=request.classification, version=request.version, authority=request.authority, content=request.content, linked_protocols=request.linked_protocols)
            return f"Created Protocol {result['protocol_number']}: {result['file_path']}"
        except Exception as e:
            raise ToolError(f"Creation failed: {str(e)}")
    
    @mcp.tool()
    async def protocol_update(request: ProtocolUpdateRequest) -> str:
        """Update an existing protocol."""
        try:
            result = get_op("protocol").update_protocol(number=request.number, updates=request.updates, reason=request.reason)
            return f"Updated Protocol {result['protocol_number']}. Fields: {', '.join(result['updated_fields'])}"
        except Exception as e:
            raise ToolError(f"Update failed: {str(e)}")
    
    @mcp.tool()
    async def protocol_get(request: ProtocolGetRequest) -> str:
        """Retrieve a specific protocol."""
        try:
            protocol = get_op("protocol").get_protocol(request.number)
            return f"Protocol {protocol['number']}: {protocol['title']}\nStatus: {protocol['status']}\nVersion: {protocol['version']}\n\n{protocol['content']}"
        except Exception as e:
            raise ToolError(f"Retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def protocol_list(request: Optional[ProtocolListRequest] = None) -> str:
        """List protocols."""
        request = request or ProtocolListRequest()
        try:
            protocols = get_op("protocol").list_protocols(request.status)
            if not protocols: return "No protocols found."
            output = [f"Found {len(protocols)} protocol(s):"]
            for p in protocols:
                output.append(f"- {p['number']:03d}: {p['title']} [{p['status']}] v{p['version']}")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"List failed: {str(e)}")
    
    @mcp.tool()
    async def protocol_search(request: ProtocolSearchRequest) -> str:
        """Search protocols by content."""
        try:
            results = get_op("protocol").search_protocols(request.query)
            if not results: return f"No protocols found matching '{request.query}'"
            output = [f"Found {len(results)} protocol(s):"]
            for r in results:
                output.append(f"- {r['number']:03d}: {r['title']}")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Search failed: {str(e)}")
    
    # Task Tools
    @mcp.tool()
    async def create_task(request: TaskCreateRequest) -> str:
        """Create a new task file in TASKS/ directory."""
        try:
            result = get_op("task").create_task(title=request.title, objective=request.objective, deliverables=request.deliverables, acceptance_criteria=request.acceptance_criteria, priority=TaskPriority(request.priority), status=TaskStatus(request.status), lead=request.lead, dependencies=request.dependencies, related_documents=request.related_documents, notes=request.notes, task_number=request.task_number)
            return f"Created Task {result.task_number:03d}: {result.file_path}"
        except Exception as e:
            raise ToolError(f"Creation failed: {str(e)}")
    
    @mcp.tool()
    async def update_task(request: TaskUpdateRequest) -> str:
        """Update an existing task's metadata or content."""
        try:
            result = get_op("task").update_task(request.task_number, request.updates)
            return f"Updated Task {result.task_number:03d}. Fields: {', '.join(request.updates.keys())}"
        except Exception as e:
            raise ToolError(f"Update failed: {str(e)}")
    
    @mcp.tool()
    async def update_task_status(request: TaskUpdateStatusRequest) -> str:
        """Change task status (moves file between directories)."""
        try:
            result = get_op("task").update_task_status(request.task_number, TaskStatus(request.new_status), request.notes)
            return f"Updated Task {result.task_number:03d} status to {request.new_status}"
        except Exception as e:
            raise ToolError(f"Status update failed: {str(e)}")
    
    @mcp.tool()
    async def get_task(request: TaskGetRequest) -> str:
        """Retrieve a specific task by number."""
        try:
            task = get_op("task").get_task(request.task_number)
            if not task: return f"Task #{request.task_number:03d} not found"
            return f"Task {task.get('number', request.task_number):03d}: {task.get('title', 'Untitled')}\nStatus: {task.get('status')}\nPriority: {task.get('priority')}\n\nObjective:\n{task.get('objective', '')}"
        except Exception as e:
            raise ToolError(f"Retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def list_tasks(request: Optional[TaskListRequest] = None) -> str:
        """List tasks with optional filters."""
        request = request or TaskListRequest()
        try:
            status_filter = TaskStatus(request.status) if request.status else None
            priority_filter = TaskPriority(request.priority) if request.priority else None
            tasks = get_op("task").list_tasks(status_filter, priority_filter)
            if not tasks: return "No tasks found."
            output = [f"Found {len(tasks)} task(s):"]
            for t in tasks:
                output.append(f"- {t['number']:03d}: {t['title']} [{t['status']}] ({t['priority']})")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"List failed: {str(e)}")
    
    @mcp.tool()
    async def search_tasks(request: TaskSearchRequest) -> str:
        """Search tasks by content (full-text search)."""
        try:
            results = get_op("task").search_tasks(request.query)
            if not results: return f"No tasks found matching '{request.query}'"
            output = [f"Found {len(results)} task(s):"]
            for t in results:
                output.append(f"- {t['number']:03d}: {t['title']} [{t['status']}]")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Search failed: {str(e)}")
    
    # ADR Tools
    @mcp.tool()
    async def adr_create(request: ADRCreateRequest) -> str:
        """Create a new ADR with automatic sequential numbering."""
        try:
            result = get_op("adr").create_adr(title=request.title, context=request.context, decision=request.decision, consequences=request.consequences, date=request.date, status=request.status, author=request.author, supersedes=request.supersedes)
            return f"Created ADR {result['adr_number']:03d}: {result['file_path']}"
        except Exception as e:
            raise ToolError(f"Creation failed: {str(e)}")
    
    @mcp.tool()
    async def adr_update_status(request: ADRUpdateStatusRequest) -> str:
        """Update the status of an existing ADR."""
        try:
            result = get_op("adr").update_adr_status(request.number, request.new_status, request.reason)
            return f"Updated ADR {result['adr_number']:03d}: {result['old_status']} → {result['new_status']}"
        except Exception as e:
            raise ToolError(f"Status update failed: {str(e)}")
    
    @mcp.tool()
    async def adr_get(request: ADRGetRequest) -> str:
        """Retrieve a specific ADR by number."""
        try:
            adr = get_op("adr").get_adr(request.number)
            return f"ADR {adr['number']:03d}: {adr['title']}\nStatus: {adr['status']}\nDate: {adr['date']}\n\nContext:\n{adr['context']}\n\nDecision:\n{adr['decision']}"
        except Exception as e:
            raise ToolError(f"Retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def adr_list(request: Optional[ADRListRequest] = None) -> str:
        """List all ADRs with optional status filter."""
        request = request or ADRListRequest()
        try:
            adrs = get_op("adr").list_adrs(request.status)
            if not adrs: return "No ADRs found"
            result = f"Found {len(adrs)} ADR(s):\n\n"
            for adr in adrs:
                result += f"ADR {adr['number']:03d}: {adr['title']} [{adr['status']}]\n"
            return result
        except Exception as e:
            raise ToolError(f"List failed: {str(e)}")
    
    @mcp.tool()
    async def adr_search(request: ADRSearchRequest) -> str:
        """Full-text search across all ADRs."""
        try:
            results = get_op("adr").search_adrs(request.query)
            if not results: return f"No ADRs found matching '{request.query}'"
            output = f"Found {len(results)} ADR(s):\n\n"
            for result in results:
                output += f"ADR {result['number']:03d}: {result['title']}\n"
            return output
        except Exception as e:
            raise ToolError(f"Search failed: {str(e)}")
    
    # Persona Tools
    @mcp.tool()
    async def persona_dispatch(request: PersonaDispatchParams) -> dict:
        """Dispatch a task to a specific persona agent."""
        try:
            return get_op("persona").dispatch(role=request.role, task=request.task, context=request.context, maintain_state=request.maintain_state, engine=request.engine, model_name=request.model_name, custom_persona_file=request.custom_persona_file)
        except Exception as e:
            raise ToolError(f"Dispatch failed: {str(e)}")
    
    @mcp.tool()
    async def persona_list_roles() -> dict:
        """List all available persona roles."""
        try:
            return get_op("persona").list_roles()
        except Exception as e:
            raise ToolError(f"Role list failed: {str(e)}")
    
    @mcp.tool()
    async def persona_get_state(request: PersonaRoleParams) -> dict:
        """Get conversation state for a specific persona role."""
        try:
            return get_op("persona").get_state(role=request.role)
        except Exception as e:
            raise ToolError(f"State retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def persona_reset_state(request: PersonaRoleParams) -> dict:
        """Reset conversation state for a specific persona role."""
        try:
            return get_op("persona").reset_state(role=request.role)
        except Exception as e:
            raise ToolError(f"State reset failed: {str(e)}")
    
    @mcp.tool()
    async def persona_create_custom(request: PersonaCreateCustomParams) -> dict:
        """Create a new custom persona."""
        try:
            return get_op("persona").create_custom(role=request.role, persona_definition=request.persona_definition, description=request.description)
        except Exception as e:
            raise ToolError(f"Persona creation failed: {str(e)}")
    
    # Config Tools
    @mcp.tool()
    async def config_list() -> str:
        """List all configuration files in the .agent/config directory."""
        try:
            configs = get_op("config").list_configs()
            if not configs: return "No configuration files found."
            output = [f"Found {len(configs)} configuration files:"]
            for c in configs:
                output.append(f"- {c['name']} ({c['size']} bytes)")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Config list failed: {str(e)}")
    
    @mcp.tool()
    async def config_read(request: ConfigReadRequest) -> str:
        """Read a configuration file."""
        try:
            content = get_op("config").read_config(request.filename)
            if isinstance(content, (dict, list)):
                return json.dumps(content, indent=2)
            return str(content)
        except Exception as e:
            raise ToolError(f"Config read failed: {str(e)}")
    
    @mcp.tool()
    async def config_write(request: ConfigWriteRequest) -> str:
        """Write a configuration file."""
        try:
            path = get_op("config").write_config(request.filename, request.content)
            return f"Successfully wrote config to {path}"
        except Exception as e:
            raise ToolError(f"Config write failed: {str(e)}")
    
    @mcp.tool()
    async def config_delete(request: ConfigDeleteRequest) -> str:
        """Delete a configuration file."""
        try:
            get_op("config").delete_config(request.filename)
            return f"Successfully deleted config '{request.filename}'"
        except Exception as e:
            raise ToolError(f"Config delete failed: {str(e)}")
    
    # Workflow Tools
    @mcp.tool()
    async def get_available_workflows() -> str:
        """List all available workflows in the .agent/workflows directory."""
        try:
            workflows = get_op("workflow").list_workflows()
            if not workflows: return "No workflows found."
            output = [f"Found {len(workflows)} available workflow(s):"]
            for wf in workflows:
                turbo = " [TURBO]" if wf.get('turbo_mode') else ""
                output.append(f"- {wf['filename']}{turbo}: {wf['description']}")
            return "\n".join(output)
        except Exception as e:
            raise ToolError(f"Workflow list failed: {str(e)}")
    
    @mcp.tool()
    async def read_workflow(request: WorkflowReadRequest) -> str:
        """Read the content of a specific workflow file."""
        try:
            content = get_op("workflow").get_workflow_content(request.filename)
            if content is None:
                return f"Workflow '{request.filename}' not found."
            return content
        except Exception as e:
            raise ToolError(f"Workflow read failed: {str(e)}")
    
    logger.info("Starting FastMCP server (STDIO Mode) with 37 tools")
    mcp.run(transport="stdio")


#============================================
# Main Execution Entry Point (ADR-066 v1.3 Canonical Selector)
#============================================
def run_server():
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if MCP_TRANSPORT not in {"stdio", "sse"}:
        logger.error(f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}. Must be 'stdio' or 'sse'.")
        sys.exit(1)
    
    if MCP_TRANSPORT == "sse":
        port = int(os.getenv("PORT", 8105))
        run_sse_server(port)
    else:
        run_stdio_server()


if __name__ == "__main__":
    run_server()
