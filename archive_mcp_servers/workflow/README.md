# Workflow MCP Server

## Overview
The **Workflow MCP Server** provides access to the project's standard operating procedures and workflows defined in `.agent/workflows`. It allows agents to discover, read, and execute defined workflows.

## Capabilities

### 1. List Workflows
- **Tool**: `get_available_workflows`
- **Purpose**: Lists all valid workflow files (`.md`) found in the configured workflow directory.
- **Attributes**: Returns filename, description (from frontmatter), and turbo mode status.

### 2. Read Workflow
- **Tool**: `read_workflow`
- **Purpose**: Retrieves the full content of a specific workflow file.
- **Usage**: Used by agents to load instructions for a specific procedure.

## Architecture
- **Layer**: Business Logic (Core)
- **Class**: `WorkflowOperations` (in `operations.py`)
- **Integration**: Exposed via `server.py` (Standalone) or `sanctuary_domain` (Gateway).

## Configuration
- **Directories**: Looks for workflows in `PROJECT_ROOT/.agent/workflows`.

## Dependenices
- `mcp_servers.lib`
- `yaml` (PyYAML) - for parsing frontmatter.
