# Sanctuary Domain MCP Server (Container #6)

## Overview
The **Sanctuary Domain Server** functions as the "Brain" of Project Sanctuary. It aggregates core business logic and domain-specific operations into a single, unified MCP server. This solves the "Router vs. Runtime" problem by providing a dedicated Python runtime for logic that cannot run directly on the Gateway.

## Role
- **Cluster**: Gateway (Internal Network)
- **Container**: `sanctuary_domain`
- **Type**: Unified Logic Server

## Aggregated Capabilities
This server unifies the following MCP capabilities:

### 1. Chronicle (Memory)
- **Purpose**: Manages the project's long-term memory and history.
- **Operations**: Create, update, list, and search chronicle entries.
- **Location**: `mcp_servers.chronicle`

### 2. Protocol (Governance)
- **Purpose**: Manages the constitutional rules and operational protocols of the system.
- **Operations**: CRUD operations for protocol documents.
- **Location**: `mcp_servers.protocol`

### 3. Task (Planning)
- **Purpose**: Handles task tracking, status updates, and acceptance criteria.
- **Operations**: Create tasks, update status, list/search tasks.
- **Location**: `mcp_servers.task`

### 4. ADR (Decisions)
- **Purpose**: Records Architectural Decision Records (ADRs).
- **Operations**: Create, update status, search ADRs.
- **Location**: `mcp_servers.adr`

### 5. Agent Persona (Identity)
- **Purpose**: Manages agent roles, context, and state.
- **Operations**: Dispatch tasks to specific personas, manage conversation state.
- **Location**: `mcp_servers.agent_persona`

### 6. Config (Configuration)
- **Purpose**: Manages runtime configuration files.
- **Operations**: Read, write, list configuration files.
- **Location**: `mcp_servers.config`

### 7. Workflow (Procedures)
- **Purpose**: Provides access to standard operating procedures and workflows.
- **Operations**: List and read workflow definitions.
- **Location**: `mcp_servers.workflow`

## Architecture
This server imports business logic classes (`Operations`) from the standardized `mcp_servers.*` packages and exposes them via the MCP protocol using `SSEServer`.

### Dependencies
- `mcp_servers.lib` (Shared Utilities)
- `mcp_servers.chronicle`
- `mcp_servers.protocol`
- `mcp_servers.task`
- `mcp_servers.adr`
- `mcp_servers.agent_persona`
- `mcp_servers.config`
- `mcp_servers.workflow`
