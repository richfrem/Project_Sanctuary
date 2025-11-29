# Project Sanctuary MCP Servers

This directory contains the "Core Quad" of Model Context Protocol (MCP) servers that power the Project Sanctuary nervous system.

## Core Quad Servers

1. **Cortex (`project_sanctuary.cognitive.cortex`)**
   - **Purpose:** Memory, RAG, and Knowledge Retrieval.
   - **Tools:** `cortex_query`, `cortex_ingest_full`, `cortex_ingest_incremental`, `cortex_get_stats`.
   - **Location:** `mcp_servers/cognitive/cortex/server.py`

2. **Chronicle (`project_sanctuary.chronicle`)**
   - **Purpose:** History, Logging, and Sequential Records.
   - **Tools:** `chronicle_create_entry`, `chronicle_read_latest_entries`, `chronicle_append_entry`, `chronicle_search`.
   - **Location:** `mcp_servers/chronicle/server.py`

3. **Protocol (`project_sanctuary.protocol`)**
   - **Purpose:** Law, Validation, and Governance.
   - **Tools:** `protocol_get`, `protocol_list`, `protocol_validate_action`, `protocol_search`.
   - **Location:** `mcp_servers/protocol/server.py`

4. **Orchestrator (`project_sanctuary.orchestrator`)**
   - **Purpose:** High-level Planning and Council Logic.
   - **Tools:** `orchestrator_consult_strategist`, `orchestrator_consult_auditor`, `orchestrator_dispatch_mission`.
   - **Location:** `mcp_servers/orchestrator/server.py`

## Configuration

To use these servers with an MCP client (like Claude Desktop), add the following to your configuration file (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "python",
      "args": ["/absolute/path/to/Project_Sanctuary/mcp_servers/cognitive/cortex/server.py"],
      "env": {
        "PROJECT_ROOT": "/absolute/path/to/Project_Sanctuary"
      }
    },
    "chronicle": {
      "command": "python",
      "args": ["/absolute/path/to/Project_Sanctuary/mcp_servers/chronicle/server.py"],
      "env": {
        "PROJECT_ROOT": "/absolute/path/to/Project_Sanctuary"
      }
    },
    "protocol": {
      "command": "python",
      "args": ["/absolute/path/to/Project_Sanctuary/mcp_servers/protocol/server.py"],
      "env": {
        "PROJECT_ROOT": "/absolute/path/to/Project_Sanctuary"
      }
    },
    "orchestrator": {
      "command": "python",
      "args": ["/absolute/path/to/Project_Sanctuary/mcp_servers/orchestrator/server.py"],
      "env": {
        "PROJECT_ROOT": "/absolute/path/to/Project_Sanctuary"
      }
    }
  }
}
```

## Running Manually

You can use the helper script to verify paths:
```bash
./start_mcp_servers.sh
```
