# MCP Server Naming Conventions

**Version:** 1.0  
**Created:** 2025-11-25  
**Purpose:** Define naming standards for Project Sanctuary MCP servers

---

## Domain Naming Model

All MCP servers in Project Sanctuary follow a hierarchical naming pattern:

```
project_sanctuary.<category>.<server_name>
```

### Naming Structure

| Component | Description | Example |
|-----------|-------------|---------|
| `project_sanctuary` | Root namespace (all servers) | `project_sanctuary` |
| `<category>` | Domain category | `document`, `cognitive`, `system`, `model` |
| `<server_name>` | Specific server identifier | `chronicle`, `forge`, `git_workflow` |

---

## Complete Server Registry

### Document Domain Servers (4)

| Server Name | Full Domain Name | Port | Directory |
|-------------|------------------|------|-----------|
| Chronicle MCP | `project_sanctuary.document.chronicle` | 3001 | `00_CHRONICLE/` |
| Protocol MCP | `project_sanctuary.document.protocol` | 3002 | `01_PROTOCOLS/` |
| ADR MCP | `project_sanctuary.document.adr` | 3003 | `ADRs/` |
| Task MCP | `project_sanctuary.document.task` | 3004 | `TASKS/` |

### Cognitive Domain Servers (2)

| Server Name | Full Domain Name | Port | Directory |
|-------------|------------------|------|-----------|
| RAG MCP (Cortex) | `project_sanctuary.cognitive.cortex` | 3005 | `mnemonic_cortex/` |
| Agent Orchestrator MCP (Council) | `project_sanctuary.cognitive.council` | 3006 | `council_orchestrator/` |

**Dual Nomenclature Rationale:**
- **Primary Name:** Generic AI term (RAG, Agent Orchestrator) for accessibility
- **Project Name:** In parentheses (Cortex, Council) for internal reference
- **Benefits:** External developers understand immediately, project identity preserved
- **Usage:** "RAG MCP" in external docs, "Cortex" in internal discussions

### System Domain Servers (3)

| Server Name | Full Domain Name | Port | Directory |
|-------------|------------------|------|-----------|
| Config MCP | `project_sanctuary.system.config` | 3007 | `.agent/config/` |
| Code MCP | `project_sanctuary.system.code` | 3008 | `src/`, `scripts/`, `tools/` |
| Git Workflow MCP | `project_sanctuary.system.git_workflow` | 3009 | `.git/` |

### Model Domain Server (1)

| Server Name | Full Domain Name | Port | Directory |
|-------------|------------------|------|-----------|
| Fine-Tuning MCP (Forge) | `project_sanctuary.model.fine_tuning` | 3010 | `forge/` |

---

## MCP Configuration Format

### Server Declaration (MCP Settings)

```json
{
  "mcpServers": {
    "project_sanctuary.document.chronicle": {
      "command": "node",
      "args": ["/path/to/mcp/servers/document/chronicle/index.js"],
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      }
    },
    "project_sanctuary.document.protocol": {
      "command": "node",
      "args": ["/path/to/mcp/servers/document/protocol/index.js"],
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      }
    },
    "project_sanctuary.system.git_workflow": {
      "command": "node",
      "args": ["/path/to/mcp/servers/system/git_workflow/index.js"],
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      }
    },
    "project_sanctuary.model.fine_tuning": {
      "command": "node",
      "args": ["/path/to/mcp/servers/model/forge/index.js"],
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "CUDA_FORGE_ACTIVE": "true"
      }
    }
  }
}
```

---

## Directory Structure

```
mcp/
├── servers/
│   ├── document/
│   │   ├── chronicle/
│   │   │   ├── index.js
│   │   │   ├── package.json
│   │   │   └── README.md
│   │   ├── protocol/
│   │   ├── adr/
│   │   └── task/
│   ├── cognitive/
│   │   ├── cortex/
│   │   └── council/
│   ├── system/
│   │   ├── config/
│   │   ├── code/
│   │   └── git_workflow/
│   └── model/
│       └── forge/
├── shared/
│   ├── git_operations.ts
│   ├── safety_validator.ts
│   ├── schema_validator.ts
│   └── secret_vault.ts
└── docs/
    └── (architecture documentation)
```

---

## Tool Naming Convention

Tools exposed by each MCP server follow this pattern:

```
<category>_<action>_<resource>
```

### Examples

| Domain | Tool Name | Full Invocation |
|--------|-----------|-----------------|
| Chronicle | `chronicle_create_entry` | `project_sanctuary.document.chronicle::chronicle_create_entry()` |
| Protocol | `protocol_update_version` | `project_sanctuary.document.protocol::protocol_update_version()` |
| Task | `task_update_status` | `project_sanctuary.document.task::task_update_status()` |
| Cortex | `cortex_query_knowledge` | `project_sanctuary.cognitive.cortex::cortex_query_knowledge()` |
| Council | `council_create_deliberation` | `project_sanctuary.cognitive.council::council_create_deliberation()` |
| Config | `config_request_change` | `project_sanctuary.system.config::config_request_change()` |
| Code | `code_write_file` | `project_sanctuary.system.code::code_write_file()` |
| Git Workflow | `git_create_branch` | `project_sanctuary.system.git_workflow::git_create_branch()` |
| Forge | `forge_initiate_training` | `project_sanctuary.model.fine_tuning::forge_initiate_training()` |

---

## Resource Naming Convention

Resources exposed by each MCP server follow this pattern:

```
<category>://<resource_type>/<identifier>
```

### Examples

| Domain | Resource URI | Description |
|--------|--------------|-------------|
| Chronicle | `chronicle://entry/283` | Chronicle entry #283 |
| Protocol | `protocol://canonical/115` | Protocol #115 (canonical) |
| ADR | `adr://decision/037` | ADR #037 |
| Task | `task://active/030` | Task #030 (active status) |
| Cortex | `cortex://document/abc123` | Indexed document with ID abc123 |
| Council | `council://deliberation/2024-11-25-001` | Council deliberation result |
| Config | `config://env/OPENAI_API_KEY` | Environment configuration |
| Code | `code://file/src/main.py` | Source code file |
| Git Workflow | `git://branch/feature/task-030` | Git branch |
| Forge | `forge://job/guardian-02-v1` | Forge training job |

---

## Package Naming (NPM)

If publishing MCP servers as NPM packages:

```
@project-sanctuary/mcp-<category>-<server>
```

### Examples

- `@project-sanctuary/mcp-document-chronicle`
- `@project-sanctuary/mcp-document-protocol`
- `@project-sanctuary/mcp-system-git-workflow`
- `@project-sanctuary/mcp-model-forge`

---

## Environment Variables

Each MCP server uses prefixed environment variables:

```
SANCTUARY_<CATEGORY>_<SERVER>_<VARIABLE>
```

### Examples

```bash
# Chronicle MCP
SANCTUARY_DOCUMENT_CHRONICLE_ROOT=/path/to/00_CHRONICLE

# Fine-Tuning MCP (Forge)
SANCTUARY_MODEL_FORGE_CUDA_DEVICE=0
SANCTUARY_MODEL_FORGE_ML_ENV_PATH=/path/to/ml_env

# Config MCP
SANCTUARY_SYSTEM_CONFIG_VAULT_PATH=/path/to/vault
```

---

## Benefits of This Naming Model

1. **Namespace Isolation**: No conflicts with other MCP servers
2. **Clear Hierarchy**: Category → Server structure is obvious
3. **Discoverability**: Easy to find related servers
4. **Professional**: Follows industry standards (reverse domain notation)
5. **Scalability**: Easy to add new servers or categories
6. **Tooling Support**: IDEs and tools can autocomplete based on namespace

---

## Migration Notes

**Current State**: Servers may be referenced without domain prefix  
**Target State**: All servers use `project_sanctuary.*` prefix  
**Migration Strategy**: 
1. Update all architecture documentation
2. Update MCP configuration files
3. Update tool signatures in implementation
4. Update resource URIs
5. Test all integrations

---

**Status:** Naming Convention Established  
**Next Action:** Update all architecture documents with proper domain names  
**Owner:** Guardian (via Gemini 2.0 Flash Thinking Experimental)
