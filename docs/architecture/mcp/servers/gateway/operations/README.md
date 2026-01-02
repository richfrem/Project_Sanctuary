# MCP Gateway Operations Inventory

**Status:** ✅ Fully Operational  
**Last Updated:** 2024-12-24  
**Total Operations:** 86 tools across 6 clusters  
**Source:** [`fleet_registry.json`](../../../../../../mcp_servers/gateway/fleet_registry.json)

---

## 🚀 Fleet Deployment Summary

| # | Container | Port | Tools | Category | Status |
|---|-----------|------|-------|----------|--------|
| 1 | `sanctuary_utils` | 8100 | 17 | Time, Calc, UUID, String | ✅ Live |
| 2 | `sanctuary_filesystem` | 8101 | 10 | File I/O, Code Analysis | ✅ Live |
| 3 | `sanctuary_network` | 8102 | 2 | HTTP Fetch, Site Status | ✅ Live |
| 4 | `sanctuary_git` | 8103 | 9 | Protocol 101 Git Ops | ✅ Live |
| 5 | `sanctuary_cortex` | 8104 | 13 | RAG, Forge LLM, Cache | ✅ Live |
| 6 | `sanctuary_domain` | 8105 | 35 | Chronicle, ADR, Task, Protocol | ✅ Live |

**Backend Services:**
- `sanctuary_vector_db` (8110) - ChromaDB Vector Store
- `sanctuary_ollama` (11434) - LLM Inference (Ollama)

---

## Complete Operations Reference (86 Tools)

### sanctuary_cortex (13 tools) - Port 8104

| Tool Name | Description |
|-----------|-------------|
| `cortex-check-sanctuary-model-status` | Check Sanctuary model availability and status |
| `cortex-query-sanctuary-model` | Query the fine-tuned Sanctuary model |
| `cortex-cortex-capture-snapshot` | Tool-driven snapshot generation (Protocol 128 v3.5) |
| `cortex-cortex-learning-debrief` | Scans repository for technical state changes (Protocol 128) |
| `cortex-cortex-cache-stats` | Get Mnemonic Cache (CAG) statistics |
| `cortex-cortex-guardian-wakeup` | Generate Guardian boot digest (Protocol 114) |
| `cortex-cortex-cache-warmup` | Pre-populate cache with genesis queries |
| `cortex-cortex-cache-set` | Store answer in cache |
| `cortex-cortex-cache-get` | Retrieve cached answer for a query |
| `cortex-cortex-ingest-incremental` | Incrementally ingest documents into the knowledge base |
| `cortex-cortex-get-stats` | Get database statistics and health status |
| `cortex-cortex-query` | Perform semantic search query against the knowledge base |
| `cortex-cortex-ingest-full` | Perform full re-ingestion of the knowledge base |

---

### sanctuary_domain (35 tools) - Port 8105

#### Chronicle (8)
| Tool Name | Description |
|-----------|-------------|
| `chronicle-list-entries` | List recent chronicle entries |
| `chronicle-search` | Search chronicle entries by content |
| `chronicle-read-latest-entries` | Read the latest entries from the Chronicle |
| `chronicle-get-entry` | Retrieve a specific chronicle entry |
| `chronicle-update-entry` | Update an existing chronicle entry |
| `chronicle-append-entry` | Append a new entry to the Chronicle |
| `chronicle-create-entry` | Create a new chronicle entry |

#### ADR (5)
| Tool Name | Description |
|-----------|-------------|
| `adr-search` | Full-text search across all ADRs |
| `adr-list` | List all ADRs with optional status filter |
| `adr-get` | Retrieve a specific ADR by number |
| `adr-update-status` | Update the status of an existing ADR |
| `adr-create` | Create a new ADR with automatic sequential numbering |

#### Protocol (5)
| Tool Name | Description |
|-----------|-------------|
| `protocol-search` | Search protocols by content |
| `protocol-list` | List protocols |
| `protocol-get` | Retrieve a specific protocol |
| `protocol-update` | Update an existing protocol |
| `protocol-create` | Create a new protocol |

#### Task (6)
| Tool Name | Description |
|-----------|-------------|
| `search-tasks` | Search tasks by content (full-text search) |
| `list-tasks` | List tasks with optional filters |
| `get-task` | Retrieve a specific task by number |
| `update-task-status` | Change task status (moves file between directories) |
| `update-task` | Update an existing task's metadata or content |
| `create-task` | Create a new task file in tasks/ directory |

#### Persona (5)
| Tool Name | Description |
|-----------|-------------|
| `persona-create-custom` | Create a new custom persona |
| `persona-reset-state` | Reset conversation state for a specific persona role |
| `persona-get-state` | Get conversation state for a specific persona role |
| `persona-list-roles` | List all available persona roles |
| `persona-dispatch` | Dispatch a task to a specific persona agent |

#### Config (4)
| Tool Name | Description |
|-----------|-------------|
| `config-delete` | Delete a configuration file |
| `config-write` | Write a configuration file |
| `config-read` | Read a configuration file |
| `config-list` | List all configuration files in .agent/config directory |

#### Workflow (2)
| Tool Name | Description |
|-----------|-------------|
| `read-workflow` | Read the content of a specific workflow file |
| `get-available-workflows` | List all available workflows in .agent/workflows directory |

---

### sanctuary_filesystem (10 tools) - Port 8101

| Tool Name | Description |
|-----------|-------------|
| `code-get-info` | Get file metadata |
| `code-write` | Write/update file with automatic backup |
| `code-read` | Read file contents |
| `code-search-content` | Search for text/patterns in code files |
| `code-list-files` | List files in a directory with optional pattern |
| `code-find-file` | Find files by name or glob pattern |
| `code-check-tools` | Check which code quality tools are available |
| `code-analyze` | Perform static analysis on code |
| `code-format` | Format code in a file or directory |
| `code-lint` | Run linting on a file or directory |

---

### sanctuary_git (9 tools) - Port 8103

| Tool Name | Description |
|-----------|-------------|
| `git-log` | Show commit history |
| `git-diff` | Show changes (diff) |
| `git-finish-feature` | Finish feature (cleanup/delete) |
| `git-start-feature` | Start a new feature branch |
| `git-push-feature` | Push feature branch to origin |
| `git-add` | Stage files for commit |
| `git-get-status` | Get standard git status |
| `git-get-safety-rules` | Return Protocol 101 safety rules |
| `git-smart-commit` | Commit with automated Protocol 101 checks |

---

### sanctuary_network (2 tools) - Port 8102

| Tool Name | Description |
|-----------|-------------|
| `check-site-status` | Check if a site is up (HEAD request) |
| `fetch-url` | Fetch content from a URL via HTTP GET |

---

### sanctuary_utils (17 tools) - Port 8100

#### Time (2)
| Tool Name | Description |
|-----------|-------------|
| `time-get-timezone-info` | Get information about available timezones |
| `time-get-current-time` | Get the current time in UTC or specified timezone |

#### Calculator (6)
| Tool Name | Description |
|-----------|-------------|
| `calculator-calculate` | Evaluate a mathematical expression safely |
| `calculator-add` | Add two numbers |
| `calculator-subtract` | Subtract b from a |
| `calculator-multiply` | Multiply two numbers |
| `calculator-divide` | Divide a by b |

#### UUID (3)
| Tool Name | Description |
|-----------|-------------|
| `uuid-generate-uuid4` | Generate a random UUID (version 4) |
| `uuid-generate-uuid1` | Generate a UUID based on host ID and current time (version 1) |
| `uuid-validate-uuid` | Validate if a string is a valid UUID |

#### String (6)
| Tool Name | Description |
|-----------|-------------|
| `string-to-upper` | Convert text to uppercase |
| `string-to-lower` | Convert text to lowercase |
| `string-trim` | Remove leading and trailing whitespace |
| `string-reverse` | Reverse a string |
| `string-word-count` | Count words in text |
| `string-replace` | Replace occurrences of old with new in text |

---

## Gateway Tool Naming Convention

All tools are prefixed with `sanctuary-<cluster>-` when accessed via the Gateway:
```
sanctuary-cortex-cortex-query
sanctuary-domain-adr-create
sanctuary-git-git-status
```

---

## Quick Reference Commands

```bash
# Health Checks
curl http://localhost:8100/health  # Utils
curl http://localhost:8104/health  # Cortex

# SSE Handshake Verification
curl -N http://localhost:8104/sse  # Should return: event: endpoint

# Gateway API
curl -k https://localhost:4444/tools  # List all tools

# Fleet Management
podman compose up -d      # Start fleet
podman compose down       # Stop fleet
podman ps --format "table {{.Names}}\t{{.Status}}"  # Status
```

---

*For verification status, see [GATEWAY_VERIFICATION_MATRIX.md](./GATEWAY_VERIFICATION_MATRIX.md)*
