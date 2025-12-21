# Complete Tools Catalog: All 63 MCP Tools

**Document Version:** 1.0  
**Last Updated:** 2025-12-15  
**Purpose:** Comprehensive catalog of all tools accessible through the Gateway

---

## Overview

The Sanctuary Gateway provides access to **63 tools** across **12 backend MCP servers**.

**Tool Categories:**
- **Knowledge & Memory:** 9 tools (RAG Cortex)
- **Version Control:** 9 tools (Git Workflow)
- **Code Operations:** 9 tools (Code MCP)
- **Task Management:** 6 tools (Task MCP)
- **Documentation:** 16 tools (Chronicle, ADR, Protocol)
- **AI & Agents:** 9 tools (Council, Persona, Forge)
- **System:** 6 tools (Config, Orchestrator)

---

## 1. RAG Cortex MCP (9 tools)

**Server:** `rag_cortex`  
**Purpose:** Knowledge base operations (RAG)  
**Category:** Knowledge & Memory

### 1.1 Query Operations

#### `cortex_query(query: str, max_results: int = 5, use_cache: bool = False, reasoning_mode: bool = False)`
**Purpose:** Perform semantic search query against the knowledge base  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

**Example:**
```python
cortex_query("What is Protocol 101?", max_results=3)
```

#### `cortex_get_stats()`
**Purpose:** Get database statistics and health status  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

### 1.2 Ingestion Operations

#### `cortex_ingest_full(purge_existing: bool = True, source_directories: list = None)`
**Purpose:** Perform full re-ingestion of the knowledge base  
**Read-Only:** No  
**Approval Required:** Yes (destructive)  
**Rate Limit:** 1/hour

#### `cortex_ingest_incremental(file_paths: list, metadata: dict = None, skip_duplicates: bool = True)`
**Purpose:** Incrementally ingest documents without rebuilding database  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

### 1.3 Cache Operations

#### `cortex_cache_get(query: str)`
**Purpose:** Retrieve cached answer for a query  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `cortex_cache_set(query: str, answer: str)`
**Purpose:** Store answer in cache for future retrieval  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `cortex_cache_warmup(genesis_queries: list = None)`
**Purpose:** Pre-populate cache with genesis queries  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

#### `cortex_cache_stats()`
**Purpose:** Get Mnemonic Cache (CAG) statistics  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

### 1.4 Guardian Operations

#### `cortex_guardian_wakeup(mode: str = "HOLISTIC")`
**Purpose:** Generate Guardian boot digest from cached bundles (Protocol 114)  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

---

## 2. Git Workflow MCP (9 tools)

**Server:** `git_workflow`  
**Purpose:** Git operations with Protocol 101 enforcement  
**Category:** Version Control

### 2.1 Safety & Status

#### `git_get_safety_rules()`
**Purpose:** Get the unbreakable Git safety rules  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `git_get_status()`
**Purpose:** Get the current status of the repository  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

### 2.2 Branch Operations

#### `git_start_feature(task_id: str, description: str)`
**Purpose:** Start a new feature branch (idempotent)  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

**Format:** `feature/task-{task_id}-{description}`

#### `git_finish_feature(branch_name: str, force: bool = False)`
**Purpose:** Finish a feature branch (cleanup)  
**Read-Only:** No  
**Approval Required:** Yes  
**Rate Limit:** 10/min

### 2.3 Commit Operations

#### `git_add(files: list = None)`
**Purpose:** Stage files for commit  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `git_smart_commit(message: str)`
**Purpose:** Commit staged files with automatic Protocol 101 v3.0 enforcement  
**Read-Only:** No  
**Approval Required:** Yes (destructive)  
**Rate Limit:** 10/min

#### `git_push_feature(force: bool = False, no_verify: bool = False)`
**Purpose:** Push the current feature branch to origin  
**Read-Only:** No  
**Approval Required:** Yes  
**Rate Limit:** 10/min

### 2.4 Inspection Operations

#### `git_diff(cached: bool = False, file_path: str = None)`
**Purpose:** Show changes in the working directory or staged files  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `git_log(max_count: int = 10, oneline: bool = False)`
**Purpose:** Show commit history  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 3. Task MCP (6 tools)

**Server:** `task`  
**Purpose:** Task management operations  
**Category:** Task Management

### 3.1 Create & Update

#### `create_task(title: str, objective: str, deliverables: list, acceptance_criteria: list, ...)`
**Purpose:** Create a new task file in TASKS/ directory  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `update_task(task_number: int, updates: dict)`
**Purpose:** Update an existing task's metadata or content  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `update_task_status(task_number: int, new_status: str, notes: str = None)`
**Purpose:** Change task status (moves file between directories)  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

**Valid Statuses:** `backlog`, `todo`, `in-progress`, `complete`, `blocked`

### 3.2 Query Operations

#### `get_task(task_number: int)`
**Purpose:** Retrieve a specific task by number  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `list_tasks(status: str = None, priority: str = None)`
**Purpose:** List tasks with optional filters  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `search_tasks(query: str)`
**Purpose:** Search tasks by content (full-text search)  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 4. Chronicle MCP (6 tools)

**Server:** `chronicle`  
**Purpose:** Historical truth and canonical records  
**Category:** Documentation

### 4.1 Create & Update

#### `chronicle_create_entry(title: str, content: str, author: str, date: str = None, status: str = "draft", classification: str = "internal")`
**Purpose:** Create a new chronicle entry  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `chronicle_update_entry(entry_number: int, updates: dict, reason: str, override_approval_id: str = None)`
**Purpose:** Update an existing chronicle entry  
**Read-Only:** No  
**Approval Required:** Yes (if >7 days old)  
**Rate Limit:** 50/min

### 4.2 Query Operations

#### `chronicle_get_entry(entry_number: int)`
**Purpose:** Retrieve a specific chronicle entry  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `chronicle_list_entries(limit: int = 10)`
**Purpose:** List recent chronicle entries  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `chronicle_read_latest_entries(limit: int = 10)`
**Purpose:** Read the latest entries from the Chronicle (alias for list_entries)  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `chronicle_search(query: str)`
**Purpose:** Search chronicle entries by content  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 5. ADR MCP (5 tools)

**Server:** `adr`  
**Purpose:** Architecture Decision Records  
**Category:** Documentation

### 5.1 Create & Update

#### `adr_create(title: str, context: str, decision: str, consequences: str, date: str = None, status: str = "proposed", author: str = "AI Assistant", supersedes: int = None)`
**Purpose:** Create a new ADR with automatic sequential numbering  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `adr_update_status(number: int, new_status: str, reason: str)`
**Purpose:** Update the status of an existing ADR  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

**Valid Transitions:**
- `proposed` → `accepted`
- `proposed` → `deprecated`
- `accepted` → `deprecated`
- `accepted` → `superseded`

### 5.2 Query Operations

#### `adr_get(number: int)`
**Purpose:** Retrieve a specific ADR by number  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `adr_list(status: str = None)`
**Purpose:** List all ADRs with optional status filter  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `adr_search(query: str)`
**Purpose:** Full-text search across all ADRs  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 6. Protocol MCP (5 tools)

**Server:** `protocol`  
**Purpose:** Protocol creation and management  
**Category:** Documentation

### 6.1 Create & Update

#### `protocol_create(number: int, title: str, status: str, classification: str, version: str, authority: str, content: str, linked_protocols: list = None)`
**Purpose:** Create a new protocol  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `protocol_update(number: int, updates: dict, reason: str)`
**Purpose:** Update an existing protocol  
**Read-Only:** No  
**Approval Required:** Yes (for canonical protocols)  
**Rate Limit:** 50/min

### 6.2 Query Operations

#### `protocol_get(number: int)`
**Purpose:** Retrieve a specific protocol  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `protocol_list(status: str = None)`
**Purpose:** List protocols  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `protocol_search(query: str)`
**Purpose:** Search protocols by content  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 7. Code MCP (9 tools)

**Server:** `code`  
**Purpose:** Code operations (read, write, analyze, lint, format)  
**Category:** Code Operations

### 7.1 File Operations

#### `code_read(path: str, max_size_mb: int = 10)`
**Purpose:** Read file contents  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `code_write(path: str, content: str, backup: bool = True, create_dirs: bool = True)`
**Purpose:** Write/update file with automatic backup  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `code_get_info(path: str)`
**Purpose:** Get file metadata  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

### 7.2 Search Operations

#### `code_find_file(name_pattern: str, directory: str = ".")`
**Purpose:** Find files by name or glob pattern  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `code_list_files(directory: str = ".", pattern: str = "*", recursive: bool = True)`
**Purpose:** List files in a directory with optional pattern  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `code_search_content(query: str, file_pattern: str = "*.py", case_sensitive: bool = False)`
**Purpose:** Search for text/patterns in code files  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

### 7.3 Quality Operations

#### `code_analyze(path: str)`
**Purpose:** Perform static analysis on code  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `code_lint(path: str, tool: str = "ruff")`
**Purpose:** Run linting on a file or directory  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `code_format(path: str, tool: str = "ruff", check_only: bool = False)`
**Purpose:** Format code in a file or directory  
**Read-Only:** No (unless check_only=True)  
**Approval Required:** No  
**Rate Limit:** 50/min

---

## 8. Council MCP (2 tools)

**Server:** `council`  
**Purpose:** Multi-agent deliberation  
**Category:** AI & Agents

#### `council_dispatch(task_description: str, agent: str = None, max_rounds: int = 3, force_engine: str = None, model_preference: str = None, output_path: str = None)`
**Purpose:** Dispatch a task to the Sanctuary Council for multi-agent deliberation  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

**Agents:** `coordinator`, `strategist`, `auditor`, or `None` (full council)

#### `council_list_agents()`
**Purpose:** List all available Council agents and their current status  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 9. Agent Persona MCP (5 tools)

**Server:** `agent_persona`  
**Purpose:** Persona-based agent dispatch  
**Category:** AI & Agents

#### `persona_dispatch(role: str, task: str, context: dict = None, maintain_state: bool = True, engine: str = None, model_name: str = None, custom_persona_file: str = None)`
**Purpose:** Dispatch a task to a specific persona agent  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

**Built-in Roles:** `coordinator`, `strategist`, `auditor`

#### `persona_create_custom(role: str, persona_definition: str, description: str)`
**Purpose:** Create a new custom persona  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

#### `persona_list_roles()`
**Purpose:** List all available persona roles (built-in and custom)  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `persona_get_state(role: str)`
**Purpose:** Get conversation state for a specific persona role  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `persona_reset_state(role: str)`
**Purpose:** Reset conversation state for a specific persona role  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

---

## 10. Forge LLM MCP (2 tools)

**Server:** `forge_llm`  
**Purpose:** Fine-tuned Sanctuary model queries  
**Category:** AI & Agents

#### `query_sanctuary_model(prompt: str, temperature: float = 0.7, max_tokens: int = 2048, system_prompt: str = None)`
**Purpose:** Query the fine-tuned Sanctuary model for specialized knowledge  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 50/min

#### `check_sanctuary_model_status()`
**Purpose:** Check if the Sanctuary model is available and ready to use  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

---

## 11. Config MCP (4 tools)

**Server:** `config`  
**Purpose:** System configuration management  
**Category:** System

#### `config_read(filename: str)`
**Purpose:** Read a configuration file  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `config_write(filename: str, content: str)`
**Purpose:** Write a configuration file  
**Read-Only:** No  
**Approval Required:** Yes (high safety)  
**Rate Limit:** 10/min

#### `config_list()`
**Purpose:** List all configuration files in the .agent/config directory  
**Read-Only:** Yes  
**Approval Required:** No  
**Rate Limit:** 100/min

#### `config_delete(filename: str)`
**Purpose:** Delete a configuration file  
**Read-Only:** No  
**Approval Required:** Yes (destructive)  
**Rate Limit:** 10/min

---

## 12. Orchestrator MCP (2 tools)

**Server:** `orchestrator`  
**Purpose:** Strategic workflows and missions  
**Category:** System

#### `orchestrator_dispatch_mission(mission_id: str, objective: str, assigned_agent: str = "Kilo")`
**Purpose:** Dispatch a mission to an agent  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 10/min

#### `orchestrator_run_strategic_cycle(gap_description: str, research_report_path: str, days_to_synthesize: int = 1)`
**Purpose:** Execute a full Strategic Crucible Loop: Ingest → Synthesize → Adapt → Cache  
**Read-Only:** No  
**Approval Required:** No  
**Rate Limit:** 1/hour

---

## Summary Statistics

### By Category

| Category | Servers | Tools | % of Total |
|----------|---------|-------|------------|
| Knowledge & Memory | 1 | 9 | 14.3% |
| Version Control | 1 | 9 | 14.3% |
| Code Operations | 1 | 9 | 14.3% |
| Task Management | 1 | 6 | 9.5% |
| Documentation | 3 | 16 | 25.4% |
| AI & Agents | 3 | 9 | 14.3% |
| System | 2 | 6 | 9.5% |
| **Total** | **12** | **63** | **100%** |

### By Read/Write

| Type | Count | % of Total |
|------|-------|------------|
| Read-Only | 38 | 60.3% |
| Write | 25 | 39.7% |

### By Approval Requirement

| Approval | Count | % of Total |
|----------|-------|------------|
| No Approval | 54 | 85.7% |
| Approval Required | 9 | 14.3% |

**Tools Requiring Approval:**
1. `cortex_ingest_full` (destructive)
2. `git_smart_commit` (destructive)
3. `git_push_feature` (remote write)
4. `git_finish_feature` (branch deletion)
5. `chronicle_update_entry` (if >7 days old)
6. `protocol_update` (if canonical)
7. `config_write` (high safety)
8. `config_delete` (destructive)

---

## Tool Naming Conventions

**Pattern:** `{server}_{action}[_{object}]`

**Examples:**
- `cortex_query` - Query the cortex
- `git_smart_commit` - Smart commit in git
- `task_create` - Create a task
- `adr_update_status` - Update ADR status

**Exceptions:**
- `create_task` (not `task_create`) - Legacy naming
- `query_sanctuary_model` (not `forge_query`) - Descriptive naming

---

## Rate Limits

**Default Rate Limits:**
- Read-only operations: 100 requests/minute
- Write operations: 50 requests/minute
- Destructive operations: 10 requests/minute
- Resource-intensive operations: 1 request/hour

**Custom Rate Limits:**
- `cortex_ingest_full`: 1/hour (expensive)
- `orchestrator_run_strategic_cycle`: 1/hour (expensive)

---

## Conclusion

The Gateway provides access to **63 tools** across **12 servers**, offering comprehensive functionality for:
- Knowledge management (RAG)
- Version control (Git)
- Code operations
- Task management
- Documentation (Chronicle, ADR, Protocol)
- AI agents (Council, Persona, Forge)
- System configuration

**60% of tools are read-only**, ensuring safe exploration and querying.  
**14% require approval**, protecting against destructive operations.
