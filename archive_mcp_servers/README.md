# 🛡️ Project Sanctuary MCP Servers - The Canonical Layer

This directory contains the canonical MCP servers for Project Sanctuary. The system is mature and includes a set of 15 specialized MCP servers (ADR 092) that provide stateful memory, tool-use, and governance capabilities used by the LLM clients and higher-order orchestrations.

Executive summary:
- Scope: 15 canonical MCP servers, production-grade testing pyramid, and canonical deploy flow.
- Maturity: The servers are covered by a three-layer test pyramid (Unit / Integration / E2E) and comprehensive docs under `docs/architecture/mcp/`.

## 🏛️ MCP Server Canonical List

The authoritative set of servers (folder names under `mcp_servers/`) and short purposes:

- `adr` — Architecture Decision Records
- `agent_persona` — Individual Agent Execution / Persona Dispatch
- `chronicle` — Audit Trail, History, and Sequential Records
- `code` — File / Code Operations (builders, analyzers)
- `config` — Configuration Management and helpers
- `council` — Multi-Agent Deliberation and Council workflows
- `evolution` — Self-Improvement and Mutation Tracking (Protocol 131)
- `forge_llm` — LLM Fine-Tuning / Inference orchestration
- `git` — Version Control Operations and commit/meta tooling
- `learning` — Session Lifecycle and Cognitive Continuity (Protocol 128)
- `orchestrator` — Strategic Mission Coordination (the System Brain)
- `protocol` — Protocol Management and validation
- `rag_cortex` — Knowledge Retrieval / Ingestion (System Memory)
- `task` — Task / Roadmap and mission tracking
- `workflow` — Standard Operating Procedures (SOPs)

Notes:
- The `orchestrator` and `rag_cortex` servers form the foundational pair for the Strategic Crucible Loop (see Protocol 056). They are the primary engines for planning, retrieval-augmented reasoning, and the strategic feedback cycle.
- Each server exposes one or more MCP tools. For deployment, the client (Claude Desktop, Antigravity, etc.) reads the JSON config produced by the deployer and starts the appropriate processes.

## Configuration & Deployer

We prefer relative paths and environment-variable expansion for portability. The canonical template is `.agent/mcp_config.json` (or `.agent/mcp_config.yml` if you elect to author YAML templates).

**Deployment Workflow Overview:**
1. **Set Environment Variables:** Define required variables (`PROJECT_SANCTUARY_ROOT`, `PYTHON_EXEC`, etc.) in your shell environment.
2. **Expand Template:** Use the deployer script to expand these variables in the canonical template (`.agent/mcp_config.json`).
3. **Write Config:** The deployer writes the expanded JSON into the platform-specific client configuration location (e.g., Claude Desktop, VS Code).

- Deployer: `mcp_servers/deploy_mcp_config.py` — expands `${VAR}` placeholders and writes the client config JSON.
- **Full Guide:** `mcp_servers/mcp_config_guide.md` — canonical pointer for where generated config files should be placed on each OS, along with helpers and examples.

## 🐍 Python Virtual Environments (.venv) Standard

In Project Sanctuary, all Python-based MCP servers **must** use a dedicated virtual environment located in the project root. This is not just a preference; it is the canonical pattern for ensuring dependency isolation and reliable execution across different MCP clients (Claude Desktop, Antigravity, VS Code).

### The .venv Pattern
The canonical configuration for an MCP server's `command` field is:
`"/Users/richardfremmerlid/Projects/Project_Sanctuary/.venv/bin/python"`

### Why This is Mandatory:
1.  **Dependency Isolation:** Prevents conflicts between server requirements (e.g., specific versions of LangChain or ChromaDB).
2.  **Explicit Resolution:** MCP servers run as independent background processes; pointing directly to the `.venv` binary ensures they find their specific "Library Closet" without needing an external `source activate` step.
3.  **Side-by-Side Sync:** For servers that also run in containers (like Cortex), the local `.venv` serves as the **Native Development Mirror**. It must be kept in sync with the container's `requirements.txt` to ensure consistent behavior between "Legacy" and "Gateway" modes.

### Setup & Maintenance:
```bash
# Navigate to project root
cd /Users/richardfremmerlid/Projects/Project_Sanctuary

# Create/Update the environment
python3 -m venv .venv
source .venv/bin/activate

# Install all canonical requirements
pip install -r mcp_servers/rag_cortex/requirements.txt
# (Repeat for other servers as needed)
```

## Configuration - The Environment-First Doctrine

In adherence to the **Doctrine of Successor-State** (Chronicle Entry 308), all MCP server configurations **must** use environment variable expansion (`${VARIABLE}`) rather than hardcoded or absolute paths. This ensures cross-platform compatibility and simplifies setup for future agents.

The canonical method uses Python module paths (`-m mcp_servers.X.server`) and the `${PROJECT_SANCTUARY_ROOT}` variable.
### Standard Pattern (CORRECT):
```json
{
    "command": "/Users/richardfremmerlid/Projects/Project_Sanctuary/.venv/bin/python",
    "args": ["-m", "mcp_servers.module_name.server"],
    "env": {
        "PYTHONPATH": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    }
}
```

### Anti-Pattern (INCORRECT):
```json
{
    "command": "python",  // ❌ Uses system Python, unstable across machines
    "args": ["-m", "mcp_servers.module_name.server"]
}
```

### ❌ Common Errors (Without proper .venv):
- `ModuleNotFoundError: No module named 'langchain_huggingface'`
- `ImportError: cannot import name 'CortexOperations'`
- Version conflicts between system packages and Project Sanctuary requirements.

### ✅ Verification Checklist:
- [ ] `.venv` directory exists in project root.
- [ ] `.venv/bin/python` is the designated interpreter in `mcp_config.json`.
- [ ] `pip list` inside the `.venv` shows all packages from `rag_cortex/requirements.txt`.
- [ ] `PYTHONPATH` in the MCP config points to the project root.

To configure an MCP client (e.g., Claude Desktop) to use the servers, add the following structure to your configuration file (e.g., `claude_desktop_config.json`).

### Canonical Configuration Example

This example for the `git` MCP reflects the environment-variable and Python module path convention used across all MCP servers (as seen in `config.json`):

```json
{
  "mcpServers": {
    "git_workflow": {
      "displayName": "Git Workflow MCP",
      "command": "${PYTHON_EXEC}",
      "args": [
        "-m",
        "mcp_servers.git.server"
      ],
      "env": {
        "PROJECT_SANCTUARY_ROOT": "${PROJECT_SANCTUARY_ROOT}",
        "PYTHONPATH": "${PROJECT_SANCTUARY_ROOT}",
        "REPO_PATH": "${PROJECT_SANCTUARY_ROOT}"
      },
      "cwd": "${PROJECT_SANCTUARY_ROOT}"
    }
  }
}
```

Required Environment Variables:

- `PROJECT_SANCTUARY_ROOT`: Absolute path to the root of the Project Sanctuary repository.
- `PYTHONPATH`: Set to `${PROJECT_SANCTUARY_ROOT}` so Python can resolve the `mcp_servers` modules.
- `PYTHON_EXEC`: Absolute path to the Python interpreter (e.g., your virtual environment's python or python3).
- `REPO_PATH` (Used by git MCP): Set to `${PROJECT_SANCTUARY_ROOT}`.

## Current Reality & Workaround

**Doctrine vs Reality:** The project retains the Environment‑First Doctrine as the intended, long‑term approach. However, some MCP clients — notably Claude Desktop — do not perform environment variable substitution when reading external JSON configs. In practice this required us to write hard‑coded absolute paths into the Claude config so the client can start all 15 MCP servers reliably.

**Warning:** The deployer and helper scripts may produce **hard‑coded** config files by default (absolute paths). This is intentional where the client requires it. Running the deployer without `--preserve-placeholders` will expand variables into absolute paths and overwrite the client config.

**Example (current hard‑coded form used by Claude Desktop):**

```json
{
  "mcpServers": {
    "git_workflow": {
      "displayName": "Git Workflow MCP",
      "command": "/Users/richardfremmerlid/Projects/Project_Sanctuary/.venv/bin/python",
      "args": ["-m","mcp_servers.git.server"],
      "env": {
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "PYTHONPATH": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "REPO_PATH": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "GIT_BASE_DIR": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      },
      "cwd": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    }
    /* ... rest of servers ... */
  }
}
```

**Recovery & Regeneration**

- Preview the deployer's expanded JSON (dry run):

```bash
python3 mcp_servers/deploy_mcp_config.py --target ClaudeDesktop --dry-run
```

- Regenerate (hard-coded absolute paths, default behavior) with a timestamped backup:

```bash
python3 mcp_servers/deploy_mcp_config.py --target ClaudeDesktop --backup
```

- Regenerate but preserve template placeholders (if you prefer to produce a portable template file instead of absolute paths):

```bash
python3 mcp_servers/deploy_mcp_config.py --target ClaudeDesktop --backup --preserve-placeholders --add-legacy-project-root
```

- Restore a previous backup (example — placeholder backup available):

```bash
# create a safety copy of the current config, then restore the placeholder backup
cp "$HOME/Library/Application Support/Claude/claude_desktop_config.json" \
   "$HOME/Library/Application Support/Claude/claude_desktop_config.json.$(date -u +%Y%m%dT%H%M%SZ).pre-restore.bak" 2>/dev/null || true
cp "$HOME/Library/Application Support/Claude/claude_desktop_config.json.20251212T031122Z.bak" \
   "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
```

After restoring or regenerating the file, restart Claude Desktop so it picks up the updated configuration.

**Notes:**
- We keep the Environment‑First Doctrine in the docs and templates because it documents intent and makes future improvements easier.
- The README documents the practical gap so operators understand why hard‑coded configs are currently in use and how to recover or re-generate configs when needed.
- **Side-by-Side Synchronization:** When modifying a server that exists both as a local script and a containerized cluster (e.g., `rag_cortex`), you **must** update both the local `.venv` AND the `Dockerfile`/`requirements.txt` in the gateway cluster folder. Failure to do so will result in "Missing Module" errors when switching between Native and Gateway modes.

## Server Launching

MCP servers are **launched automatically by MCP clients** (Claude Desktop, Antigravity, VS Code) based on the JSON configuration. There is no need to manually start servers.

**How it works:**
1. The client reads the config JSON (e.g., `claude_desktop_config.json`)
2. For each entry in `mcpServers`, the client spawns the process using the specified `command`, `args`, `env`, and `cwd`
3. The servers run as child processes of the client

**For debugging individual servers**, use the test pyramid:

```bash
# Run all MCP server tests (unit + integration)
pytest tests/mcp_servers/

# Run tests for a specific server
pytest tests/mcp_servers/git/ -v

# Run only unit tests for a server
pytest tests/mcp_servers/chronicle/unit/ -v
```

For environment setup scripts, see `start_sanctuary.sh` (macOS/Linux) and `start_sanctuary.ps1` (Windows) which set `PROJECT_SANCTUARY_ROOT` and `PYTHON_EXEC` automatically.

## 🧪 Testing & Validation — The Three-Layered Test Pyramid

Integrity of the MCP layer is maintained by a rigorous three-layer test approach. All tests live under `tests/mcp_servers/` and follow the same pyramid model per-server.

Note: component-level README and quick-run instructions live in `tests/mcp_servers/README.md`. See that file for per-server commands, structure expectations, and CI recommendations for component tests.

### Unit Tests (Leaf)

- Scope: Function-level logic, model behavior, and utilities.
- Location: `tests/mcp_servers/<server>/unit/`.
- Goal: Fast, isolated verification suitable for PRs and local TDD.

### Integration Tests (Mid)

- Scope: Integration points within a server (filesystem, local DB mocks, config parsing, I/O flows).
- Location: `tests/mcp_servers/<server>/integration/`.
- Goal: Confirm component interoperability (use `tmp_path` fixtures for safe I/O).

### End-to-End (Apex)

- Scope: Full flows exercised by the external MCP client (Claude Desktop / Antigravity). Typical E2E scenarios simulate mission-level tool chains (e.g., `cortex_query` → `agent_persona` dispatch → `orchestrator` plan).
- Invocation: Perform via the client using the generated JSON configs; see examples like `simple_orchestration_test.md` in the docs.

All tests are discoverable under `tests/mcp_servers/` (one folder per MCP server). Example quick runs:

```bash
# unit tests
pytest tests/mcp_servers/chronicle/unit/ -v

# integration tests
pytest tests/mcp_servers/chronicle/integration/ -v
```

CI: We recommend wiring a validator / test runner to execute unit and integration layers on PRs; E2E tests are executed optionally as part of deployment verification with a running client.

## 📚 Documentation & Governance

The canonical developer and architecture references live under `docs/architecture/mcp/`.
Key artifacts:

- System Architecture: `docs/architecture/mcp/architecture_diagram.md`
- Orchestration Workflows: `mcp_servers/council/orchestration_workflows.md`
- Git & Commit Standards: `docs/operations/git/git_workflow.md` (see Protocol 101/118)
- Tool Usage Protocols: `01_PROTOCOLS/118_Agent_Session_Initialization_and_MCP_Tool_Usage_Protocol.md`
- Core Values / Doctrine: `00_CHRONICLE/ENTRIES/311_the_gemini_signal_a_declaration_of_core_values.md`

Use these sources as the authoritative references for design, operational procedures, and governance.
