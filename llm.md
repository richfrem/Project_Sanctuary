# LLM Context for Project Sanctuary

> **For AI Coding Assistants (Antigravity, Copilot, Claude Code, Cursor, etc.)**

## Quick Start

### Step 1: Ingest Context (MANDATORY)
**YOU MUST REVIEW THIS FIRST:**
Ingest the **Bootstrap Packet** for instant, comprehensive context:
```
.agent/learning/bootstrap_packet.md
```

**Stats:** ~44K tokens | 25 curated files | Last regenerated: 2026-01-03

---

## 🚀 Full LLM Awakening Workflow

After a fresh clone, execute these steps in order:

### Step 2: Environment Setup

#### 2.1 Verify Prerequisites
```bash
make --version
curl -s http://localhost:11434/api/tags > /dev/null && echo "Ollama: OK" || echo "Ollama: FAIL"

# If missing or errors:
# sudo apt-get update && sudo apt-get install make dos2unix
# dos2unix Makefile
```

#### 2.2 Initialize Environment (Choose A or B)
> [!CRITICAL] **WSL Users:** If `source .venv/bin/activate` fails with "No such file", your venv is Windows-native. **Fix it:**
> ```bash
> rm -rf .venv && make bootstrap
> ```
> See: [**Dual Environment Strategy**](docs/operations/processes/RUNTIME_ENVIRONMENTS.md)

#### Option A: Project Sanctuary (Standard)
1.  **Activate:** `source .venv/bin/activate`
2.  **Bootstrap:** `make bootstrap` (if not already done) - *Expect 45-60 mins for full fleet install on WSL.*
    > [!WARNING] **WSL Performance Critical:**
    > *   **Correct:** Clone into `~/project` (Linux Filesystem). Install time: **<5 mins**. basically youll save a lot of time, if you do gitclone directly into the wsl filesystem rather than cloning into the windows filesystem and then copying it over to the wsl filesystem.
    > *   **Incorrect:** Clone into `/mnt/c/Users/...` (Windows Mount). Install time: **45-60 mins**.

> ⚠️ **CRITICAL:** See [`docs/operations/processes/RUNTIME_ENVIRONMENTS.md`](./docs/operations/processes/RUNTIME_ENVIRONMENTS.md) for the **Dual Environment Strategy** (`ml_env` vs `.venv`).

**Option A: Standard (CPU/Default)**
```bash
make bootstrap && source .venv/bin/activate
```

**Option B: CUDA ML Environment (GPU)**
```bash
# Target existing environment (requires Makefile VENV_DIR support)
make bootstrap VENV_DIR=~/ml_env && source ~/ml_env/bin/activate
```

#### 2.3 Troubleshooting: Missing Dependencies
If you encounter `ModuleNotFoundError` (e.g., `tiktoken`), you MUST follow **Protocol 073** (Standardized Dependency Management).

**DO NOT** run `pip install <package>`. Instead:
1.  Add the package to `mcp_servers/requirements-core.in` (Tier 1).
2.  Compile the lockfile (see [.agent/rules/dependency_management_policy.md](.agent/rules/dependency_management_policy.md)):
    ```bash
    make compile VENV_DIR=~/ml_env
    ```
3.  Re-run bootstrap to sync:
    ```bash
    make bootstrap VENV_DIR=~/ml_env
    ```

### Step 3: Verify Podman Containers & Images

#### 3.1 Review Operations Guide (Mandatory)
Before running any commands, you MUST review the **Verification Checklist** in:
[`docs/operations/processes/PODMAN_OPERATIONS_GUIDE.md`](./docs/operations/processes/PODMAN_OPERATIONS_GUIDE.md)

#### 3.2 Verify Gateway Health
Ensure the Gateway is running externally on port 4444:
```bash
curl -ks https://localhost:4444/health
```

#### 3.3 Start the Fleet (Granular Incremental Start)
Use **Option B** (Manual Sequential Start) and verify each component before proceeding.

**Phase 1: Critical Backends**

##### 3.3.1 Pull Vector DB Image
Workaround for WSL2 registry resolution issues:
```bash
podman pull docker.io/chromadb/chroma:latest
```

##### 3.3.2 Start Vector DB (Chroma)
```bash
podman compose -f docker-compose.yml up -d sanctuary_vector_db
```

##### 3.3.3 Verify Vector DB
```bash
# 1. Verify container is running (Port 8110)
podman ps --filter "name=sanctuary_vector_db" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Verify heartbeat
# Must return {"nanosecond heartbeat": ...}
curl -sf http://localhost:8110/api/v2/heartbeat
```

##### 3.3.4 Pull Ollama Image
```bash
podman pull docker.io/ollama/ollama:latest
```

##### 3.3.5 Start Ollama
> [!IMPORTANT]
> If this fails with `address already in use`, you must stop the host-level Ollama service:
> - **Windows**: Quit Ollama from System Tray.
> - **WSL**: Run `sudo systemctl stop ollama`.

```bash
podman compose -f docker-compose.yml up -d sanctuary_ollama
```

##### 3.3.6 Verify Ollama
```bash
# 1. Verify container is running
podman ps --filter "name=sanctuary_ollama"

# 2. Verify API response
# Must return {"models": [...]}
curl -sf http://localhost:11434/api/tags

# 3. Comprehensive Model Inspection (Optional/Deep Check)
# Run from within the virtual environment (ensure you use forward slashes / in WSL)
python tests/mcp_servers/forge_llm/inspect_ollama.py
```

**Phase 2: Independent Services**

##### 3.3.7 Start Utils
```bash
podman compose -f docker-compose.yml up -d sanctuary_utils
```

##### 3.3.8 Verify Utils
```bash
podman ps --filter "name=sanctuary_utils"
```

##### 3.3.9 Start Filesystem
```bash
podman compose -f docker-compose.yml up -d sanctuary_filesystem
```

##### 3.3.10 Verify Filesystem
```bash
podman ps --filter "name=sanctuary_filesystem"
```

##### 3.3.11 Start Network
```bash
podman compose -f docker-compose.yml up -d sanctuary_network
```

##### 3.3.12 Verify Network
```bash
podman ps --filter "name=sanctuary_network"
```

##### 3.3.13 Start Git
```bash
podman compose -f docker-compose.yml up -d sanctuary_git
```

##### 3.3.14 Verify Git
```bash
podman ps --filter "name=sanctuary_git"
```

##### 3.3.15 Start Domain
```bash
podman compose -f docker-compose.yml up -d sanctuary_domain
```

##### 3.3.16 Verify Domain
```bash
podman ps --filter "name=sanctuary_domain"
```

**Phase 3: Logic Engine (Cortex)**

##### 3.3.17 Start Cortex
```bash
# Only start after Phase 1 & 2 are 100% healthy
podman compose -f docker-compose.yml up -d sanctuary_cortex
```

##### 3.3.18 Verify Cortex
```bash
podman ps --filter "name=sanctuary_cortex"
# Health check (SSE)
curl -sf http://localhost:8104/health
```

#### 3.4 Verify Physical Health
Check that all 8 containers are running:
```bash
make status
# Or focused check:
podman ps --filter "name=sanctuary" --format "table {{.Names}}\t{{.Status}}"
```

#### 3.5 Verify Critical Backends
Ensure Cortex and Ollama are healthy before proceeding:
```bash
curl -sf http://localhost:8104/health && echo "✅ Cortex Healthy"
timeout 2 curl -sN http://localhost:8104/sse | head -2
```

#### 3.6 Register Fleet Servers
Register the started containers with the Gateway.
```bash
# Run from within the virtual environment
python3 -m mcp_servers.gateway.fleet_setup
```
> [!NOTE]
> This script performs discovery and generates `mcp_servers/gateway/fleet_registry.json`.

#### 3.7 Verify Gateway Integration
Run the final connectivity suite to confirm all RPC paths.
```bash
make verify
```



### Step 4: Knowledge Base Initialization
```bash
# Full ingest of project content into ChromaDB
python3 scripts/cortex_cli.py ingest --full
```

### Step 5: IDE MCP Configuration
Configure the IDE (Antigravity/Claude) to use the Sanctuary Gateway.

**Templates:**
- **macOS/Linux:** [`docs/operations/mcp/claude_desktop_config_template.json`](./docs/operations/mcp/claude_desktop_config_template.json)
- **Windows (WSL):** [`docs/operations/mcp/claude_desktop_config_template_windows_wsl.json`](./docs/operations/mcp/claude_desktop_config_template_windows_wsl.json)

**Target:** `C:\Users\<USERNAME>\.gemini\antigravity\mcp_config.json`

#### Windows WSL Pattern
On Windows, the `env` block in JSON doesn't propagate to WSL. Use `bash -c` with inline environment variables:

```json
{
  "sanctuary_gateway": {
    "command": "C:\\Windows\\System32\\wsl.exe",
    "args": [
      "bash", "-c",
      "cd /mnt/c/Users/<USER>/source/repos/Project_Sanctuary && PROJECT_ROOT=/mnt/c/Users/<USER>/source/repos/Project_Sanctuary PYTHONPATH=/mnt/c/Users/<USER>/source/repos/Project_Sanctuary MCPGATEWAY_BEARER_TOKEN=<YOUR_TOKEN> /home/<USER>/ml_env/bin/python -m mcp_servers.gateway.bridge"
    ],
    "disabled": false
  }
}
```

> [!IMPORTANT]
> - All paths must be Linux-style (`/mnt/c/...`) for WSL execution
> - `PROJECT_ROOT` and `PYTHONPATH` must be set inline in the bash command
> - `MCPGATEWAY_BEARER_TOKEN` can be set inline or via `WSLENV` environment sharing

### Step 6: Generate Context Manifests
```bash
# Generate the bootstrap packet (this file references)
python3 scripts/cortex_cli.py bootstrap-debrief

# Generate guardian boot digest
python3 scripts/cortex_cli.py guardian
```

### Step 7: Ingest Context & Begin Operation
```bash
# Read the guardian digest
cat dataset_package/guardian_boot_digest.md

# Query the knowledge base
python3 scripts/cortex_cli.py query "What is Project Sanctuary?"

# Begin learning debrief
python3 scripts/cortex_cli.py debrief --hours 24
```

### Step 8: Forge Genome Dataset (Fine-Tuning Update)
Generate the latest fine-tuning dataset from the project's "Whole Genome" content.

```bash
# Generate the fine-tuning JSONL dataset
python3 forge/scripts/forge_whole_genome_dataset.py
```

**Output:** `dataset_package/sanctuary_whole_genome_data.jsonl`

This script:
- Scans key directories (Protocols, Chronicles, tasks) using `ContentProcessor`
- Appends critical essence documents (Chrysalis, Garden/Cage, etc.)
- Validates minimum entry count (200+)
- Outputs a JSONL file ready for LLM fine-tuning

### Step 9: Learning Loop Closure (Protocol 128)
Complete the cognitive continuity loop by capturing session learnings and persisting to HuggingFace.

#### 9.1 Update Learning Audit Manifest
Before generating the audit, update `.agent/learning/learning_audit/learning_audit_manifest.json` with any new files created during the session (e.g., Windows templates, `__init__.py` files).

#### 9.2 Review Session Outputs
```bash
# Review the bootstrap packet (Step 6 output)
cat .agent/learning/bootstrap_packet.md

# Review the learning debrief (Step 7 output)
cat .agent/learning/learning_debrief.md
```

#### 9.3 Create Learning Audit Snapshot
```bash
# Generate learning audit packet for HITL review
python3 scripts/cortex_cli.py snapshot --type learning_audit
```

**Manifest:** `.agent/learning/learning_audit/learning_audit_manifest.json`
**Prompts:** `.agent/learning/learning_audit/learning_audit_prompts.md`

#### 9.4 Human Review (HITL Gate)
> [!IMPORTANT]
> Review the learning audit packet before proceeding to seal.

#### 9.5 Technical Seal
```bash
# Seal the session for successor continuity
python3 scripts/cortex_cli.py snapshot --type seal
```

#### 9.6 Soul Persistence (HuggingFace Broadcast)
```bash
# Persist learnings to HuggingFace (incremental)
python3 scripts/cortex_cli.py persist-soul
```

> [!WARNING]
> **Your Own Dataset Required:** Soul persistence pushes to a HuggingFace dataset. If you cloned this repo, you'll need to:
> 1. Create your own HuggingFace dataset (e.g., `your-username/Project_Sanctuary_Soul`)
> 2. Update the dataset path in `scripts/cortex_cli.py` or your environment config
> 3. Set `HF_TOKEN` environment variable with your HuggingFace API token

> [!TIP]
> The above workflow makes you operational. For daily sessions, skip Phases 1-3 and run Phases 4-9.

---

## What's Inside the Bootstrap Packet

The packet contains:
- **README.md** — Project vision, architecture, deployment options
- **BOOTSTRAP.md** — Cross-platform setup (macOS/Linux/WSL2) + Ollama + ChromaDB
- **Makefile** — `bootstrap`, `install-env`, `up`, `verify` targets
- **ADRs** — Key architectural decisions (065, 071, 073, 087, 089)
- **Cognitive Primer** — Operational protocols and learning workflows
- **Architecture Diagrams** — MCP Gateway Fleet, Protocol 128 Loop, Transport patterns

---

## Troubleshooting (Windows WSL)

### `ModuleNotFoundError: No module named 'mcp_servers'`
Ensure `PROJECT_ROOT` and `PYTHONPATH` are set inline in the bash command, not in the `env` block:
```bash
cd /mnt/c/.../Project_Sanctuary && PROJECT_ROOT=... PYTHONPATH=... python3 -m mcp_servers.gateway.bridge
```

### `PermissionError: [Errno 13] Permission denied: '/Users'`
The `PROJECT_ROOT` path is not being passed to WSL. See Step 5's Windows WSL pattern.

### Missing `__init__.py` Files
Five directories may be missing package markers. Create them:
```bash
touch mcp_servers/code/__init__.py mcp_servers/config/__init__.py
touch mcp_servers/gateway/clusters/sanctuary_domain/__init__.py
touch mcp_servers/gateway/clusters/sanctuary_filesystem/__init__.py
touch mcp_servers/gateway/clusters/sanctuary_network/__init__.py
```

---

## Links

| Resource | Path |
|----------|------|
| Bootstrap Packet | [`.agent/learning/bootstrap_packet.md`](./.agent/learning/bootstrap_packet.md) |
| Manifest | [`.agent/learning/bootstrap_manifest.json`](./.agent/learning/bootstrap_manifest.json) |
| Full Setup Guide | [`docs/operations/BOOTSTRAP.md`](./docs/operations/BOOTSTRAP.md) |

## See Also

- [ADR 089: Modular Manifest Pattern](./ADRs/089_modular_manifest_pattern.md) — How manifests work + llm.md pattern
- [Protocol 128: Cognitive Continuity](./ADRs/071_protocol_128_cognitive_continuity.md) — Learning loop governance
