# Workflow Enforcement & Development Policy

## Core Principle: Command-Driven Continuous Improvement

All agent interactions with the codebase MUST be mediated by **Antigravity Commands (Slash Commands)** found in `@[.agent/workflows]`. This ensures:
1. **Compound Intelligence** - Each command usage is an opportunity to improve the underlying tools.
2. **Reusability** - Workflows are codified, sharable, and consistently improved.
3. **Bypass Prohibition** - Using raw shell commands (`grep`, `cat`, `find`) on source data is STRICTLY PROHIBITED if a command exists.

---

## 1. Interaction Model (Command-First)

The Antigravity Command System is the **authoritative** interface for all Project Sanctuary tasks.

### Architecture (ADR-036: Thick Python / Thin Shim)
| Layer | Location | Purpose |
|:------|:---------|:--------|
| **Slash Commands** | `.agent/workflows/*.md` | The **Interface**. User-friendly workflows. |
| **Thin Shims** | `scripts/bash/*.sh` | The **Gateway**. Dumb wrappers that `exec` Python CLI. |
| **CLI Tools** | `tools/cli.py` | The **Router**. Dispatches to orchestrator/tools. |
| **Python Orchestrator** | `tools/orchestrator/` | The **Logic**. Enforcement, Git checks, ID generation. |

**Rule of Thumb:** Use a Slash Command to *do* a task; use a CLI tool to *implement* how that task is done.

---

## 2. Agent Usage Instructions (Enforcement)

- **Prioritize Commands**: Always check `.agent/workflows/` first.
- **Upgrade, Don't Bypass**: **NEVER** use `grep`/`find` if a tool exists. If the tool is lacking, **UPGRADE IT**.
- **Stop-and-Fix Protocol**: If a workflow is broken or rough:
    - **STOP** the primary task.
    - **UPGRADE** the tool/workflow to fix the friction.
    - **RESUME** using the improved command.
- **Progress Tracking**: Before major workflows, create a tracking doc (e.g., in `tasks/todo/`) to log progress.

---

## 3. Workflow Creation & Modification Standards

When creating new workflows, you MUST follow these standards:

### 3.1 File Standards
- **Location**: `.agent/workflows/[name].md`
- **Naming**: `kebab-case` (e.g., `workflow-bundle.md`)
- **Frontmatter**:
  ```yaml
  ---
  description: Brief summary of the workflow.
  tier: 1
  track: Factory # or Discovery
  ---
  ```

### 3.2 Architecture Alignment
- **Thin Shim**: If a CLI wrapper is needed, create `scripts/bash/[name].sh`.
- **No Logic in Shims**: Shims must only `exec` Python scripts.
- **Reuse**: Prefer using `/workflow-start` for complex flows. Only creation atomic shims for specific tools.

### 3.3 Registration Process (MANDATORY)
After creating/modifying a workflow (`.md`) or tool (`.py`):
1. **Inventory Scan**: `python tools/curate/documentation/workflow_inventory_manager.py --scan`
2. **Tool Registration**: `python tools/curate/inventories/manage_tool_inventory.py add --path <path>` (if new script)
3. **RLM Distillation**: `python tools/codify/rlm/distiller.py --file <path> --type tool`

---

## 4. Command Domains
- 🗄️ **Retrieve**: Fetching data (RLM, RAG).
- 🔍 **Investigate**: Deep analysis, mining.
- 📝 **Codify**: Documentation, ADRs, Contracts.
- 📚 **Curate**: Maintenance, inventory updates.
- 🧪 **Sandbox**: Prototyping.
- 🚀 **Discovery**: Spec-Driven Development (Track B).

---

## 5. Anti-Patterns (STRICTLY PROHIBITED)
❌ **Bypassing Tools**:
```bash
grep "pattern" path/to/source.py
find . -name "*.md" | xargs cat
```

✅ **Using/Improving Tools**:
```bash
python tools/retrieve/rlm/query_cache.py --type tool "pattern"
# If it fails, improve query_cache.py!
```
