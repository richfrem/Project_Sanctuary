# Workflow Enforcement Policy

**Tool discovery details → `.agent/skills/tool_discovery/SKILL.md`**
**Spec workflow details → `.agent/skills/spec_kitty_workflow/SKILL.md`**

## Core Principle
All agent interactions MUST be mediated by **Slash Commands** (`.agent/workflows/*.md`). No bypassing with raw shell.

## Architecture (ADR-036: Thick Python / Thin Shim)

| Layer | Location | Purpose |
|:------|:---------|:--------|
| **Slash Commands** | `.agent/workflows/*.md` | User-facing interface |
| **Thin Shims** | `scripts/bash/*.sh` | Dumb wrappers that `exec` Python |
| **CLI Router** | `tools/cli.py` | Dispatches to orchestrator/tools |
| **Orchestrator** | `tools/orchestrator/` | Logic, enforcement, Git checks |

## Command Domains
- 🗄️ **Retrieve** — Fetching data (RLM, RAG)
- 🔍 **Investigate** — Deep analysis, mining
- 📝 **Codify** — Documentation, ADRs, contracts
- 📚 **Curate** — Maintenance, inventory updates
- 🧪 **Sandbox** — Prototyping
- 🚀 **Discovery** — Spec-Driven Development (Track B)

## Registration (MANDATORY after creating/modifying workflows or tools)
```bash
python tools/curate/documentation/workflow_inventory_manager.py --scan
python tools/curate/inventories/manage_tool_inventory.py add --path <path>
```

## Workflow File Standards
- **Location**: `.agent/workflows/[kebab-case-name].md`
- **Frontmatter**: `description`, `tier`, `track`
- **Shims**: No logic — only `exec` Python scripts
