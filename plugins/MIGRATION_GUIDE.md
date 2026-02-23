# Plugin Migration Guide
> For agents adopting the `plugins/` architecture in a new or existing repo.

## Plugin Catalog

| Plugin | Description | Has Scripts | Deps |
|:---|:---|:---:|:---|
| `adr-manager` | Create/list/search Architecture Decision Records with auto-numbering | ✅ (2) | None |
| `agent-orchestrator` | Dual-loop agent delegation (Outer=plan, Inner=execute) | ✅ (1) | spec-kitty-cli |
| `claude-cli` | Pipe large contexts to Anthropic models with persona-based analysis | ❌ | claude CLI |
| `code-snapshot` | Bundle repo files into token-counted, role-specific context packages | ✅ (3) | tiktoken |
| `coding-conventions` | Code standards and header templates (Python/TS/C#) | ❌ | None |
| `context-bundler` | Bundle source files into single-file LLM context packages | ✅ (3) | None |
| `dependency-management` | pip-compile locked-file workflow reference | ❌ | pip-tools |
| `link-checker` | Validate \u0026 auto-repair broken documentation links | ✅ (3) | None |
| `mermaid-export` | Render .mmd diagrams to PNG/SVG | ✅ (1) | node, @mermaid-js/mermaid-cli |
| `plugin-manager` | Universal Installer — maps plugins to agent environments | ✅ (1) | None |
| `rlm-factory` | Distill semantic summaries using Ollama for context retrieval | ✅ (5) | ollama, requests |
| `spec-kitty` | Spec-Driven Development lifecycle + Universal Bridge sync | ✅ (5) | spec-kitty-cli |
| `task-manager` | JSON-backed kanban board (backlog/todo/in-progress/done) | ✅ (1) | None |
| `tool-inventory` | Tool registries + embedded ChromaDB for semantic discovery | ✅ (7) | chromadb |
| `vector-db` | Semantic search via ChromaDB + Super-RAG context injection | ✅ (4) | chromadb, sentence-transformers |
| `workflow-inventory` | Registry of official agent workflows and slash commands | ✅ (1) | None |
| `stock-valuation` | AI-driven stock valuation (Bear/Base/Bull) with research reports | ✅ (1) | yfinance |
| `thesis-balancer` | Portfolio health monitoring, drift analysis, thesis alignment | ❌ | None |

> **ADR-021: Direct Plugin Execution**  
> We no longer mirror scripts to the `tools/` folder. All tools are executed directly from their canonical `plugins/` locations. The `tools/` directory is reserved for project-level routers (like `cli.py`) and application-specific logic.

> **Excluded**: `chronicle-manager`, `protocol-manager` (project-specific, not portable).

---

## Plugin Anatomy (Standard)

```
plugins/<name>/
├── .claude-plugin/
│   └── plugin.json          # Manifest (name, version, deps, keywords)
├── commands/                # Slash commands (*.md)
│   └── <action>.md          # /plugin-name:action
├── scripts/                 # Implementation (*.py, *.sh)
│   └── <tool>.py
├── skills/                  # Agent persona / SKILL.md
│   └── <skill-name>/
│       └── SKILL.md
├── docs/                    # Architecture diagrams, etc.
└── README.md
```

---

## Workflow Directory Organization

Workflows in `.agent/workflows/` are organized into **subdirectories matching their plugin name**.

```
.agent/workflows/
├── adr-manager/
│   ├── adr-manager_create.md
│   └── adr-manager_list.md
├── spec-kitty/              ← Managed by speckit_system_bridge.py
│   ├── spec-kitty.accept.md
│   ├── spec-kitty.implement.md
│   └── ...
├── stock-valuation/
│   └── stock-valuation_evaluate-stock.md
├── thesis-balancer/
│   └── thesis-balancer_review-portfolio.md
└── tool-inventory/
    ├── tool-inventory_add.md
    └── ...
```

### How Workflows Get There

| Source | Mechanism | Output Directory |
|:---|:---|:---|
| `plugins/*/commands/*.md` | `bridge_installer.py` | `.agent/workflows/{plugin-name}/` |
| `.windsurf/workflows/` (Spec Kitty CLI) | `speckit_system_bridge.py` | `.agent/workflows/spec-kitty/` |
| Custom sync | `sync_workflows.py` | `.agent/workflows/{plugin-name}/` (recursive, skips `spec-kitty/`) |

### Key Scripts Updated for Subdirectory Support

| Script | Change |
|:---|:---|
| `workflow_inventory_manager.py` | `glob("*.md")` → `rglob("*.md")` |
| `speckit_system_bridge.py` | Output path → `workflows/spec-kitty/` |
| `bridge_installer.py` | Output path → `workflows/{plugin_name}/` |
| `sync_workflows.py` | Already recursive ✅ |

---

## Migration Strategy (6 Phases)

### Phase 1: Inventory Old Tools
Build a JSON manifest of every existing tool reference.

```bash
# Agent should create: migration_inventory.json
{
  "plugins/vector-db/scripts/ingest.py": {
    "new_path": "plugins/vector-db/scripts/ingest.py",
    "references_found": 12,
    "status": "pending"
  },
  "plugins/rlm-factory/scripts/query_cache.py": {
    "new_path": "plugins/rlm-factory/scripts/query_cache.py",
    "references_found": 8,
    "status": "pending"
  }
}
```

### Phase 1: Inventory Old Tools
Build a JSON manifest of every existing tool reference.

```bash
# Agent should create: migration_inventory.json
{
  "plugins/vector-db/scripts/ingest.py": {
    "new_path": "plugins/vector-db/scripts/ingest.py",
    "references_found": 12,
    "status": "pending"
  }
}
```

**How to build**:
1. Run `python plugins/migration-utils/scripts/generate_inventory.py`.
2. This script now uses `query_cache.py` to dynamically map old `tools/` filenames to their new `plugins/` locations using the RLM content ledger.
3. It keeps `KNOWN_MAPPINGS` as overrides but prefers dynamic cache lookup for everything else, ensuring the inventory matches the actual on-disk state.

### Phase 2: Inventory Old Workflows/Skills/Rules
Same approach for non-script artifacts:

```bash
# Scan these directories:
# .agent/workflows/     → commands/ in the relevant plugin
# .agent/skills/        → skills/ in the relevant plugin
# .agent/rules/         → stays (rules are project-specific config)
# .claude/commands/     → generated by plugin-bridge
# .github/prompts/      → generated by plugin-bridge
# .gemini/commands/     → generated by plugin-bridge
```

### Phase 3: Update `cli.py` References
The CLI router (`tools/cli.py`) contains hardcoded paths. Update these:

```python
# OLD (scattered across tools/)
DOCS_DIR = Path(resolve_path("tools/codify/documentation"))
RLM_DIR = Path(resolve_path("tools/retrieve/rlm"))
INVENTORIES_DIR = Path(resolve_path("tools/curate/inventories"))

# NEW (consolidated in plugins/)
RLM_DIR = Path(resolve_path("plugins/rlm-factory/scripts"))
INVENTORIES_DIR = Path(resolve_path("plugins/tool-inventory/scripts"))
```

> [!IMPORTANT]
> Keep `tools/cli.py` as the main entry point. Only update the path constants it references.

### Phase 4: Search \u0026 Replace Script
Create a temporary migration script:

```python
#!/usr/bin/env python3
"""migration_replace.py — Batch update tool references."""
import json, re
from pathlib import Path

# Load inventory
with open("migration_inventory.json") as f:
    inventory = json.load(f)

# File extensions to scan
SCAN_EXTS = {".py", ".md", ".sh", ".yaml", ".yml", ".json", ".toml"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}

def scan_and_replace(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in SCAN_EXTS:
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            content = path.read_text(errors="ignore")
            changed = False
            for old_path, info in inventory.items():
                if old_path in content:
                    content = content.replace(old_path, info["new_path"])
                    info["status"] = "replaced"
                    changed = True
            if changed:
                path.write_text(content)
                print(f"  Updated: {path}")

    # Save updated inventory
    with open("migration_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

if __name__ == "__main__":
    scan_and_replace(Path("."))
```

### Phase 5: Install via Plugin Bridge
After all references point to `plugins/`, run the bridge to populate agent-specific environments.

```bash
# Install all plugins to all detected environments
python3 plugins/plugin-bridge/scripts/install_all_plugins.py
```

This auto-populates:
- `.agent/workflows/`, `.agent/skills/` (Antigravity)
- `.claude/commands/` (Claude Desktop)
- `.github/prompts/` (Copilot)
- `.gemini/commands/` (Gemini)

> [!IMPORTANT]
> The bridge no longer mirrors scripts to `tools/`. It strictly manages agent-facing configurations (workflows, prompts, instructions).

### Phase 6: Selective Cleanup & Verification
Once references are updated, verify and clean up:

1.  **Run Audit Tool**: Use the dedicated audit script to find any remaining stale references.
    ```bash
    # Check for stale refs (dry run)
    python3 plugins/migration-utils/scripts/audit_stale_refs.py
    
    # Auto-fix found references
    python3 plugins/migration-utils/scripts/audit_stale_refs.py --fix
    ```
2.  **Delete Mapped Files**: Remove *only* the specific files that were successfully mapped and exist in `plugins/`.
3.  **Preserve Directories**: Do not `rm -rf` entire folders if they contain unmapped files.
4.  **Review Unmapped**: Create a list of unmapped files to evaluate for new plugin creation.

```bash
# Example safety check before deletion
# python tools/curate/migration/verify_deletion.py
```

> [!CAUTION]
> Only delete after running the migration script AND verifying all tests pass. Create a git branch first.

### Real World Deviations (Lessons Learned)
During actual migration, several categories of files will effectively be orphaned or are better suited for deletion rather than migration:

1.  **Standalone Tools (`tools/standalone/`)**: These are typically one-off bundles or previous export attempts. **Do not migrate**. Delete them in Phase 6.
2.  **Test Infrastructure**: Files like `plugins/legacy-system-oracle-forms/scripts/test_infrastructure.py` might not have a direct plugin equivalent yet. Move them to a `plugins/shared/` or `plugins/legacy-tests/` if essential, otherwise archive.
3.  **Unmapped Scripts**: Scripts like `plugins/legacy-doc-gen/scripts/generate_todo_list.py` that do not map to an existing plugin. **Do NOT delete these**.
    *   **Action**: Analyze these scripts.
    *   **Decision**: If useful, **create a new plugin** (e.g., `plugins/productivity-tools/`) and migrate them there.
    *   **Fallback**: Leave them in `tools/` until a decision is made.
4.  **Config Files**: Files like `plugins/tool-inventory/scripts/rlm_config.py` should be moved to `plugins/tool-inventory/scripts/rlm_config.py` (or similar) to match the new architecture.
5.  **Investment Screener**: The `tools/investment-screener/` directory appears to be a full application/backend. **Do not migrate this as a standard plugin**. It should likely remain as a standalone app or be refactored into its own top-level directory (e.g., `apps/investment-screener`). For now, **exclude from migration** to avoid breakage.
6.  **Migration Utilities**: A set of helper scripts has been created in `plugins/migration-utils/scripts/` to assist with the migration. these include `generate_inventory.py`, `migration_replace.py`, `cleanup_mapped_files.py`, and `audit_stale_refs.py`. These should be preserved for future use or reference.
7.  **Path Resolution**: Scripts migrated from `tools/` to `plugins/` (especially those using `__file__` to find `PROJECT_ROOT`) need their path resolution logic updated. For example, moving from `tools/bridge` (depth 2) to `plugins/spec-kitty/scripts` (depth 3) requires adding an extra `.parent` call.
8.  **Zombie Skills**: Skills installed in `.agent/skills` from old tools might persist if not explicitly pruned. Verify `.agent/skills` matches `plugins/*/skills` after migration.

> [!IMPORTANT]
> **Forward-Looking Strategy**: Items that do not map are likely candidates for **new plugins**. Do not discard them.

> [!IMPORTANT]
> **Conservative Deletion Policy**: Only delete files that have been successfully migrated and verified. If a script in `tools/` has no clear mapping in `migration_inventory.json` (status="unmapped"), **leave it alone**. Do not delete it.

#### Updated Inventory Strategy
The initial exact-match inventory script will miss many files. The recommended approach is:
1.  **Auto-match**: Match known filenames to `plugins/` scripts.
2.  **Manual Map**: Hardcode mappings for renamed or moved logic (e.g., `path_resolver.py` -> `context-bundler`).
3.  **Exclude**: Explicitly exclude `tools/standalone/` and `tools/investment-screener/` from the "pending" list to reduce noise.

---

## Known Old → New Path Mappings

| Old Path | New Plugin Path |
|:---|:---|
| `plugins/vector-db/scripts/ingest.py` | `plugins/vector-db/scripts/ingest.py` |
| `plugins/vector-db/scripts/query.py` | `plugins/vector-db/scripts/query.py` |
| `plugins/vector-db/scripts/cleanup.py` | `plugins/vector-db/scripts/cleanup.py` |
| `plugins/rlm-factory/scripts/distiller.py` | `plugins/rlm-factory/scripts/distiller.py` |
| `plugins/rlm-factory/scripts/query_cache.py` | `plugins/rlm-factory/scripts/query_cache.py` |
| `plugins/tool-inventory/scripts/manage_tool_inventory.py` | `plugins/tool-inventory/scripts/manage_tool_inventory.py` |
| `plugins/link-checker/scripts/check_broken_paths.py` | `plugins/link-checker/scripts/check_broken_paths.py` |
| `plugins/link-checker/scripts/map_repository_files.py` | `plugins/link-checker/scripts/map_repository_files.py` |
| `plugins/context-bundler/scripts/bundle.py` | `plugins/context-bundler/scripts/bundle.py` |
| `plugins/spec-kitty/scripts/speckit_system_bridge.py` | `plugins/spec-kitty/scripts/speckit_system_bridge.py` |
| `plugins/spec-kitty/scripts/sync_workflows.py` | `plugins/spec-kitty/scripts/sync_workflows.py` |
| `plugins/spec-kitty/scripts/sync_rules.py` | `plugins/spec-kitty/scripts/sync_rules.py` |

> This table is a starting point. The agent should build the complete inventory in Phase 1.

---

### Phase 7: RLM Refresh & Inventory Update
Finalize the migration by ensuring the semantic discovery layer is up-to-date.

1.  **Run Audit Tool**: Validate inventory vs filesystem and RLM cache.
    ```bash
    python3 plugins/tool-inventory/scripts/audit_plugins.py
    ```
2.  **Verify Inventory**: Ensure `tools/tool_inventory.json` paths point to canonical `plugins/` locations.
3.  **Verify RLM Config**: Ensure `rlm_config.py` uses robust root detection.
4.  **Clear Stale Cache**: Remove old `tools/` paths from the RLM ledger.
    ```bash
    python3 plugins/tool-inventory/scripts/cleanup_cache.py --type tool --apply --prune-orphans
    ```
5.  **Distill Missing Plugins**: If audit reports "Missing from RLM Cache", run the distillation agent.
    ```bash
    # Option A: Agent-Driven (High Quality, see .agent/workflows/tool-inventory/tool-inventory_distill-agent.md)
    # Option B: Manual CLI -> python3 plugins/tool-inventory/scripts/distiller.py --file <path>
    ```

6.  **Legacy Cleanup (Optional)**:



---

## Development Workflow: Plugin-First

To maintain strict separation of concerns and ensure portability, always follow the **Plugin-First** rule:

1.  **Modify in `plugins/`**: All logic changes, script updates, or configuration tweaks MUST be made in the `plugins/<name>/scripts/` directory.
2.  **Verify Locally**: Run tests or verification steps against the plugin source.
3.  **No Mirroring**: Do NOT copy scripts to the `tools/` directory. If a script is useful at the project level, add a router entry in `tools/cli.py`.
4.  **Commit Plugins**: All authoritative code lives in `plugins/`.

### Phase 9: Agent-Driven "Flash Distill"
...
> [!TIP]
> **Bulk Distill**: To distill the entire toolset, process each script individually for accuracy. Then run:
> ```bash
> python3 plugins/tool-inventory/scripts/manage_tool_inventory.py sync-from-cache --cache .agent/learning/rlm_tool_cache.json
> python3 plugins/tool-inventory/scripts/manage_tool_inventory.py generate
> ```

#### Phase 9 Results (Completed 2026-02-15)

| Metric | Value |
|:---|:---|
| **Scripts Distilled** | 45 (all plugins + project-level tools) |
| **Descriptions Enriched** | 33 tools updated in `tool_inventory.json` |
| **Cache Verified** | Semantic queries for "Universal Bridge", "Kanban", "strategy packet", "inventory" all returned correct matches |
| **Time vs Granite** | ~10 minutes (agent) vs ~2+ hours (local Ollama) |

---

## Environment Configuration

The RLM and semantic discovery layers support project-wide configuration via a `.env` file in the **project root**:

| Variable | Default | Purpose |
|:---|:---|:---|
| `OLLAMA_MODEL` | `granite3.2:8b` | The model used for semantic distillation |
| `OLLAMA_HOST` | `http://localhost:11434` | The endpoint for the local Ollama server |
| `RLM_TOOL_CACHE` | `.agent/learning/rlm_tool_cache.json` | Path to the tool discovery ledger |
| `RLM_SUMMARY_CACHE` | `.agent/learning/rlm_summary_cache.json` | Path to the general project ledger |

> [!TIP]
> Use `.env` to switch between models (e.g., `qwen2.5:7b`) or to redirect cache files to temporary locations during massive refactors.

---

## Project-Specific Plugins

When applying this architecture to a **new project repo**, the portable plugins (listed in the catalog above) cover agent infrastructure. However, every project will have **domain-specific workflows** that need their own plugins.

### Identifying Project-Specific Plugins

Ask these questions when onboarding a new repo:

1. **What domain workflows exist?** Scan `.agent/workflows/` and `.agent/skills/` for commands that are NOT covered by the portable plugin catalog.
2. **What architecture docs exist?** Check `docs/architecture/` for design documents that should live alongside a plugin.
3. **What application-specific scripts exist?** Look in `tools/` for scripts tied to the project's business logic (not agent infrastructure).

### Example: MyProject

This project identified two domain plugins:

| Plugin | Origin | Contains |
|:---|:---|:---|
| `stock-valuation` | `/evaluate-stock` workflow + `stock_valuation` skill + `docs/architecture/stock-valuation/` | Commands, skill, architecture docs |
| `thesis-balancer` | `/review-portfolio` workflow + `thesis-balancer` skill + `docs/architecture/thesis-alignment-and-portfolio-valuation/` | Commands, skill, architecture docs |

### Template: Creating a Project Plugin

```bash
mkdir -p plugins/{name}/commands plugins/{name}/skills/{name} plugins/{name}/docs
# 1. Move workflow → plugins/{name}/commands/
# 2. Move skill → plugins/{name}/skills/{name}/
# 3. Move relevant architecture docs → plugins/{name}/docs/
# 4. Create README.md
# 5. Run bridge_installer.py to install into .agent/workflows/{name}/
```

> [!IMPORTANT]
> **Portable vs Project-Specific**: Plugins like `spec-kitty`, `rlm-factory`, `tool-inventory` are **portable** across repos. Plugins like `stock-valuation` are **project-specific** and should NOT be copied to unrelated repos.

---

## Checklist for Target Agent

- [ ] Copy `plugins/` directory (minus `chronicle-manager`, `protocol-manager`) to target repo
- [x] Run Phase 1: Build `migration_inventory.json`
- [x] Run Phase 2: Inventory workflows/skills/rules
- [x] Run Phase 3: Update `cli.py` path constants
- [x] Run Phase 4: Execute `migration_replace.py`
- [x] Run Phase 5: `plugin-bridge` install all
- [x] Run Phase 6: Selective cleanup & verify with `audit_stale_refs.py`
- [x] Run Phase 7: RLM Refresh & Tool Sync
- [x] Run Phase 9: Flash Distill all 45 scripts & sync `tool_inventory.json`
- [x] Cleanup: Remove mirrored `tools/` scripts (committed & pushed)
- [x] Verify: Cache queries return high-fidelity results
- [x] Git: Committed on `feat/tool-a-valuation-assistant`, pushed to origin
