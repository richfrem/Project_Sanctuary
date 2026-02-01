# CLI Gap Analysis: cortex_cli.py vs tools/cli.py

**Objective**: Audit functionality in `scripts/cortex_cli.py` (Legacy) against `tools/cli.py` (Target) to ensure full parity without duplicating business logic code.

**Strategy**: `tools/cli.py` should import business logic from `mcp_servers` directly (preserving existing location) but expose all necessary CLI commands.

## 1. Command Inventory

| Command | Subcommands / Args | `cortex_cli.py` | `tools/cli.py` | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ingest** | `--incremental`, `--hours` | ✅ | ✅ | **Parity** | Logic shared via `CortexOperations`. |
| | `--no-purge` | ✅ | ✅ | **Parity** | |
| | `--dirs` | ✅ | ✅ | **Parity** | |
| **snapshot** | `--type` | ✅ | ✅ | **Parity** | `tools/cli` adds `bootstrap` type. |
| | `--manifest` | ✅ | ✅ | **Parity** | |
| | `--context` | ✅ | ✅ | **Parity** | |
| | `--override-iron-core` | ✅ | ✅ | **Parity** | Security protocol implemented in both. |
| **persist-soul** | `--snapshot` | ✅ | ✅ | **Parity** | |
| | `--valence`, `--uncertainty` | ✅ | ✅ | **Parity** | |
| | `--full-sync` | ✅ | ✅ | **Parity** | |
| **persist-soul-full**| *(no args)* | ✅ | ✅ | **Parity** | |
| **guardian** | `wakeup` (mode) | ✅ | ✅ | **Parity** | |
| | `snapshot` | ❌ | ✅ | **Enhancement** | `tools/cli.py` has explicit `guardian snapshot`. |
| | `--manifest` | ✅ | ❌ | **Gap** | `tools/cli.py` `guardian wakeup` does not seem to expose `--manifest` arg explicitly? |
| **debrief** | `--hours` | ✅ | ✅ | **Parity** | |
| | `--output` | ✅ | ❌ | **Gap** | `tools/cli.py` prints to stdout, does not accept `--output`. |
| **bootstrap-debrief**| `--manifest`, `--output` | ✅ | ❌ | **Gap** | Missing in `tools/cli.py`. |
| **stats** | `--samples` | ✅ | ✅ | **Parity** | |
| **query** | `--max-results` | ✅ | ✅ | **Parity** | |
| **cache-stats** | *(no args)* | ✅ | ✅ | **Parity** | |
| **cache-warmup** | `--queries` | ✅ | ✅ | **Parity** | |
| **evolution** | `fitness`, `depth`, `scope` | ✅ | ✅ | **Parity** | |
| **rlm-distill** | `target` | ✅ | ✅ | **Parity** | |
| **dream** | *(disabled)* | ❌ | ❌ | **Ignore** | Commented out in legacy. |

## 2. Shared Utilities Audit

| Utility | Source | Status | Notes |
| :--- | :--- | :--- | :--- |
| `verify_iron_core` | `cortex_cli.py` (lines 78-127) | **Replicated** | Re-implemented in `tools/cli.py` (lines 158-201). **Action**: Consider moving to `tools/utils` or `mcp_servers.lib` to DRY. |
| `find_project_root`| `mcp_servers.lib.path_utils` | **Implicit** | `tools/cli.py` uses `resolve_path` and `sys.path` hacks. |

## 3. Action Plan

### A. Missing Commands to Add
1.  **`debrief`**: Add `--output` argument support.
2.  **`bootstrap-debrief`**: Implement as a top-level command in `tools/cli.py`. (Or clearly alias it).
3.  **`guardian`**: Check `--manifest` support for wakeup.

### B. Refactoring (Standardization)
1.  **Imports**: Ensure `tools/cli.py` imports `LearningOperations`, `CortexOperations`, `EvolutionOperations` from `mcp_servers.*` consistently. (Current imports look correct).
2.  **Iron Core**: `verify_iron_core` is duplicated. Move to `mcp_servers.lib.security` or `tools.lib.security`? *Decision: Keep duplication for now to minimize refactor scope, or consolidate if easy.*

### C. Deprecation
-   Once `tools/cli.py` has full parity, `scripts/cortex_cli.py` can be deprecated.
