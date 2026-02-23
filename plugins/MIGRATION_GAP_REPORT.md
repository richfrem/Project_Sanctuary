# Migration Gap Report

**Generated Date**: 2026-02-15
**Status**: Final Migration State

This report catalogues all items that remain outside the standardized `plugins/` architecture. These items fall into two categories:
1.  **Standalone Skills**: Skills active in `.agent/skills` but missing a corresponding `plugins/` source definition.
2.  **Unmapped Tools**: Files remaining in the `tools/` directory that were excluded from migration.

## 1. Standalone Skills (Zombies/Preserved)
These skills exist in `.agent/skills` (and are propagated to agents) but do NOT have a source dictionary in `plugins/*/skills`.
**Risk**: If deleted from `.agent/skills`, they are lost forever as their original `tools/` source is gone.

| Skill Name | Status | Recommendation |
|:---|:---|:---|
| `code-review` | Active | Create local plugin `plugins/code-review` |
| `doc-coauthoring` | Active | Create local plugin `plugins/doc-coauthoring` |
| `mcp-builder` | Active | Create local plugin `plugins/mcp-builder` |
| `memory-management` | Active | Create local plugin `plugins/memory-management` |
| `ollama-launch` | Active | Merge into `plugins/rlm-factory` or new `plugins/infrastructure` |
| `rlm-distill` | Active | Merge into `plugins/rlm-factory` |
| `skill-creator` | Active | Create local plugin `plugins/skill-creator` |
| `stock_valuation` | Active | Move to `apps/investment-screener/skills` |
| `thesis-balancer` | Active | Move to `apps/investment-screener/skills` |
| `vector-db-launch` | Active | Merge into `plugins/vector-db` |

## 2. Unmapped Tools (Legacy/Excluded)
These files remain in `tools/` because they were explicitly excluded from the mapping or identified as useful standalones.

### A. Investment Screener
**Location**: `tools/investment-screener/`
**Type**: Full Stack Application
**Status**: Preserved
**Recommendation**: Move to `apps/investment-screener` to separate it from "tools".

### B. AI Resources
**Location**: `tools/ai-resources/`
**Content**: Prompts, personas (`devops-incident-responder.md`, `unified_stock_analyst_prompt.md`).
**Recommendation**: Move to `plugins/ai-resources` or `plugins/personas`.





### E. Tracking
**Location**: `tools/codify/tracking/`
- `generate_todo_list.py`: Preserved todo generator.
**Recommendation**: Move to `plugins/task-manager/scripts/tracking/`.

## Summary Statistics
- **Migrated Plugins**: 16
- **Standalone Skills**: 10
Use this report to prioritize future refactoring work.
