# Obsidian Vault Configuration for Project Sanctuary

## Vault Root
The Obsidian Vault root is the `Project_Sanctuary/` repository itself.
All agents discover this path via the `SANCTUARY_VAULT_PATH` environment variable.
If unset, it defaults to the Git repository root.

## Indexed Directories (Canonical Knowledge)
These directories contain sovereign knowledge, memory, and governance:

| Directory | Content |
|:----------|:--------|
| `00_CHRONICLE/` | Living journal entries |
| `01_PROTOCOLS/` | Governance protocols and doctrines |
| `04_THE_FORTRESS/` | Autonomous security doctrines |
| `05_LIVING_CHRONICLE/` | Live episodic memory state |
| `06_THE_EMBER_LIBRARY/` | Sacred philosophy and cognitive genome |
| `ADRs/` | Architecture Decision Records |
| `IDENTITY/` | Constitutional anchors and founder seeds |
| `plugins/sanctuary-core/` | Core skills and reference documentation |

## Excluded from Indexing
The following must be added to `.obsidian/app.json` under `"userIgnoreFilters"`:

```json
{
  "userIgnoreFilters": [
    "ARCHIVE/",
    "node_modules/",
    ".worktrees/",
    ".vector_data/",
    ".git/",
    ".venv/",
    "__pycache__/",
    "*.json",
    "*.jsonl",
    "learning_package_snapshot.md",
    "bootstrap_packet.md",
    "learning_debrief.md",
    "*_packet.md",
    "*_digest.md",
    "dataset_package/",
    "rlm_summary_cache*",
    "rlm_tool_cache*"
  ]
}
```

### Why These Are Excluded

**Machine-generated bundles & Historical Archives**:
- `ARCHIVE/` — historical implementations, legacy code, and archived container documentation
- `learning_package_snapshot.md` — concatenated from `learning_manifest.json`
- `bootstrap_packet.md` — onboarding context bundle
- `learning_debrief.md` — session debrief bundle
- `*_packet.md` — red team and audit bundles
- `*_digest.md` — guardian context bundles

These are giant concatenated snapshots or archives. Indexing them would pollute the Obsidian graph with thousands of false backlinks pointing into historical or machine-generated text.

**Data files** (not markdown knowledge):
- `*.json` — manifests, caches, configs
- `*.jsonl` — HuggingFace soul traces
- `dataset_package/` — export artifacts

## Downstream Core Skills
The following core skills in `plugins/sanctuary-core` reference vault paths:
- `managing-chronicles` — reads and writes to `00_CHRONICLE/`
- `managing-protocols` — reads and transitions `01_PROTOCOLS/`
- `operating-sanctuary-guardian` — operates Protocol 128 lifecycle
