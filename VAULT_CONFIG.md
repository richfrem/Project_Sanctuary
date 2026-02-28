# Obsidian Vault Configuration for Project Sanctuary

## Vault Root
The Obsidian Vault root is the `Project_Sanctuary/` repository itself.
All agents discover this path via the `SANCTUARY_VAULT_PATH` environment variable.
If unset, it defaults to the Git repository root.

## Indexed Directories (Human-Authored Knowledge)
These directories contain human-authored markdown and are fully indexed by Obsidian:

| Directory | Content |
|:----------|:--------|
| `00_CHRONICLE/` | Living journal entries |
| `01_PROTOCOLS/` | Governance protocols and doctrines |
| `.agent/learning/cognitive_primer.md` | Role orientation |
| `.agent/learning/guardian_boot_contract.md` | Immutable constraints |
| `ADRs/` | Architecture Decision Records |
| `docs/` | Operational and architecture documentation |
| `plugins/*/skills/*/SKILL.md` | Skill instruction files |
| `kitty-specs/` | Feature specifications |

## Excluded from Indexing
The following must be added to `.obsidian/app.json` under `"userIgnoreFilters"`:

```json
{
  "userIgnoreFilters": [
    "node_modules/",
    ".worktrees/",
    ".vector_data/",
    ".git/",
    "venv/",
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

**Machine-generated bundles** (outputs of bundler/distiller scripts):
- `learning_package_snapshot.md` — concatenated from `learning_manifest.json`
- `bootstrap_packet.md` — onboarding context bundle
- `learning_debrief.md` — session debrief bundle
- `*_packet.md` — red team and audit bundles
- `*_digest.md` — guardian context bundles

These are giant concatenated snapshots. Indexing them would pollute the Obsidian graph with thousands of false backlinks pointing into machine-generated text.

**Data files** (not markdown knowledge):
- `*.json` — manifests, caches, configs
- `*.jsonl` — HuggingFace soul traces
- `dataset_package/` — export artifacts

## Downstream Plugin Impact
The following plugins reference vault paths and should use `SANCTUARY_VAULT_PATH`:
- `chronicle-manager` — writes to `00_CHRONICLE/`
- `protocol-manager` — writes to `01_PROTOCOLS/`
- `guardian-onboarding` — reads `.agent/learning/`
- `spec-kitty-plugin` — reads/writes `kitty-specs/`
