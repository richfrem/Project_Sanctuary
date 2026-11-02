# Memory Stack Architecture & Mappings

Maps Project Sanctuary's storage tiers to the sovereign memory architecture:

| Tier | Sanctuary Asset | Primary Capability / Skill |
|---|---|---|
| **Layer 1: Fast Boot** | `context/*` & `.agent/learning/guardian_boot_digest.md` | `operating-sanctuary-guardian` (wakeup bootloader) |
| **Layer 2: Compounding Knowledge** | `LEARNING/*` & `chronicle/ledger.json` | `managing-sovereign-memory` (progressive disclosure, graph queries) |
| **Layer 2: Ledger Summaries** | `.agent/learning/rlm_summary_cache.json` | `managing-rlm-cache` (global skill) |
| **Layer 3: External Soul Dataset** | `huggingface/data/soul_traces.jsonl` | `preserving-soul-traces` + `managing-huggingface-assets` |
| **Visualization (Human UI)** | `$OBSIDIAN_VAULT_PATH` (optional viewer) | `managing-obsidian-vaults` (optional) |

*Note: Vector databases (ChromaDB) have been discontinued. Agent reasoning relies entirely on zero-dependency, filesystem-native structured markdown, associative wikilinks, and sub-millisecond JSON index lookups.*
