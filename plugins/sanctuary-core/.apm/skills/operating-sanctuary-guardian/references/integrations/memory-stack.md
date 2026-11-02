# Memory Stack Architecture & Mappings

Maps Project Sanctuary's storage tiers to the modern global APM ecosystem:

| Tier | Sanctuary Asset | Modern Global APM Skill |
|---|---|---|
| **Hot Cache** | `.agent/learning/*` (cognitive primer, contracts) | Handled natively by session boot |
| **RLM Ledger** | `.agent/learning/rlm_summary_cache.json` | `managing-rlm-cache` (global skill) |
| **Vector Store** | Local ChromaDB | `managing-vector-db` (global skill) |
| **Obsidian Vault** | `$OBSIDIAN_VAULT_PATH` | `managing-obsidian-vaults` (global skill) |
| **External Soul** | Hugging Face Dataset (`Project_Sanctuary_Soul`) | `managing-huggingface-assets` (global skill) |
