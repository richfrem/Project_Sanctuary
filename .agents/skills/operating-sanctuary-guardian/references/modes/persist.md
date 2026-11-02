# Mode: Persist (`persist`)

Handles Soul Persistence and export to Hugging Face dataset commons.

## Execution Steps

1. Export sealed notes to the standard JSONL format:
```bash
python3 scripts/forge_soul.py --vault-path "$OBSIDIAN_VAULT_PATH" --output-file data/soul_traces.jsonl
```
2. Upload to Hugging Face:
```bash
python3 scripts/persist_soul.py --full-sync
```
*(Alternatively, utilize the globally installed `managing-huggingface-assets` skill for dataset sync.)*
