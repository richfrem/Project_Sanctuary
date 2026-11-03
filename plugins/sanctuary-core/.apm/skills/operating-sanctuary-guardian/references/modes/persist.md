# Mode: Persist (`persist`)

Handles Soul Persistence (Protocol 128, Stage 5): export sealed knowledge and publish it to the Hugging Face dataset commons.
Credentials come only from the shell environment (`HUGGING_FACE_TOKEN`, `HUGGING_FACE_USERNAME`, `HUGGING_FACE_REPO`,
`HUGGING_FACE_DATASET_PATH`); never write them to disk or logs. All paths below are relative to the repository root.

## Execution Steps

1. Extract and verify the dataset records (markdown only; re-extraction refreshes every record's timestamp):
```bash
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/trace_extractor.py --source-dir chronicle/entries --append
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/trace_extractor.py --verify
```
2. Publish the dataset folder (`huggingface/`), dry run first:
```bash
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/sync_soul_dataset.py --dry-run
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/sync_soul_dataset.py --message "feat(soul): sync soul traces"
```
3. Optional, for a single sealed snapshot (and the RLM summary cache if present):
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/persist_soul.py --dry-run
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/persist_soul.py
```
`persist_soul.py --full-sync` only prints the commands of steps 1-2.
4. Verify by re-downloading the published files and comparing them with the repository copies.

For a single file (for example the dataset README) use the global `managing-huggingface-assets` skill:
`python3 ~/.claude/skills/managing-huggingface-assets/scripts/hf_upload.py --file huggingface/README.md --remote-path README.md`.

`forge_soul.py --vault-root <vault> [--full-sync] [--dry-run] [--output PATH]` exports records from an Obsidian vault; it is not part of the standard path.
