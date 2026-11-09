# Mode: Persist (`persist`)

Handles Soul Persistence (Protocol 128, Stage 5): export sealed knowledge and publish it to the Hugging Face dataset commons.
Credentials come only from the shell environment (`HUGGING_FACE_TOKEN`, `HUGGING_FACE_USERNAME`, `HUGGING_FACE_REPO`,
`HUGGING_FACE_DATASET_PATH`); never write them to disk or logs. All paths below are relative to the repository root.

## Execution Steps

1. Export through the coverage policy and report the four facts (schema, coverage, local export, remote). Run it AFTER the mission's changes are merged to `main`, so artifact references point at published commits:
```bash
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/soul_coverage.py closure
```
The old Chronicle-only command (`trace_extractor.py --source-dir chronicle/entries --append`) is NOT a closure: it skips mission manifests, experiment findings, the event ledger and supporting-artifact references. See `preserving-soul-traces/references/coverage.md`.
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
4. `sync_soul_dataset.py` verifies the Hub revision itself after the upload (dataset and coverage-manifest hashes) and appends a receipt to `.agent/learning/soul_sync_receipts.jsonl`; commit that receipt. Only `SOUL PERSISTED` from `soul_coverage.py check` means all four facts hold. Never report "persisted" on local export alone.

For a single file (for example the dataset README) use the global `managing-huggingface-assets` skill:
`python3 ~/.claude/skills/managing-huggingface-assets/scripts/hf_upload.py --file huggingface/README.md --remote-path README.md`.

`forge_soul.py --vault-root <vault> [--full-sync] [--dry-run] [--output PATH]` exports records from an Obsidian vault; it is not part of the standard path.

Completion wiring: `record-result` durably marks the final manifest for preservation; after publishing sources to `origin/main`, `merge_prep.py regenerate` runs incremental local extraction, schema and coverage checks. Startup reports preservation status without changing preservation files. Capture and recovery occur at mission/session closure (`soul_coverage.py closure`) or an explicitly requested recovery action. Pending/failed/local-complete status is separate from the scientific verdict. Commit the export separately, then sync remotely only with authorization. See the coverage contract (`coverage.md` in the `preserving-soul-traces` skill).
