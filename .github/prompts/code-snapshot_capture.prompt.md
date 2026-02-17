---
description: Capture a full or subfolder code snapshot into a token-counted markdown artifact
argument-hint: "[subfolder] [--role guardian|auditor|coordinator|strategist] [--out dir] [--manifest path.json]"
---

# Capture Code Snapshot

Bundle repository files into a single LLM-distilled markdown snapshot with token counts.

## Usage
```bash
# Full project snapshot (generates genome + awakening seeds)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/capture_code_snapshot.py

# Specific subfolder only
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/capture_code_snapshot.py mcp_servers/rag_cortex

# Custom role
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/capture_code_snapshot.py --role auditor

# Custom output directory
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/capture_code_snapshot.py --out temp/snapshots

# From a manifest file (skip traversal)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/capture_code_snapshot.py --manifest path/to/manifest.json --output snapshot.txt
```

## Prerequisites
- `pip install tiktoken` (for token counting)

## Output
- `dataset_package/markdown_snapshot_full_genome_llm_distilled.txt` — Full snapshot
- `dataset_package/seed_of_ascendance_awakening_seed.txt` — Awakening seeds (full mode only)
- Per-role core essence files

## Modes
| Mode | Trigger | Awakening Seeds? |
|:---|:---|:---|
| **Full** | No subfolder arg | ✅ Yes |
| **Subfolder** | Pass directory name | ❌ No |
| **Manifest** | `--manifest` flag | ❌ No |
