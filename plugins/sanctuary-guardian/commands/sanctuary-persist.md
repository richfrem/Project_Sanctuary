---
description: Protocol 128 Phase VII - Soul Persistence (Broadcast to Hugging Face)
---
# Workflow: Persist

> **CLI Command**: `python3 plugins/sanctuary-guardian/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md`
> **Output**: Uploads to HuggingFace `richfrem/Project_Sanctuary_Soul`

## Steps

1. **Broadcast Soul**:
   // turbo
   python3 plugins/sanctuary-guardian/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md

2. **Ingest Changes** (Optional - can also use `/sanctuary-ingest`):
   // turbo
   python3 tools/cli.py ingest --incremental --hours 24

