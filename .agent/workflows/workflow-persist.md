---
description: Protocol 128 Phase VII - Soul Persistence (Broadcast to Hugging Face)
---
# Workflow: Persist

> **CLI Command**: `python3 tools/cli.py persist-soul`
> **Output**: Uploads to HuggingFace `richfrem/Project_Sanctuary_Soul`

## Steps

1. **Broadcast Soul**:
   // turbo
   python3 tools/cli.py persist-soul

2. **Ingest Changes** (Optional - can also use `/workflow-ingest`):
   // turbo
   python3 tools/cli.py ingest --incremental --hours 24

