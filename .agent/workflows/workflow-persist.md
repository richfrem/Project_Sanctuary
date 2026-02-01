---
description: Protocol 128 Phase VI - Soul Persistence (Broadcast to Hugging Face)
---
# Workflow: Persist

1. **Broadcast Soul**:
   // turbo
   python3 scripts/cortex_cli.py persist-soul

2. **Ingest Changes**:
   // turbo
   python3 scripts/cortex_cli.py ingest --incremental --hours 24
