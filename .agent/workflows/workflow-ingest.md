---
description: Run RAG Ingestion (Protocol 128 Phase IX)
---
# Workflow: Ingest

1. **Ingest Changes**:
   // turbo
   python3 scripts/cortex_cli.py ingest --incremental --hours 24
