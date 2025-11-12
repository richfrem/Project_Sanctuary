# TASK: Harden Mnemonic Cortex Ingestion & RAG Pipeline

**Status:** IN-PROGRESS
**Priority:** Critical
**Lead:** GUARDIAN-01
**Related Documents:** `mnemonic_cortex/scripts/README.md`

## Objective
Diagnose and repair the catastrophic failure of the Mnemonic Cortex ingestion and query pipeline. Ensure the system is fully operational and its integrity is verifiable.

## Sub-Tasks
- [x] Diagnose root cause of ingestion failures (batch size, serialization, imports).
- [x] Reforge `ingest.py` with disciplined batch processing and correct serialization.
- [x] Reforge `vector_db_service.py` to align with the persistent, serialized architecture.
- [x] Forge `agentic_query.py` to enable end-to-end cognitive loop testing.
- [x] Forge `test_cognitive_layers.sh` verification harness.
- [ ] **Execute final verification and confirm all tests pass.**
