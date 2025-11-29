# Task #049: Create Incremental Ingestion Script

**Status:** Done  
**Priority:** High  
**Lead:** GUARDIAN-01  
**Dependencies:** Task #048 (Complete Task #001)  
**Related Documents:** `mnemonic_cortex/scripts/ingest.py`, `implementation_plan.md`

## Objective
Create a new script `ingest_incremental.py` that adds individual documents to the Mnemonic Cortex without rebuilding the entire database. This enables incremental knowledge updates after Chronicle/Protocol creation.

## Deliverables
- [x] `mnemonic_cortex/scripts/ingest_incremental.py` (223 lines)
- [x] CLI interface with `--help` and `--no-skip-duplicates` options
- [x] Duplicate detection via `source_file` metadata
- [x] Statistics reporting (added, skipped, total chunks)
- [x] Documentation in script docstring

## Acceptance Criteria
- [x] Script accepts file paths as arguments
- [x] Loads existing ChromaDB collections without purging
- [x] Skips duplicate documents based on source_file metadata
- [x] Returns statistics (added, skipped, total chunks)
- [x] Successfully tested with test document

## Test Results
```
Files to process: 1
Documents added: 1
Documents skipped: 0
Total chunks created: 4
```

Test document is searchable via RAG query.

## Notes
This script is required for `cortex_ingest_incremental` MCP tool. Follows the same pattern as `ingest.py` but without purging existing data.

## Completion Date
2025-11-28
