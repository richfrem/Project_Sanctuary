# TASK: Analyze Code Duplication in Cortex Cluster

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Analyze and summarize code duplication between the deployment service (sanctuary_cortex) and the libraries (rag_cortex, forge_llm) to ensure a Single Source of Truth architecture.

## 2. Deliverables

1. Duplication Analysis Report
2. Cleanup Implementation Plan

## 3. Detailed Execution Plan (File-by-File)

The goal is to strictly enforce **mcp_servers/rag_cortex** as the Single Source of Truth. The **mcp_servers/gateway/clusters/sanctuary_cortex** directory should only contain deployment configurations.

### Strategy Example: `cache.py`
*   **Role:** Implements the Cached Augmented Generation (CAG) layer.
*   **Current State:** Exists in both Library (`rag_cortex`) and Cluster (`sanctuary_cortex`).
*   **Analysis:** `rag_cortex/opertions.py` imports it as a module. It is large and distinct, so it should **NOT** be merged into `operations.py`, but should live alongside it in the Library.
*   **Action:** 
    1. Verify `rag_cortex/cache.py` is the superior version (checked: yes, has `env_helper`).
    2. Archive `sanctuary_cortex/cache.py` -> `.bak`.
    3. Ensure `server.py` imports from `mcp_servers.rag_cortex.cache` if needed.

### File-by-File Cleanup Table

| File | Role | Library (`rag_cortex`) Action | Cluster (`sanctuary_cortex`) Action | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **cache.py** | Caching Layer | **KEEP**. Core module used by Operations under `get_cache()`. | **ARCHIVE** (`.bak`). | Ensure server imports from library if direct access needed. |
| **file_store.py** | File Management | **KEEP**. Handles hash storage. | **ARCHIVE** (`.bak`). | Used internally by Operations. |
| **genesis_queries.py** | Cache Warmup Data | **KEEP**. Static data. | **ARCHIVE** (`.bak`). | Data file. |
| **ingest_code_shim.py** | Ingestion Logic | **KEEP**. Complex logic. | **ARCHIVE** (`.bak`). | Helper module for Ingestion. |
| **mcp_client.py** | Ext. MCP Conn | **KEEP**. Connectivity layer. | **ARCHIVE** (`.bak`). | |
| **structured_query.py** | Query Parsing | **KEEP**. Parser logic. | **ARCHIVE** (`.bak`). | |
| **models.py** | Data Definitions | **KEEP**. The Menu (Types). | **ARCHIVED** (Done). | Validates SSOT. |
| **operations.py** | Core Logic | **KEEP**. The Chef (Logic). | **ARCHIVED** (Done). | Validates SSOT. |
| **validator.py** | Input Checks | **KEEP**. The Bouncer. | **ARCHIVED** (Done). | Validates SSOT. |

## 4. Next Steps
1.  [ ] **Archive Duplicates:** Rename all "Cluster Action: ARCHIVE" files to `.bak`.
2.  [ ] **Verify Imports:** Run `verify_imports.py` to ensure `server.py` doesn't break.
3.  [ ] **Clean Imports:** If `server.py` has `from .cache import` (relative), change to `from mcp_servers.rag_cortex.cache import`.
4.  [ ] **Rebuild:** Docker rebuild to prove the Cluster container relies solely on the Library copy.


