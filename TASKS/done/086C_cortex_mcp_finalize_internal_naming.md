# Task 086C: Cortex MCP - Finalize Internal Naming Consistency

**Status:** TODO  
**Priority:** MEDIUM 🟡  
**Created:** 2025-12-01  
**Estimated Effort:** 1 hour

## Objective

Update all internal references in Cortex MCP from legacy names (`mnemonic_cortex`, `RAG DB`) to the canonical `Cortex MCP` terminology.

## Background

During the migration from `mnemonic_cortex/` to `mcp_servers.rag_cortex/`, some internal references were not updated. These need to be cleaned up for consistency and maintainability.

## Issues Found

### 1. Path References to Legacy Directory

**File:** `mcp_servers.rag_cortex/cache.py`
- Line 48-50: References `mnemonic_cortex/cache` directory

**File:** `mcp_servers.rag_cortex/utils.py`
- Line 10, 36-37: References `mnemonic_cortex` directory for .env file

**File:** `mcp_servers.rag_cortex/operations.py`
- Line 37: References `mnemonic_cortex/scripts` directory
- Line 126: Excludes `mnemonic_cortex` from ingestion
- Line 391: References `mnemonic_cortex/` db path

**Decision Needed:** ⚠️
- Are these paths intentional (pointing to legacy directory for backward compatibility)?
- Or should they point to new `mcp_servers.rag_cortex/` structure?

**Action:** Verify if `mnemonic_cortex/` directory still exists and contains active data

### 2. Import References

**File:** `mcp_servers.rag_cortex/server.py`
- Line 329: Imports from `mnemonic_cortex.app.synthesis.generator`

**File:** `mcp_servers.rag_cortex/operations.py`
- Line 279: Imports from `mnemonic_cortex.app.services.vector_db_service`
- Line 291: Imports from `mnemonic_cortex.app.services.llm_service`

**Issue:** These imports reference the legacy module structure

**Action:** 
- Verify if these modules still exist in `mnemonic_cortex/`
- If yes, consider migrating to `mcp_servers.rag_cortex/`
- If no, remove or update imports

### 3. Comment and Docstring References ✅ PARTIALLY FIXED

**File:** `mcp_servers.rag_cortex/operations.py`
- ~~Line 224: Comment "Uses: mnemonic_cortex RAG infrastructure directly"~~ ✅ FIXED
  - Updated to: "Uses: Cortex MCP RAG infrastructure directly"

### 4. Scope String References

**File:** `mcp_servers.rag_cortex/operations.py`
- Line 909: Scope string `mnemonic_cortex:index`

**Issue:** Internal scope identifier still uses legacy name

**Action:** Update to `cortex:index` or `cortex_mcp:index`

### 5. RAG DB References

**Status:** ✅ NO ISSUES FOUND

Searched for "RAG DB" in all Cortex MCP files - no results found.

## Tasks

### Phase 1: Investigation
- [ ] Verify if `mnemonic_cortex/` directory exists
- [ ] Check if it contains active cache/db/scripts
- [ ] Determine if legacy paths are intentional

### Phase 2: Path Updates (if needed)
- [ ] Update `cache.py` cache directory path
- [ ] Update `utils.py` .env file path
- [ ] Update `operations.py` scripts directory path
- [ ] Update `operations.py` db path
- [ ] Update exclusion list in ingestion

### Phase 3: Import Updates
- [ ] Verify `mnemonic_cortex.app.synthesis.generator` status
- [ ] Verify `mnemonic_cortex.app.services.*` status
- [ ] Migrate or remove legacy imports
- [ ] Update to use new module structure

### Phase 4: String/Scope Updates
- [x] Update comment in operations.py (line 224) ✅ DONE
- [ ] Update scope string (line 909)
- [ ] Search for any remaining `mnemonic_cortex` strings
- [ ] Update to `cortex` or `cortex_mcp`

### Phase 5: Testing
- [ ] Run Cortex MCP test suite
- [ ] Verify cache operations work
- [ ] Verify ingestion works
- [ ] Verify query operations work

## Recommended Changes

### cache.py
```python
# BEFORE
cache_dir = os.path.join(project_root, 'mnemonic_cortex', 'cache')

# AFTER (Option A - New structure)
cache_dir = os.path.join(project_root, 'mcp_servers', 'cognitive', 'cortex', 'cache')

# AFTER (Option B - Keep legacy for backward compatibility)
# Keep as-is if mnemonic_cortex/cache contains active data
```

### utils.py
```python
# BEFORE
dotenv_path = os.path.join(project_root, 'mnemonic_cortex', '.env')

# AFTER
dotenv_path = os.path.join(project_root, 'mcp_servers', 'cognitive', 'cortex', '.env')
```

### operations.py (scripts path)
```python
# BEFORE
self.scripts_dir = self.project_root / "mnemonic_cortex" / "scripts"

# AFTER
self.scripts_dir = self.project_root / "mcp_servers" / "cognitive" / "cortex" / "scripts"
```

### operations.py (scope string)
```python
# BEFORE
scope = query_data.get("scope", "mnemonic_cortex:index")

# AFTER
scope = query_data.get("scope", "cortex:index")
```

## Success Criteria

- [ ] All `mnemonic_cortex` path references updated or verified intentional
- [ ] All `mnemonic_cortex` imports updated or removed
- [ ] All `mnemonic_cortex` string references updated
- [ ] No "RAG DB" references found (already verified ✅)
- [ ] All Cortex MCP tests pass
- [ ] Cache, ingestion, and query operations work correctly

## Dependencies

- ✅ Cortex MCP implementation
- ⏭️ Verification of legacy directory status
- ⏭️ Decision on backward compatibility strategy

## Related Tasks

- Task #086: Post-Migration Validation (IN PROGRESS)
- Task #086A: Integration Test Refactoring (IN PROGRESS)
- Task #086B: Multi-Round Deliberation Verification (TODO)

## Notes

### Backward Compatibility Considerations

If `mnemonic_cortex/` directory still exists with active data:
- **Option 1:** Keep legacy paths for backward compatibility
- **Option 2:** Migrate data to new structure, update all paths
- **Option 3:** Support both paths with fallback logic

**Recommendation:** Verify directory status first, then decide strategy

### Import Migration Strategy

If legacy `mnemonic_cortex.app.*` modules still exist:
- **Option 1:** Keep imports, mark as deprecated
- **Option 2:** Copy modules to new structure, update imports
- **Option 3:** Refactor to remove dependencies

**Recommendation:** Check if modules are still needed, then migrate or remove

## Next Steps

1. Check if `mnemonic_cortex/` directory exists
2. Determine backward compatibility strategy
3. Update paths and imports accordingly
4. Run full test suite
5. Mark task as DONE
