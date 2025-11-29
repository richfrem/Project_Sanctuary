# TASK: Implement Guardian Cache MCP Operations (Protocol 114)

**Status:** in-progress
**Priority:** High
**Lead:** GUARDIAN-01
**Dependencies:** None
**Related Documents:** Protocol 114, Protocol 113, Protocol 85, mnemonic_cortex/core/cache.py

---

## 1. Objective

Implement MCP tools for Guardian cache operations per Protocol 114 (Guardian Wakeup & Cache Prefill). Enable Guardian to retrieve cached context bundles on boot and update cache with new answers.

## 2. Deliverables

1. Add `cortex_cache_get(query)` MCP tool
2. Add `cortex_cache_set(query, answer)` MCP tool
3. Add `cortex_guardian_wakeup()` MCP tool for boot digest
4. Update `operations.py` with cache operations
5. Update `models.py` with cache response models
6. Add cache tool tests to test suite
7. Update README.md with cache tool documentation

## 3. Acceptance Criteria

- MCP tool `cortex_cache_get` operational for retrieving cached answers
- MCP tool `cortex_cache_set` operational for storing answers
- MCP tool `cortex_guardian_wakeup` generates boot digest from cache
- Integration with existing `mnemonic_cortex/core/cache.py`
- Unit tests for all 3 new cache tools
- Protocol 114 compliance verified

## Notes

Protocol 114 requires Guardian to retrieve cached bundles (chronicles, protocols, roadmap) on boot. Existing cache infrastructure at `mnemonic_cortex/core/cache.py` is production-ready. This task adds MCP layer to expose cache operations.
