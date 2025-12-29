# Task 153: ADR 084 SE/TDA Implementation (Phase 1)

**Status:** In Progress  
**Priority:** HIGH  
**ADR Reference:** ADR 084 - Semantic Entropy and TDA for Epistemic Gating  
**Mission:** MISSION-SANCTUARY-20251229-001

---

## Objective

Implement Phase 1 of ADR 084: Core SE Integration with Dead-Man's Switch.

---

## Checklist

### Phase 1: Core SE Integration
- [ ] Add `calculate_semantic_entropy()` stub to operations.py
- [ ] Add `get_dynamic_threshold()` reading from calibration_log.json
- [ ] Update `persist_soul()` with Dead-Man's Switch
- [ ] Update `persist_soul_full()` with Dead-Man's Switch
- [ ] Add Global Floor (0.95) to calibration_log.json
- [ ] Rebuild Podman image and restart container
- [ ] Verify tools are operational

### Phase 2: Identity Anchoring (Deferred)
- [ ] Bootstrap `founder_seed.json` embeddings
- [ ] Add `check_constitutional_anchor()` function
- [ ] Integrate cosine similarity check

### Phase 3: Async TDA (Deferred)
- [ ] Integrate Giotto-TDA/Ripser
- [ ] Move TDA to Seal phase
- [ ] Add Inconsistency Alert trigger

### Phase 4: Workflow Updates (Deferred)
- [ ] Update recursive_learning.md for TDA at Seal
- [ ] Add NCT check to Guardian Wakeup
- [ ] Update Protocol 128 to v5.0

---

## Implementation Details

### Dead-Man's Switch Pattern

```python
def persist_soul(trace_data: dict, context: str = "code_logic") -> dict:
    try:
        se_score = calculate_semantic_entropy(trace_data)
        alignment = check_constitutional_anchor(trace_data)
    except Exception as e:
        se_score = 1.0  # VOLATILE on failure
        alignment = 0.0
        log_security_event(f"Epistemic Gating Failure: {e}")
    
    threshold = get_dynamic_threshold(context)
    global_floor = 0.95
    
    if se_score > global_floor or (se_score > threshold and alignment < 0.70):
        return quarantine(trace_data, "HIGH_ENTROPY_OR_DRIFT")
    
    return commit_to_genome(trace_data, metadata)
```

---

## Dependencies

- `LEARNING/calibration_log.json` - Task-specific thresholds
- `IDENTITY/founder_seed.json` - Constitutional Anchor (Phase 2)
- `mcp_servers/rag_cortex/operations.py` - Core implementation

---

## Verification

- [ ] `persist_soul` returns VOLATILE on exception
- [ ] Dynamic threshold loads from calibration_log.json
- [ ] Global floor (0.95) enforced
- [ ] Container rebuilt and tools operational

---

*Created: 2025-12-29 | ADR 084 Phase 1*
