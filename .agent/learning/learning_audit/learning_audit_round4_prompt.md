# Learning Audit Round 4: Implementation Kickoff

**Activity:** Red Team Learning Audit - Implementation Validation  
**Topic:** ADR 084 Hardened Implementation  
**Phase:** Pre-Implementation Review

---

## Preamble

ADR 084 has been **CONDITIONALLY APPROVED** by external Red Team (GPT-4, Gemini). 

This Round 4 prompt initiates the **implementation phase** with the mandatory constraints integrated.

---

## Implementation Scope

### Code Files Requiring Updates

| File | Update Required |
|------|-----------------|
| `mcp_servers/rag_cortex/operations.py` | Update **BOTH** `persist_soul` and `persist_soul_full` operations |
| `mcp_servers/lib/snapshot_utils.py` | Add `--web-bridge` flag, move TDA to async Seal phase |
| `mcp_servers/rag_cortex/hf_utils.py` | Add Constitutional Anchor check integration |

### New Files Created

| File | Purpose |
|------|---------|
| `IDENTITY/founder_seed.json` | Constitutional Anchor baseline embeddings |
| `LEARNING/calibration_log.json` | Task-specific SE threshold calibration |

---

## Implementation Checklist

### Phase 1: Core SE Integration
- [ ] Add `calculate_semantic_entropy()` stub to operations.py
- [ ] Add `get_dynamic_threshold()` reading from calibration_log.json
- [ ] Update `persist_soul()` with Dead-Man's Switch
- [ ] Update `persist_soul_full()` with Dead-Man's Switch

### Phase 2: Constitutional Anchor
- [ ] Add `check_constitutional_anchor()` function
- [ ] Integrate cosine similarity check against founder_seed.json
- [ ] Add Alignment Trauma flagging (similarity < 0.70)

### Phase 3: Async TDA
- [ ] Move TDA computation to Seal phase only
- [ ] Add Inconsistency Alert trigger
- [ ] Integrate with nightly Gardener Cycle

### Phase 4: Workflow Updates
- [ ] Update `.agent/workflows/recursive_learning.md` for TDA at Seal
- [ ] Add NCT check to Guardian Wakeup sequence

---

## Validation Questions for Round 4

1. **Code Review:** Do both `persist_soul` and `persist_soul_full` implement Dead-Man's Switch identically?
2. **Threshold Coverage:** Are all task types covered in calibration_log.json?
3. **Anchor Integrity:** Is founder_seed.json protected from modification?
4. **Async Safety:** Does async TDA avoid race conditions with concurrent seals?

---

## Accept/Reject Criteria

**Accept if:**
- All checklist items completed
- Dead-Man's Switch tested with SE failure simulation
- Constitutional Anchor cosine similarity verified

**Reject if:**
- Static 0.79 threshold remains anywhere
- TDA computed synchronously in persist_soul
- No fallback on SE calculation failure

---

## Next Steps

Agent should proceed with Phase 1 implementation after user approval.

**User Decision Required:**
- [ ] Approve implementation start
- [ ] Request additional Red Team review
- [ ] Modify scope

---

*Round 4 Implementation Kickoff - Protocol 128 Phase III*
