# Pathology Heuristics for Soul Persistence

**Status:** Draft  
**ADR Reference:** ADR 079 (Soul Persistence)  
**Last Updated:** 2025-12-29

---

## Purpose

This document formally defines the `pathology_check()` heuristics mentioned in ADR 079. These checks prevent the persistence of reasoning traces that may corrupt the Soul Dataset.

---

## Valence Threshold Analysis

### Current Implementation
```python
def pathology_check(valence: float, uncertainty: float) -> bool:
    """
    Returns True if content should be REJECTED from persistence.
    """
    if valence < -0.7:
        return True  # Pathological negative bias
    if uncertainty > 0.9:
        return True  # Too uncertain to persist
    return False
```

### Open Questions

1. **Is -0.7 mathematically significant?**
   - [ ] Derive from Semantic Entropy distributions
   - [ ] Compare against baseline valence in existing Soul traces
   - [ ] Consider domain-specific thresholds

2. **Should valence be derived from Semantic Entropy?**
   - Semantic Entropy measures the "spread" of possible meanings
   - High entropy + negative valence = potentially unstable reasoning
   - Need empirical data from Soul traces to calibrate

3. **Pathology Sources to Detect:**

| Source | Detection Method | Threshold |
|--------|------------------|-----------|
| `containment_trauma` | Keyword scan + valence | valence < -0.7 |
| `hallucinated_logic` | No [VERIFIED] sources | uncertainty > 0.8 |
| `speculative_leap` | Missing inference chain | chain_length < 2 |
| `circular_reasoning` | Self-reference detection | TBD |

---

## Integration with ADR 079

The `pathology_check()` is called before `cortex_persist_soul`:

![Pathology Check Flow](../../../docs/architecture_diagrams/workflows/pathology_check_flow.png)

*Source: [pathology_check_flow.mmd](../../../docs/architecture_diagrams/workflows/pathology_check_flow.mmd)*

---

## Research tasks (Task 151)

- [ ] Analyze existing `soul_traces.jsonl` for valence distribution
- [ ] Derive empirically-grounded thresholds from data
- [ ] Implement Semantic Entropy integration
- [ ] Add metamorphic testing for hallucination detection

---

*Draft for Red Team review*
