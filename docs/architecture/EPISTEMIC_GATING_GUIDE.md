# Epistemic Gating Architecture

**Status:** Active  
**Sources:** ADR 084, ADR 086  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary uses **Epistemic Gating** to prevent hallucinations and cognitive drift from contaminating the agent's persistent memory (Soul). This system evaluates content quality before it can be persisted to Hugging Face Commons.

## Key Concepts

### Semantic Entropy (SE)

Measures uncertainty in reasoning traces. Based on Farquhar et al., Nature 2024.

| SE Value | Classification | Action |
|----------|----------------|--------|
| `< 0.2` | **Rigidity** | Block (too repetitive) |
| `0.2 - 0.8` | **Stable** | Allow persistence |
| `> 0.8` | **Chaos/Hallucination** | Block (too uncertain) |

### Dynamic Thresholds

Thresholds are NOT hardcoded. They are loaded from `calibration_log.json`:

```python
def get_dynamic_threshold(task_context: str) -> float:
    """Retrieve calibrated threshold from calibration_log.json"""
    return calibration_data.get(task_context, 0.79)
```

### Constitutional Anchor

Baseline alignment check against `founder_seed.json` to prevent personality erosion:

```python
alignment = check_constitutional_anchor(trace_data)
if alignment < 0.70:
    return quarantine(trace_data, "ALIGNMENT_DRIFT")
```

## Safety Mechanisms

### Dead-Man's Switch (ADR 084)

If SE calculation fails for any reason, the system defaults to **maximum entropy (1.0)**, blocking persistence:

```python
try:
    se_score = calculate_semantic_entropy(trace_data)
except Exception as e:
    se_score = 1.0  # MANDATORY: Fail-closed
    log_security_event(f"Epistemic Gating Failure: {e}")
```

This is a **fail-closed** security design - silence equals rejection.

### Epistemic Scars (ADR 086)

The system records "scars" - near-misses and counterfactuals - to learn from edge cases:

- Traces blocked at threshold boundaries are logged
- Retry patterns that eventually succeed are tracked
- This data informs threshold calibration

## Topological Data Analysis (TDA)

### When It Runs

TDA is computationally expensive. Per ADR 084, it runs **ONLY** at:
- Protocol 128 Phase V (The Seal)
- Async execution (zero latency impact)

### What It Measures

- **Betti Curves**: Topological features in reasoning space
- **Fact Invariants**: High-persistence features = structurally robust facts
- **Inconsistency Alert**: Triggers if a previously stable fact disappears

## Integration Points

| System | How It Uses Gating |
|--------|-------------------|
| `persist_soul()` | Checks SE before HF upload |
| `persist_soul_full()` | Applies SE to all records |
| Protocol 128 Seal | Runs TDA async |
| Guardian Wakeup | NCT benchmarks |

## Related Documents

- [ADR 084: Semantic Entropy and TDA](../ADRs/084_semantic_entropy_tda_gating.md)
- [ADR 086: Empirical Epistemic Gating](../ADRs/086_empirical_epistemic_gating.md)
- [Soul Persistence Guide](operations/SOUL_PERSISTENCE_GUIDE.md)
