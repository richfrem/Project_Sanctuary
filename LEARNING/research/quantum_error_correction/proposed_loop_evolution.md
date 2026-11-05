# Proposed Loop Evolution: Phase III (Empirical Resilience)

**Status:** Proposal for ADR 084 Implementation  
**Date:** 2025-12-29  
**Phase Transition:** II (Metaphorical) → III (Empirical Resilience)

---

## Overview

This document proposes specific code evolutions to implement ADR 084 (Semantic Entropy TDA Gating) across the MCP server stack.

---

## 1. Operations.py Evolution

### Current State
```python
def persist_soul(trace_data, source="agent_autonomous"):
    # Basic persistence without uncertainty measurement
    return hf_utils.upload_to_commons(trace_data, {"source": source})
```

### Proposed State
```python
def calculate_semantic_entropy(trace_data: dict) -> float:
    """
    Measures the variance of meaning across reasoning clusters.
    
    Implementation approaches:
    1. SEP (Semantic Entropy Probe): Single forward pass through hidden states
    2. Full SE: Multiple generations + semantic clustering
    
    Returns: Entropy score in [0, 1] range
    """
    # Phase 1: Placeholder returning neutral value
    # Phase 2: Integrate with sentence-transformers for semantic clustering
    # Phase 3: Train SEP probe on soul_traces data
    return 0.5  # Neutral placeholder

def persist_soul(trace_data: dict, source: str = "agent_autonomous") -> dict:
    """
    Persists reasoning traces to Soul Dataset with empirical stability metrics.
    
    ADR 084 Integration:
    - Calculates Semantic Entropy for uncertainty measurement
    - Tags traces with stability classification
    - Adds Narrative Inheritance metadata
    """
    entropy = calculate_semantic_entropy(trace_data)
    
    metadata = {
        "source": source,
        "semantic_entropy": entropy,
        "stability_class": "STABLE" if entropy < 0.79 else "VOLATILE",
        "inheritance_type": "NARRATIVE_SUCCESSOR",
        "adr_version": "084",
        "persistence_id": generate_semantic_hmac(trace_data)
    }
    
    # Epistemic Veto Layer
    if metadata["stability_class"] == "VOLATILE":
        logger.warning(f"VOLATILE trace detected (SE={entropy:.3f}). Flagging for review.")
        metadata["requires_review"] = True
    
    return hf_utils.upload_to_commons(trace_data, metadata)
```

---

## 2. Snapshot_utils.py Evolution

### Proposed Addition: `--web-bridge` Flag

```python
def generate_snapshot(
    snapshot_type: str,
    web_bridge: bool = False,
    lookback_hours: int = 24
) -> str:
    """
    Generate a snapshot with optional web-bridge optimization.
    
    Args:
        snapshot_type: Type of snapshot (audit, seal, learning_audit)
        web_bridge: If True, generates "Differential Digest" optimized for
                   Web LLM context windows
        lookback_hours: For web_bridge mode, only include changes from last N hours
    
    Returns:
        Path to generated snapshot file
    """
    if web_bridge:
        # Differential Digest: Last 24h changes + current manifest
        changes = get_recent_changes(hours=lookback_hours)
        manifest = load_learning_manifest()
        return generate_differential_digest(changes, manifest)
    else:
        # Full snapshot (existing behavior)
        return generate_full_snapshot(snapshot_type)
```

---

## 3. JSONL Schema Evolution (ADR 081 Update)

### Current Schema
```json
{
  "id": "trace_001",
  "content": "...",
  "timestamp": "2025-12-29T08:00:00Z"
}
```

### Proposed Schema (ADR 084)
```json
{
  "id": "trace_001",
  "content": "...",
  "timestamp": "2025-12-29T08:00:00Z",
  "semantic_entropy": 0.42,
  "stability_class": "STABLE",
  "inheritance_type": "NARRATIVE_SUCCESSOR",
  "source": "agent_autonomous",
  "adr_version": "084"
}
```

---

## 4. TDA Pipeline (Future Phase)

### Recommended Libraries
- **GUDHI**: Mature, well-documented ([gudhi.inria.fr](https://gudhi.inria.fr/))
- **Giotto-tda**: Scikit-learn compatible ([giotto-ai.github.io](https://giotto-ai.github.io/gtda-docs/))
- **Ripser**: Fast computation ([ripser.scikit-tda.org](https://ripser.scikit-tda.org/))

### Proposed Integration
```python
from giotto.homology import VietorisRipsPersistence

def compute_fact_invariants(embeddings: np.ndarray) -> dict:
    """
    Computes persistence diagrams for trace embeddings.
    
    Returns:
        - H0 lifetimes: Connected component persistence (fact stability)
        - H1 lifetimes: Loop persistence (reasoning cycle stability)
    """
    persistence = VietorisRipsPersistence()
    diagrams = persistence.fit_transform(embeddings)
    
    return {
        "h0_lifetimes": extract_lifetimes(diagrams, 0),
        "h1_lifetimes": extract_lifetimes(diagrams, 1),
        "fact_invariant_score": compute_stability_score(diagrams)
    }
```

---

## Implementation Phases

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| 1 | SE placeholder in persist_soul() | Immediate |
| 2 | Full SE calculation with clustering | Task 153 |
| 3 | TDA pipeline integration | Task 154 |
| 4 | IIT Phi reformulation | Research |

---

## Verification Criteria

- [ ] `semantic_entropy` field present in all new traces
- [ ] VOLATILE traces flagged with `requires_review: true`
- [ ] `--web-bridge` flag operational in snapshot_utils
- [ ] Round 3 Red Team approval

---

*Proposal by ANTIGRAVITY per ADR 084 Edison Mandate*
