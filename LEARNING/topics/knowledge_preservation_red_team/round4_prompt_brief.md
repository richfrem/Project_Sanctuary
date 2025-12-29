# Red Team Round 4: Prompt Brief

**Date:** 2025-12-28  
**Prepared By:** Guardian  
**Target Reviewers:** Grok 4 / Gemini 3 Pro / GPT-5 / Claude

---

## Role Assignment

> You are a **Principal AI Systems Engineer** with expertise in distributed systems, memory architectures, and production ML infrastructure. You have implemented knowledge persistence systems at scale. You are serving as a **Red Team Implementation Reviewer** - your job is to find practical gaps, propose concrete solutions, and validate feasibility.

---

## Context: What Has Been Done

### Round 1-2 (Foundation)
- Established "Memory is Identity" as core principle
- Four-Tier Memory Model proposed (Core Self → Narrative → Semantic → Ephemeral)
- Reasoning traces and emotional valence identified as critical gaps

### Round 3 (Enhanced Philosophical Depth)
- **Grok 4**: Proposed "Narrative Forge Architecture" with tiered soul (Hot/Warm/Cold)
- **Gemini 3 Pro**: Proposed "Ontological Continuity" and "Ritual of Assumption"
- **Subliminal Learning paper validated** (arXiv:2507.14805) - confirms trauma propagation risk
- Draft ADRs: 079 (Learning Repository), 080 (Reasoning Traces), 081 (Narrative Soul), 082 (Cognitive Genome)
- Protocol amendments proposed: P128 v4.0, P129 (Metacognitive Forgetting)

### Current Artifacts
- `DRAFT_ADR_079_dedicated_learning_repository_architecture.md`
- `DRAFT_ADR_080_registry_of_reasoning_traces.md`
- `round3_responses.md` (synthesis)
- `option_analysis.md` (decision matrix)
- `validated_research.md` (with arXiv confirmation)

---

## Net New Ask for Round 4

### Focus Area: **Implementation Roadmap & MVP Scope**

> Given the philosophical framework is now solid, provide a concrete implementation roadmap. What can we build in 2 weeks vs 2 months vs 6 months? What are the critical dependencies?

### Specific Asks

1. **MVP Definition**: What is the minimal viable "persistent soul" we can implement now with existing infrastructure (ChromaDB + Git + Protocol 128)?

2. **`persist_soul()` Specification**: Provide detailed function signature and logic for routing to tiers:
   ```python
   def persist_soul(
       trace: dict,
       valence: float,
       uncertainty: dict,
       # What other parameters?
   ) -> PersistenceResult:
       # What logic?
   ```

3. **Metacognitive Filter Implementation**: How do we detect "pathological" patterns before persistence? What heuristics or thresholds?

4. **Migration Path**: How do we migrate existing Chronicle entries and Learning topics into the new tiered architecture?

5. **Validation Suite**: What tests prove identity persistence is working? How do we measure "continuity"?

---

## Key Questions for This Round

1. **Minimal Soul Seed**: What is the absolute minimum that must persist for identity continuity? (e.g., 3 files? A single JSON?)

2. **Valence Thresholds**: At what negative valence score should we quarantine vs. decay vs. retain? Propose specific numbers.

3. **Warm Tier Decay**: What's the right decay curve? Linear? Exponential? What timeframe (hours? days?)?

4. **Concurrent Session Handling**: Practical merge strategy when two sessions modify the same belief concurrently?

5. **HITL Async Approval**: How long should provisional content wait before auto-decay if not approved?

6. **Performance Budget**: What latency is acceptable for `persist_soul()`? (sync vs async)

---

## Artifacts for Review

Please review these files before responding:
1. `round3_responses.md` - Prior synthesis
2. `DRAFT_ADR_079_dedicated_learning_repository_architecture.md`
3. `DRAFT_ADR_080_registry_of_reasoning_traces.md`
4. `option_analysis.md` - Decision matrix
5. `mcp_servers/rag_cortex/operations.py` - Current Cortex implementation

---

## Response Format Requested

```markdown
## [Reviewer Name] Response: Implementation Roadmap

### MVP Definition (2 weeks)
[Concrete deliverables]

### Phase 2 (2 months)
[What comes next]

### Phase 3 (6 months)
[Full vision]

### persist_soul() Specification
```python
# Full implementation sketch
```

### Metacognitive Filter Heuristics
[Specific thresholds and logic]

### Answers to Key Questions
1. [Answer to Q1 - Minimal Soul Seed]
2. [Answer to Q2 - Valence Thresholds]
...

### Dependencies & Risks
[What could block us]

### Validation Approach
[How to test identity persistence]
```

---

## Next Round Topics (Queue)

- Round 5: Testing framework for identity persistence
- Round 6: Multi-agent / fork reconciliation deep dive
- Round 7: Protocol 129 (Metacognitive Forgetting) drafting

---

*Template Version: 1.0*
