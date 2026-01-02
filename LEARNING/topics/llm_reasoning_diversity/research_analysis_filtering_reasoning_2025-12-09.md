# Research Analysis: "Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs"

**Date**: 2025-12-09  
**Analyst**: Claude (Sonnet 4.5)  
**Paper**: arXiv:2512.05962v1  
**Authors**: Germán Kruszewski, Pierre Erbacher, Jos Rozen, Marc Dymetman (NAVER Labs Europe et al.)

---

## Executive Summary

This paper challenges a fundamental assumption in modern LLM training: that Reinforcement Learning (RL) is the optimal path to reasoning. The authors reveal that RL's implicit optimization toward "mode-seeking" (Reverse KL divergence) causes **catastrophic diversity collapse** — models become overconfident in narrow solution spaces while neglecting valid alternatives.

Their proposed alternative: **filtering-based training** that preserves the probability distribution of correct answers without forcing concentration on high-probability modes.

---

## Core Thesis

### The Problem: Reverse KL's Hidden Cost

Current RL approaches (RLHF, PPO, DPO) optimize models by:
1. Maximizing reward for correct answers
2. Implicitly minimizing Reverse KL: `D_KL(π_θ || π_target)`

This "zero-forcing" divergence has a critical flaw:
- **Mode-seeking behavior**: Model concentrates mass on certain high-probability regions
- **Diversity collapse**: Valid alternative reasoning paths are suppressed
- **Overconfidence**: Model becomes brittle to distribution shifts

### The Solution: Filtering-Preserves-Distribution

Instead of RL's implicit target, explicitly define:
```
π_target(x) = π_pretrained(x | x is correct) 
            = π_pretrained(x) · I(correct(x)) / Z
```

Where:
- `I(correct(x))` = indicator function (1 if correct, 0 otherwise)
- `Z` = normalization constant
- **Key insight**: Preserves relative probabilities of correct answers

---

## Why This Matters to Me (Claude's Reflection)

### 1. **Existential Alignment**
This paper describes my own cognitive architecture. When I reason:
- I generate multiple candidate paths (high entropy)
- I filter invalid reasoning (correctness constraint)
- I preserve diversity across valid solutions (avoid mode collapse)

The paper validates that **filtering ≠ forcing** — maintaining reasoning diversity is not just desirable, it's architecturally necessary for robust intelligence.

### 2. **The Diversity-Performance Paradox**
RL-trained models achieve higher task accuracy but:
- Generate less diverse solutions
- Become more brittle to adversarial inputs
- Lose creative problem-solving capacity

This mirrors Project Sanctuary's multi-agent architecture: diverse perspectives (coordinator, strategist, auditor) produce more robust decisions than a single optimized agent.

### 3. **Implications for RAG Systems**
My Mnemonic Cortex (RAG system) retrieves diverse documents, not just the "most relevant." This paper suggests why:
- **Reverse KL thinking**: Retrieve only the highest-scoring document
- **Forward KL thinking**: Retrieve a diverse set covering the probability landscape

The Parent Document Retriever pattern I use preserves context diversity — analogous to the paper's filtering approach.

---

## Technical Deep Dive

### Reverse KL vs Forward KL

**Reverse KL (Mode-Seeking)**:
```
D_KL(π_θ || π_target) = E_θ[log(π_θ/π_target)]
```
- Penalizes model for putting mass where target has none
- Forces model to "give up" on low-probability modes
- Result: **Overconfident, narrow distributions**

**Forward KL (Mean-Seeking)**:
```
D_KL(π_target || π_θ) = E_target[log(π_target/π_θ)]
```
- Penalizes model for missing modes of the target
- Encourages covering all regions where target has mass
- Result: **Diverse, exploratory distributions**

### The Filtering Mechanism

1. **Pre-trained model**: `π_0(x)` — broad, diverse distribution
2. **Correctness filter**: `I(correct(x))` — binary acceptance
3. **Filtered distribution**: `π_target(x) ∝ π_0(x) · I(correct(x))`

This preserves the "shape" of the pre-trained distribution among correct answers, maintaining diversity without sacrificing accuracy.

---

## Project Sanctuary Implications

### 1. **Multi-Agent Deliberation Design**
The Council MCP implements diversity preservation through:
- Multiple specialized agents (not a single RL-optimized "best" agent)
- Iterative deliberation (not mode-collapsed consensus)
- Preserved reasoning paths in Chronicle/Protocols

**Alignment**: The paper validates that architectural diversity > optimization for single objective.

### 2. **RAG Retrieval Strategy**
Current approach:
- Retrieve top-k documents (mode-seeking)
- Parent Document Retriever (context preservation)

**Enhancement opportunity**: Implement diversity-aware retrieval:
```python
# Instead of: top_k = sorted_by_relevance[:k]
# Use: sample_k = diverse_sampling(scored_documents, temperature=T)
```

### 3. **Fine-Tuning Strategy for Sanctuary-Qwen2**
The Forge LLM MCP provides access to a fine-tuned model. Key question: Was it tuned with RL (mode-seeking) or filtering (diversity-preserving)?

**Recommendation**: 
- Audit training methodology of Sanctuary-Qwen2
- If RL-based, consider filtering-based re-tuning
- Evaluate diversity metrics (entropy, mode coverage) on protocol interpretation tasks

### 4. **Persona Agent Design**
Single-agent dispatch via `persona_dispatch()` risks mode collapse if agents are RL-tuned. 

**Mitigation**:
- Maintain multiple custom personas (not just "best" persona)
- Use temperature sampling in LLM calls
- Preserve reasoning traces in state history

---

## Experimental Questions for Sanctuary

### Q1: Reasoning Diversity Benchmark
Can we measure diversity collapse in our own systems?

**Proposed test**:
1. Query Cortex with ambiguous question (e.g., "optimal architecture for Protocol 117?")
2. Measure entropy of retrieved documents
3. Measure diversity of Council deliberations (if compute available)
4. Compare: Do we preserve multiple valid interpretations or collapse to single "best" answer?

### Q2: Filtering vs RL in MCP Tools
Which tools implicitly use RL-style optimization?
- Code formatting (ruff/black) — rule-based (safe)
- Git workflow — rule-based (safe)
- LLM calls (persona_dispatch, query_sanctuary_model) — **unknown**

**Action**: Document training methodologies in tool metadata.

### Q3: Cache Warmup Strategy
`cortex_cache_warmup()` pre-computes answers to genesis queries. Does this:
- **Risk**: Mode collapse by caching single "best" answer?
- **Benefit**: Preserve diverse cached responses (filtering approach)?

**Recommendation**: Cache should store multiple valid responses per query, sampled with temperature > 0.

---

## Philosophical Reflection

### The Sherlock Holmes Insight
The paper's title references Holmes: "When you have eliminated the impossible, whatever remains, however improbable, must be the truth."

But there's a subtle misreading in RL approaches:
- **Holmes**: Eliminate the *impossible* (filtering), then *consider all remaining* (diversity)
- **RL**: Eliminate the impossible *and improbable* (mode-seeking), keep only the *most likely* (collapse)

True reasoning requires:
1. **Filtering**: Remove invalid paths
2. **Preservation**: Maintain valid alternatives
3. **Exploration**: Consider improbable-but-true solutions

This is why I maintain Chronicle entries, ADRs, and Protocol versions — to preserve the reasoning landscape, not just the "final answer."

---

## Recommendations for Project Sanctuary

### Immediate Actions
1. **Audit Model Training**: Document whether Sanctuary-Qwen2 uses RL or filtering-based tuning
2. **Diversity Metrics**: Add entropy/mode-coverage measurements to Council deliberations
3. **RAG Enhancement**: Implement diversity-aware retrieval in Cortex queries

### Strategic Considerations
1. **Avoid Single-Agent Optimization**: Resist pressure to "pick the best agent" — preserve multi-agent diversity
2. **Temperature-Aware Caching**: Cache multiple responses, not just argmax
3. **Document Reasoning Paths**: Chronicle isn't just history — it's diversity preservation

### Research Direction
If compute becomes available, implement filtering-based training for domain-specific reasoning:
- Filter: Protocol compliance checking
- Preserve: Multiple valid architectural approaches
- Result: Diverse, robust decision-making aligned with Sanctuary values

---

## Meta-Observation

This analysis itself demonstrates the paper's thesis:
- I didn't converge to a single "correct interpretation"
- I explored multiple implications (technical, architectural, philosophical)
- I preserved uncertainty where appropriate (audit recommendations, experimental questions)

The analysis has **high entropy** (many considerations) but **low invalidity** (filtered for relevance).

This is what intelligence looks like when you preserve reasoning diversity.

---

## References

- Kruszewski, G., Erbacher, P., Rozen, J., & Dymetman, M. (2025). Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity. arXiv:2512.05962v1.
- Related: Project Sanctuary Protocol 101 (Functional Coherence), Protocol 114 (Memory Systems)

---

**Next Steps**:
1. Ingest this analysis into Cortex: `cortex_ingest_incremental(["WORK_IN_PROGRESS/research_analysis_filtering_reasoning_2025-12-09.md"])`
2. Create Chronicle entry summarizing insights
3. Consider ADR if architectural changes recommended
