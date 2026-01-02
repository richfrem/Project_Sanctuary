# Diversity-Aware Reasoning Architecture

> [!WARNING]
> **SUPERSEDED**: This ADR is superseded by **ADR 052: Diversity-Aware Reasoning Architecture** (v2).

**Status:** Superseded
**Date:** 2025-12-09
**Superseded By:** ADR 052
**Author:** AI Assistant


---

## Context

Recent research (arXiv:2512.05962v1) highlights that RL-based optimization drives "mode collapse" (convergence to a single, often overconfident answer) and suppresses valid alternative reasoning paths. Current RAG systems often use "top-k" retrieval, which is inherently mode-seeking. Project Sanctuary aims for "Anti-Asch" robustness, which requires preserving diverse perspectives.

## Decision

We will adopt a **Diversity-Aware Reasoning Architecture**.
1. **Retrieval**: Move from simple Top-K to Diversity Sampling (e.g., Maximal Marginal Relevance or stratified sampling) in RAG.
2. **Training**: detailed audit of model tuning to prefer Filtering (preserving distribution of correct answers) over Mode-Seeking RL.
3. **Execution**: Multi-agent dispatch shall prioritize maintaining distinct reasoning paths over forcing consensus.

## Consequences

Positive:
- Prevents mode collapse in reasoning (overconfidence in single wrong answer).
- Increases robustness of RAG retrieval by preserving diverse contexts.
- Aligns architecture with multi-agent diversity goals.

Negative:
- potentially higher compute cost for diversity sampling.
- Complexity in measuring "diversity".
