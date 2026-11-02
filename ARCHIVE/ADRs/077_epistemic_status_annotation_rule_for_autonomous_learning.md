# Epistemic Status Annotation Rule for Autonomous Learning

**Status:** PROPOSED
**Date:** 2025-12-28
**Author:** Claude (Antigravity Agent)


---

## Context

Red team review of the first autonomous learning audit (Entry 337) revealed that high-coherence synthesis can mask epistemic confidence leaks. Claims from ancient sources, modern empirical research, and speculative inference were presented with uniform authority, making it difficult for reviewers to assess reliability without external verification.

GPT's meta-feedback: "Tone alone can launder uncertainty into apparent fact."

This creates risk for RAG ingestion where unqualified claims become canonical memory.

## Decision

All autonomous learning documents MUST include explicit epistemic status annotations for claims:

1. **HISTORICAL** — Ancient/primary sources (e.g., Herodotus, Petrie excavation reports)
2. **EMPIRICAL** — Peer-reviewed modern research with citations (DOI/URL required)
3. **INFERENCE** — Logical deduction from available data (GPR anomalies → possible chambers)
4. **SPECULATIVE** — Creative synthesis without direct evidence

Format: Use inline tags `[HISTORICAL]`, `[EMPIRICAL]`, `[INFERENCE]`, or add an Epistemic Status Box at section headers.

Example:
```markdown
## The Hawara Labyrinth
**Epistemic Status:** HISTORICAL (Herodotus) + INFERENCE (GPR data)
```

## Consequences

**Positive:**
- Prevents epistemic confidence leaks in autonomous learning
- Makes knowledge quality auditable
- Aligns with Anti-Asch Engine goals (resist conformity bias)
- Enables successor agents to assess claim reliability

**Negative:**
- Increases documentation overhead
- Requires discipline during synthesis phase
