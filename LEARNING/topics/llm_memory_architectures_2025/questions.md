# Research Questions: LLM Memory Architectures

**Topic:** Long-Term Memory for Cognitive Continuity
**Session:** 2026-01-05

## Primary Questions

1. **HINDSIGHT Framework:** How does the 4-tier memory separation (World/Experience/Opinion/Reflection) compare to Sanctuary's CAG/RAG/LoRA triad?

2. **SciBORG FSA:** Can finite-state automaton memory improve Protocol 128's state tracking reliability?

3. **Mem0 Implementation:** Is the open-source Mem0 approach compatible with our ChromaDB + LangChain stack?

4. **Context Engineering:** What are Anthropic's specific recommendations for managing context in long-horizon agentic tasks?

8. **HINDSIGHT Implementation:** How do we specifically implement the "Opinion Network" (O) and "Observation Network" (S) using our existing ChromaDB/Postgres stack?
9.  **Nested Learning:** How can the "Associative Optimizer" concept be materialized as a specific "Synaptic Phase" in Protocol 128?
10. **CARA Integration:** How do we implement the "Reflect" operation (Disposition Parameters) without creating excessive latency in the learning loop?

## Secondary Questions

5. **Episode Boundaries:** How should Chronicle entries be clustered into "holistic episodes" for better retrieval?

6. **Memory Decay:** Should old memories decay/summarize over time, or maintain full fidelity?

7. **Cross-Session Verification:** Beyond HMAC, what other integrity mechanisms exist for persistent memory?

## Alignment with Protocol 128

| Protocol Phase | Memory Architecture Parallel |
|----------------|------------------------------|
| I. Scout | Short-term context acquisition |
| II. Synthesize | Experience/Opinion memory formation |
| III. Strategic Gate | Reflection memory |
| IV. Red Team Audit | Memory verification/pruning |
| V. Seal | Mid-term persistence |
| VI. Soul Persist | Parameter-based memory (weights) |
