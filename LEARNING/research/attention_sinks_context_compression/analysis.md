# Attention Sinks & Context Compression in Agent Loops

**Status:** CONFIRMED  
**Researcher:** Claude Code CLI (`haiku`) via `cli-agent`  
**Date:** 2026-11-01  
**Source Literature:** Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (arXiv:2309.17453, ICLR 2024)  
**Tags:** ["attention_sinks", "context_compression", "streaming_llm", "kv_cache", "soul_md", "haiku"]

---

## 1. Executive Summary

Autonomous agents running continuous loops face the challenge of KV-cache eviction in long-running contexts. In transformer architectures with causal masking, the initial tokens (position 0 to 4) absorb disproportionate attention mass regardless of semantics, acting as **Attention Sinks**.

Evicting initial tokens causes immediate perplexity explosion even when the sliding window of recent tokens is fully intact. This finding provides the mathematical grounding for why Project Sanctuary's Layer 1 fast-boot context (`context/soul.md`) must be pinned at position 0 as an eviction-proof invariant.

---

## 2. Theoretical Mechanism

### A. Why Initial Tokens Become Attention Sinks
- Causal attention weights originate from a softmax, enforcing $\sum_j \text{softmax}(Q_i K_j) = 1$ across all visible keys.
- When an attention head finds no semantically relevant key to attend to, it must still dump its remaining probability mass somewhere.
- Under causal masking, the initial token is visible to every subsequent query across the entire horizon.
- The model learns to park surplus attention on the initial tokens. These initial keys/values contribute near-zero semantic information to the residual stream, serving effectively as a computational null sink.

### B. Why Evicting Initial Tokens Triggers Perplexity Explosion
- In a naive rolling window or sliding-window cache, the oldest tokens (the sink) are evicted first.
- The softmax probability mass previously dumped into the sink is redistributed across recent tokens that were never trained to absorb it.
- This creates massive distortion in attention distributions, spiking perplexity even if 95% of recent context is preserved.

---

## 3. Engineering Recommendations for Project Sanctuary

### R1. Pin Invariant Prefix at Position 0
Keep identity, human partner invariants, and the Supreme Law Human Gate as a strictly bounded block at the very top of `context/soul.md` (< 50 lines). Ensure local model harnesses (`llama-server`, Ollama) configure `--keep` to at least the token length of this invariant block.

### R2. Separate Stable Prefix from Volatile State
Split runtime context into:
- **Stable Prefix**: Identity, constitutional invariants, and non-negotiables (cached in KV-cache across invocations).
- **Volatile Suffix**: Active task state, recent session friction, and status pointers (frequently modified without invalidating the prefix cache).

### R3. Treat Context as Lossy & Rely on Ledger-First Retrieval
Sliding windows and context windows are inherently lossy over long sessions. Memory continuity must never rely on keeping raw tokens in context; instead, the agent must rely on Layer 2 progressive disclosure (`LEARNING/concepts/`, `chronicle/ledger.json`, `huggingface/soul_traces.jsonl`).
