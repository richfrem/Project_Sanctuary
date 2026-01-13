# Learning Topic: Recursive Language Models (RLM) & DeepMind Titans

**Status:** Synthesized (Source Text Verified)
**Date:** 2026-01-12
**Epistemic Status:** <entropy>0.05</entropy> (Verified Source Text vs Public Narrative)
**Sources:**
- **RLM Paper:** *Recursive Language Models* (Zhang, Kraska, Khattab - MIT CSAIL, Dec 2025) - [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- **Titans Paper:** *Titans: Learning to Memorize at Test Time* (Google DeepMind, Jan 2025) - [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

## I. The Narrative De-Confliction
**The Viral Claim:** "DeepMind built RLM which kills RAG."
**The Ground Truth:** The viral narrative conflates two separate breakthroughs.
1.  **RLM (Research Strategy):** Developed by **MIT CSAIL** (Alex L. Zhang, Tim Kraska, Omar Khattab). It is a purely *inference-time* strategy using code execution (REPL) to manage context.
2.  **Titans (Model Architecture):** Developed by **Google DeepMind**. It introduces a new neural architecture with "Test-Time Training" and persistent memory modules.

---

## II. Recursive Language Models (RLM) - Deep Dive
**Core Concept:** *Context as Environment*
RLM fundamentally shifts how LLMs interact with long contexts. Instead of tokenizing the entire document into the prompt, RLM treats the context as an **external object (variable)** in a Python REPL.

### 1. The Mechanism (The "REPL" Loop)
*   **Initialization:** The RLM initializes a generic Python REPL. The long prompt is loaded as a variable `context` (e.g., a 10M char string).
*   **The Interface:** The LLM is given tools to:
    1.  **Inspect:** `print(context[:1000])` or `len(context)`.
    2.  **Decompose:** Write Python code to slice or chunk the `context`.
    3.  **Recurse:** Call `llm_query(chunk)` to spawn a *sub-agent* (recursive call) on a specific slice.
*   **The "MapReduce" Effect:** This converts "reading a book" from a linear attention task into a hierarchical programming task. The model effectively writes a "MapReduce" job on the fly.

### 2. Benchmark Findings (The "Context Rot" Solution)
The paper (Zhang et al.) demonstrates that effective context length is task-dependent.
*   **S-NIAH (Needle in Haystack):** Modern frontier models (GPT-5) handle this well natively.
*   **OOLONG (Dense Reasoning):** Frontier models fail catastrophically as length increases because the *reasoning* requires connecting every line.
*   **RLM Performance:**
    *   **OOLONG-Pairs (Quadratic Complexity):** RLM (using GPT-5) achieves **58.0% F1**, while base GPT-5 scores **<0.1%**.
    *   **Scale:** successfully handles inputs **two orders of magnitude** larger than the model's window (tested up to 10M+ tokens).
*   **Cost:** RLM is often *cheaper* than base models because it reads selectively. Instead of paying for 1M tokens for every query, it pays for the "MapReduce" orchestration + small slice reads.

---

## III. DeepMind Titans - The "Perfect Memory"
**Core Concept:** *Neural Long-Term Memory*
Titans (arXiv:2501.00663) is the likely source of the "No RAG / Perfect Memory" claim.

*   **Architecture:** It adds a **Neural Memory Module** to the Transformer.
*   **Test-Time Training:** It updates its *weights* during inference based on a "surprise metric." If data is surprising, it is "memorized" (weights updated).
*   **RAG Killer?** DeepMind argues that Attention is "Short-Term Memory" and these new Weights are "Long-Term Memory," potentially removing the need for external vector databases.

---

## IV. Strategic Synthesis for Sanctuary
We should adopt RLM strategies immediately as they are **model-agnostic inference patterns**, whereas Titans is a proprietary architecture.

### Actionable Protocols
1.  **Recursive Summarization (RLM-Lite):** When we ingest large docs, we should not just "chunk and embed." We should have an agent write a plan to "read and summarize" hierarchically.
2.  **Context-as-Variable:** For massive files (like full repo verification), we should provide the agent with `grep` / `read_slice` tools (which we have) and encouraging *iterative probing* rather than "read whole file."
3.  **Future Architecture:** Monitor Titans for when open-weights versions (or API access to "memory-updating" models) become available, as this aligns with our **Soul Persistence** goals.
