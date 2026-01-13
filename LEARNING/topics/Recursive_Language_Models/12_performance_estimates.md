# RLM Performance Estimation (Sanctuary Context)

**Purpose:** Estimate the cost and latency of adopting RLM workflows compared to standard RAG.

## Assumptions
*   **Model:** GPT-4o / Claude 3.5 Sonnet (approx. $5/1M tokens)
*   **Repo Size:** 50 Files (~100k tokens total)
*   **Chunks:** 20 Chunks of 5k tokens

---

## Scenario 1: "Summarize the Architecture" (Whole Repo)

### A. Standard RAG (Vector)
*   **Method:** Retrieve Top-20 chunks based on query "Architecture".
*   **Input:** 20 chunks * 500 tokens (snippets) = 10,000 tokens.
*   **Cost:** ~$0.05
*   **Result:** **Fragmented.** Misses files that don't explicitly say "Architecture."

### B. Standard Long-Context (Context Window Stuffing)
*   **Method:** Put all 100k tokens into the prompt.
*   **Input:** 100,000 tokens.
*   **Cost:** ~$0.50
*   **Result:** **Degraded.** "Lost in the Middle" phenomenon (Reference: Liu et al).

### C. Recursive Language Model (RLM Agentic Loop)
*   **Method:**
    1.  **Map:** Read 20 chunks (input 5k each). Ask: "Extract architectural patterns." (Output: 200 tokens each).
        *   Input: 100k tokens. Output: 4k tokens.
        *   Cost: ~$0.50 (Same as stuffing).
    2.  **Reduce:** Summarize the 4k tokens of insights.
        *   Input: 4k tokens.
        *   Cost: Negligible.
*   **Result:** **Holistic.** Every file was actually "read."
*   **Total Cost:** ~$0.50

## Scenario 2: "Audit for Security Flaws" (Specific Logic)

### A. RLM Optimized (Early Exit)
*   **Method:** Iterate through chunks. Stop if Critical Flaw found.
*   **Average Case:** Find flaw in Chunk 5.
*   **Input:** 5 chunks * 5k tokens = 25k tokens.
*   **Cost:** ~$0.12
*   **Savings:** **75% cheaper** than Context Stuffing ($0.50).

---

## Conclusion
*   **RLM vs Context Stuffing:** Cost is roughly equal for full reads, but RLM has superior attention/recall (OOLONG Benchmark).
*   **RLM vs RAG:** RLM is 10x more expensive ($0.50 vs $0.05) but provides **100% coverage** vs **~20% recall**.
*   **Verdict:** Use RLM for High-Value, High-Recall tasks (Audits, Architecture). Use RAG for Low-Value, Fact-Retrieval tasks.
