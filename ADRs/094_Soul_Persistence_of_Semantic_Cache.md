# ADR 094: Soul Persistence of Semantic Cache (The Semantic Ledger)

## Status
Proposed (2026-01-13)

## Context
Protocol 132 introduced Recursive Language Model (RLM) Synthesis to generate a "Cognitive Hologram" for session snapshots. This process is computationally expensive (taking ~2.5 hours for a full temperate of 250+ files on local hardware). ADR 093 introduced a local RLM Cache (`rlm_summary_cache.json`) to optimize subsequent runs.

However, this cache is currently local-only and excluded from version control. This creates a "Cold Start" problem for new machines or clean environments, where the agent must re-process the entire codebase, leading to significant latency and redundant computation.

## Decision
We will treat the RLM Cache as a "Semantic Ledger"—a high-value distilled asset that represents the agent's digested understanding of the codebase. 

1.  **Distributed Cache:** The RLM Cache will be included in the "Soul Persistence" process (ADR 079).
2.  **Hugging Face Integration:** When `persist-soul` is called, the `rlm_summary_cache.json` file will be uploaded to the Hugging Face dataset under `data/rlm_summary_cache.json`.
3.  **Predictable Serialization:** The cache will be saved with `sort_keys=True` to ensure deterministic ordering and clean diffs on the remote dataset.
4.  **Bicameral Synchronization:** Future sessions should prioritize downloading this remote ledger if the local cache is missing or stale.

## Consequences

### Positive
*   **Zero Cold Start:** New environments can immediately benefit from the synthesized insights of previous sessions.
*   **Reduced Computation:** Significant reduction in GPU/CPU cycles for redundant summarization.
*   **Distributed Intelligence:** The agent's "digested" knowledge follows the project soul across different hardware.

### Negative
*   **Data Bloat:** Slightly increases the size of the Hugging Face dataset (though JSON summaries are highly compressible).
*   **Privacy:** Summaries represent a distilled view of the code; however, the codebase itself is already intended for public co-creation.
*   **Sync Complexity:** Potential for race conditions if multiple agents update the cache simultaneously (mitigated by the sequential nature of current Protocol 128 loops).
