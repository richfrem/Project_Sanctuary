# ADR 093: Incremental RLM Cache (The "Semantic Ledger")

## Status
Proposed

## Context
Protocol 132 (RLM Synthesis) requires the system to recursively "read" and summarize the entire repository (Protocols, ADRs, Code) to generate the "Cognitive Hologram". 
Initial execution (Phase VIII) revealed this is computationally expensive and slow when running on sovereign hardware (local 7B model), processing ~250 files.
Most files in the repository (e.g., older ADRs, stable Protocols) do not change between sessions. Re-summarizing them is a waste of energy and time.

## Decision
We will implement an **Incremental RLM Cache** ("The Semantic Ledger") to optimize the synthesis process.

### 1. The Cache Manifest
A JSON registry located at `.agent/learning/rlm_summary_cache.json` will store the authorized summaries.

```json
{
  "01_PROTOCOLS/001_Genesis.md": {
    "hash": "sha256_of_content...",
    "mtime": 1736734500,
    "summary": "Establishes the foundational constraints...",
    "model_version": "Sanctuary-Qwen2-7B-v1.0"
  }
}
```

### 2. The Logic (Update to `_rlm_map`)
Before invoking the LLM, the system will:
1. Calculate the SHA256 hash of the target file.
2. Check the Semantic Ledger.
3. **If Match:** Return cached summary (Cost: 0s).
4. **If Mismatch:** Invoke Local LLM, generate new summary, update Ledger (Cost: ~30s).

### 3. Verification & Staleness
* The Ledger MUST be checked into git to allow "shared memory" between agents.
* If the `model_version` changes (e.g., we upgrade to Qwen2.5-14B), the entire cache is invalidated to ensure the new intelligence is applied.

### 4. Exclusion
The cache file itself `rlm_summary_cache.json` must be excluded from the `learning_package_snapshot.md` to prevent recursion loops (the map cannot contain the map of the map).

### 5. Modular Targeting (Agility)
The `_rlm_map` function must be refactored to accept versatile inputs, supporting granular execution:
* **Single File:** `summarize("protocol_128.md")` -> Returns singleton summary.
* **Directory:** `summarize("ADRs/")` -> Returns summary of all files in dir.
* **Manifest:** `summarize("changes.json")` -> Summarizes only files listed.

This enables:
1. **Rapid Testing:** Verify RLM logic on 1 file in 10s instead of 200 files in 20m.
2. **Targeted Updates:** Re-synthesize only the "ADR" section if only ADRs changed.

## Consequences
* **Positive:** Seal time reduces from ~45 minutes to <2 minutes for typical sessions.
* **Positive:** "RLM CLI" becomes a useful tool for ad-hoc queries (e.g., "What does this file do?").
* **Negative:** Adds state management complexity.

## Implementation Plan (Phase IX)
1. Modify `LearningOperations._rlm_map` to implement the read-through cache.
2. Refactor `_rlm_map` to accept `Union[str, List[str], Path]` for flexible targeting.
3. Expose via CLI: `cortex_cli.py rlm --target <path>`.
4. Add `rlm_summary_cache.json` to `.agent/learning/`.
