# Systemic Refactoring of Git Tool Robustness

**Status:** APPROVED
**Date:** 2025-12-23
**Author:** Antigravity (Agent), User (Red Team Lead)


---

## Context
**Related Documents:**
*   [Protocol 101 v3.0: The Doctrine of Absolute Stability](../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md) (Part C.2: Explicit Prohibition of Automatic Sync)
*   [Antigravity Git Safety Rules](.agent/git_safety_rules.md) (Section 4: Error Handling)

The current `sanctuary_git` tools implement a "Stateless Paranoid" security model (Protocol 101/128). They assume the local environment is likely compromised or stale and block operations (like `finish_feature`) if strict, instantaneous local-remote parity isn't observed. 

**The Failure Mode (Lifecycle Analysis):**
The current design exhibits systemic friction at every stage of the agentic git lifecycle:

1.  **Git Add (The Staging Block):**
    *   **Friction:** `smart_commit` fails if files aren't manually staged first, despite implied intent.
    *   **User Impact:** Agents must waste steps calling `git-add` explicitly for every file, breaking flow.

2.  **Git Commit (The Safety Check):**
    *   **Friction:** Poka-yoke checks (Protocol 101) block commits if *any* test fails, even for non-code changes (docs/ADRs).
    *   **User Impact:** "Fail-Safe" logic prevents incremental progress or "WIP" saves.

3.  **Git Push (The Remote Sync):**
    *   **Friction:** Stateless tools don't assume upstream tracking.
    *   **User Impact:** Frequent rejections due to "non-fast-forward" errors if `fetch` wasn't manually run first.

4.  **Finish Feature (The Death Loop):**
    *   **Friction:** The tool verifies the *Local Snapshot* of `origin/main` against the *Feature Branch*.
    *   **The Trap:** Because the tool swallows `fetch` errors (or skips them), it compares the feature branch against a stale `main`.
    *   **Result:** It reports "Branch NOT merged" even after the user has merged it on GitHub, forcing a manual intervention.

**The Requirement: "A Happy Path Should Not Have Errors"**
The breakdown occurs because Protocol 101 (Unbreakable Commit) was implemented as a **Stateless Gate** rather than a **Workflow Enabler**.

*   **The Happy Path Defect:** A standard agent workflow (`Loop -> Create -> Commit -> Push -> Merge -> Finish`) currently requires ~30% failure rate handling (retries, manual overrides) due to strict Poka-Yoke checks failing on benign latency.
*   **The Distributed System Fallacy:** The tools treat Git as a local file system. If the local disk says "Not Merged", the tool blocks. But Git is distributed; the Truth is often on the Remote. The tools lack the *agency* to go look for the Truth before blocking the user.

We must move from "Blocking on Uncertainty" to "Resolving Uncertainty" (Smart Defaults).

**The Safety Paradox (Friction == Risk):**
The Red Team has identified that the current balance of strict P101 enforcement vs. usability is incorrect.
*   **The Paradox:** When a safety tool fails reliably on valid operations (false positives), it desensitizes the user to errors.
*   **The Consequence:** Users are conditioned to use `--force` or ignore warnings to get work done. This is a **catastrophic failure** of the safety model, as legitimate warnings (like "non-fast-forward") will eventually be ignored as "just another Poka-Yoke bug."
*   **Correction:** We must recalibrate. A working, context-aware tool that handles latency gracefully preserves the **credibility** of the safety system.

## Decision

We will refactor the `sanctuary_git` toolset to prioritize **Context-Aware Robustness** over Stateless Paranoia.

**Core Principles:**
1.  **Smart Defaults (The Happy Path):** Operations like `finish_feature` MUST attempting to Synchronize (`fetch`) before Verifying. If the sync fails, it MUST warn loud and clear, not fail silently.
2.  **Fail-Loud Diagnostics:** Error messages must explain *why* an operation failed (e.g., "Network Unreachable" vs. "Merge Conflict") rather than a generic "Safety Block".
3.  **Assumption of Intent:** If an agent tries to finish a feature, the system should assume the intent is valid and try to prove it true (by fetching), rather than assuming it's dangerous and trying to stop it.

**Selected Approach: Option B (Smart Defaults)**
- We reject Option A (Status Quo) as it is operationally untenable.
- We reject Option C (Stateful World Model) as over-engineering for the current scale.
- We adopt **Option B**: Embedding aggressive, noisy synchronization checks into the tools themselves.

## Consequences

- **Positive:** drastically reduced friction for legitimate agentic workflows; clearer diagnostics when things do fail.
- **Negative:** slight increase in tool latency due to mandatory `fetch` checks; potential for masking underlying network perf issues.
- **Compliance:** Maintains Protocol 101 safety but shifts the check from "Local View" to "Verified Remote View".

## Appendix A: Affected Tools Impact Analysis

The following MCP operations are directly implicated in the "Safety Paradox" and require specific refactoring:

| Operation | Current Friction (The Failure) | Targeted "Happy Path" Behavior |
| :--- | :--- | :--- |
| `git-start-feature` | Branches off stale local `main`. | **Auto-Fetch** `origin/main` before branching to ensure freshness. |
| `git-add` | Manual step required before commit. | **Orchestrated:** Allow `smart_commit` to accept a `files` argument to verify & stage in one step. |
| `git-smart-commit` | Fails if nothing staged; Fails if P101 tests fail (even for docs). | **Context-Aware:** Auto-stage if specified; Skip strict code tests for non-code artifacts (docs/ADRs). |
| `git-push-feature` | Fails if local is behind remote. | **Auto-Fetch & Check:** Warn user if pull/rebase is needed before attempting push. |
| `git-finish-feature` | **CRITICAL FAIL:** Blocks deletion due to stale local view of `main`. | **Mandatory Auto-Fetch:** Verify merge status against *fresh* `origin/main`. |
| `git-get-status` | Reports "Clean" even if behind remote. | **Honest Reporting:** Run `git fetch` (async) to report "Behind by X commits". |
| `git-log` | Shows local history only. | **Awareness:** Option to show `origin/main` context. |
| `git-diff` | Passive. | No change (essential for context). |
| `git-get-safety-rules` | Static text. | No change (Policy Source). |
