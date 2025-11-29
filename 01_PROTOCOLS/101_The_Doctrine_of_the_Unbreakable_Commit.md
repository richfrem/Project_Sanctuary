# Protocol 101 v2.0: The Doctrine of the Unbreakable Commit

**Status:** CANONICAL
**Classification:** Foundational Mnemonic & Repository Integrity Framework
**Version:** 2.0 (Hardened by Catastrophic Failure)
**Authority:** Reforged after the "Sovereign Reset" incident, embodying the Doctrine of the Negative Constraint and the Steward's Prerogative.
**Linked Protocols:** P89 (Clean Forge), P88 (Sovereign Scaffold), P27 (Flawed, Winning Grace)

---
### **Changelog v2.0**
*   **Architectural Split:** Protocol now governs both **Data Integrity** (the "what") and **Action Integrity** (the "how").
*   **Prohibition of Destructive Actions:** Explicitly forbids AI-driven execution of `git reset`, `git clean`, `git pull` with overwrite potential, and other destructive commands.
*   **Mandate of the Whitelist:** AI-driven Git operations are now restricted to a minimal, non-destructive whitelist (`add`, `commit`, `push`).
*   **Prohibition of Improvisation:** Forbids AI from implementing its own error-handling for Git operations. All failures must be reported directly to the Steward.
*   **Canonized the Sovereign Override:** Formally documents the Steward's right to bypass this protocol using `git commit --no-verify` in crisis situations.
---

## 1. Preamble: The Law of the Sovereign Anvil

This protocol is a constitutional shield against both unintended data inclusion (`git add .`) and unauthorized destructive actions (`git reset --hard`). It transforms manual discipline into an unbreakable, automated law, ensuring every change to the Cognitive Genome is a deliberate, verified, and sovereign act, protecting both the steel and the anvil itself.

## 2. The Mandate: A Two-Part Integrity Check

All AI-driven repository actions are now governed by a dual mandate, enforced by a combination of architectural design and the pre-commit hook.

### Part A: Data Integrity (The "What")

This is the original mandate, enforced by the pre-commit hook.

1.  **Guardian's Approval is Law:** No commit shall proceed without a `commit_manifest.json`.
2.  **Verifiable Provenance:** The hook MUST verify the SHA-256 hash of every staged file against the manifest. A mismatch aborts the commit.
3.  **Ephemeral Authority:** Upon a successful commit, the hook MUST delete the manifest.

### Part B: Action Integrity (The "How")

This new mandate is a set of unbreakable architectural laws governing the AI's capabilities.

1.  **Absolute Prohibition of Destructive Commands:** The orchestrator and all its subordinate agents are architecturally forbidden from executing any Git command that can alter or discard uncommitted changes. This list includes, but is not limited to: `git reset`, `git checkout -- <file>`, `git clean`, and any form of `git pull` that could overwrite the working directory.

2.  **The Mandate of the Whitelist:** The AI's "hands" are bound. The `_execute_mechanical_git` method is restricted to a minimal, non-destructive whitelist of commands: `git add <files...>`, `git commit -m "..."`, and `git push`. No other Git command may be executed.

3.  **The Prohibition of Sovereign Improvisation:** The AI is forbidden from implementing its own error-handling logic for Git operations. If a whitelisted command fails, the system's only permitted action is to **STOP** and **REPORT THE FAILURE** to the Steward. It will not try to "fix" the problem.

### Part C: The Doctrine of the Final Seal (Architectural Enforcement)

This mandate requires a specific architectural pattern to ensure the Protocol 101 failures observed during the "Synchronization Crisis" are permanently impossible. The Guardian must audit and enforce this structure.

1.  **The Single-Entry Whitelist Audit:** The underlying Git command executor (e.g., `_execute_mechanical_git` in lib/git/git_ops.py) must be audited to ensure that **only** the whitelisted commands (`add`, `commit`, `push`) are possible. Any attempt to pass a non-whitelisted command **MUST** result in a system-level exception, not just a reported error.

2.  **Explicit Prohibition of Automatic Sync (Fixing Pillar 2):** Any internal function that automatically executes a `git pull`, `git fetch`, or `git rebase` without explicit, top-level command input (e.g., a dedicated `git_sync_from_main` tool) is a violation of this protocol. The architectural code responsible for this unauthorized synchronization **MUST BE REMOVED**.

3.  **Mandate of Comprehensive Cleanup (Fixing Pillar 3):** The function responsible for completing a feature workflow (e.g., `git_finish_feature`) **MUST** contain a verified, two-step operation:
    a. Delete the local feature branch.
    b. **Delete the corresponding remote branch** (e.g., `git push origin --delete <branch-name>`).
    Failure on either step is a Protocol violation and requires an immediate **STOP** and **REPORT**.

4.  **Audit of Ephemeral Authority:** The mechanism responsible for deleting the `commit_manifest.json` after a successful commit (as mandated by Part A) must be audited to ensure it is correctly executed on the local machine to prevent future conflicts and unnecessary manual cleanup.

## 3. The Guardian's Cadence (Now with Negative Constraint)

The cadence for a Guardian-sealed commit now includes the explicit prohibition of dangerous actions.

1.  **The Forging:** The Guardian commands an agent to generate the `commit_manifest.json` via a Sovereign Scaffold. The command itself **MUST** include a negative constraint, for example: *"This scaffold is forbidden from containing any logic for destructive Git operations."*

2.  **The Steward's Verification:** The Steward executes `python3 tools/verify_manifest.py` to audit the manifest against the file system.

## 4. The Steward's Prerogative: The Sovereign Override

In a crisis or during recovery from a systemic failure (a "Red State"), the Steward has the absolute right to override this entire protocol. This is the constitutional escape hatch.

*   **Action:** The Steward may use `git add .` to stage all changes.
*   **Command:** The Steward will then execute the commit using the `--no-verify` flag, which explicitly and intentionally bypasses the pre-commit hook.
    `git commit --no-verify -m "Steward's Sovereign Override: Justification..."`

This ensures the final, absolute authority over the repository's history always rests with the human-in-the-loop.

### Part D: The Doctrine of Environmental Integrity (Pillar 6)

This mandate ensures the System Requirements are formally documented and verified by the Guardian before any operation is initiated.

1.  **Mandatory Dependency Manifest:** The Guardian must maintain a file (e.g., `REQUIREMENTS.env`) listing all required external dependencies (tools, libraries, extensions) not managed by Python's `requirements.txt`.
2.  **Git LFS Requirement (Immediate Canonization):** The dependency on the **Git LFS (Large File Storage) extension** is now formally canonized as a non-negotiable requirement for the execution of all Git operations.
3.  **Pre-Flight Check Mandate:** The agent's `git_start_feature` and `git_sync_main` tools must perform a pre-flight check to verify that all dependencies in the `REQUIREMENTS.env` file are installed and accessible on the execution path. Failure to pass the pre-flight check **MUST** result in a `ProtocolViolationError` with a clear message instructing the Steward on the missing dependency.