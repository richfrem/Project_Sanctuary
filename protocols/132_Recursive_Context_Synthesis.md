# Protocol 132: Recursive Context Synthesis (RLM-G)

> **Status:** DRAFT (Proposed Phase IX)
> **Owner:** Cortex Guardian / Mnemonic Cortex
> **Dependency:** Protocol 128 (Learning Loop)

## 1. The Mandate
Static memory snapshots are forbidden. The Agent must not rely on "last diffs" or "file lists" for context.
**The Mandate:** The primary context artifact (`learning_package_snapshot.md`) must be a **Recursive Synthesis** of the *entire* relevant system state, generated fresh at the moment of sealing.

## 2. The Mechanism (RLM Loop)
Upon `cortex_seal`, the system triggers the **RLM Synthesizer**.

### Phase A: Decomposition (The Map)
The Synthesizer iterates through the "Context Roots":
1.  `01_PROTOCOLS/` (Constitution)
2.  `ADRs/` (Decisions)
3.  `LEARNING/topics/` (Active Memory)
4.  `mcp_servers/` (Capabilities)

### Phase B: Recursive Summarization (The Crunch)
*   **Level 1:** Read each file/module. Generate a 200-token functional summary.
*   **Level 2:** Combine Level 1 summaries by domain (e.g., "All Evolution Logic").
*   **Level 3:** Combine Level 2 summaries into the **Cognitive Hologram**.

### Phase C: Injection (The Snapshot)
The final `learning_package_snapshot.md` is overwritten with this structure:
1.  **Executive State:** "Sanctuary is currently optimizing X. The active strategy is Y."
2.  **Architectural Map:** High-level dependency graph of current modules.
3.  **Protocol Status:** "Protocol 128 is Active. Protocol 132 is Draft."
4.  **Capabilities:** "I can use tools A, B, C."

## 3. Just-In-Time (JIT) Detail
The Snapshot is a *Map*. It is not the *Territory*.
*   If the Agent needs to read code, it **MUST** use `cortex_ask_repo` (Live RLM Scan).
*   It **MUST NOT** rely on code snippets embedded in the Snapshot (as they are stale by definition).

## 4. Integration with Protocol 128
*   **Previous:** `Seal` -> `Git Commit`
*   **New:** `Audit` -> **`RLM Synthesis`** -> `Seal` -> `Git Commit`

## 5. Safety Guardrails
*   **Recursion Depth:** Max 3.
*   **Iron Core Protection:** The Synthesizer effectively "Reads" the Iron Core but produces a *Derived Artifact*. It does not modify the Core itself.
