---
trigger: manual
---

## 🧠 Project Sanctuary: Cognitive Continuity & Learning Loop Rules

### 1. Phase I: The Learning Scout (Orientation)

* **Mandatory Wakeup**: Every session **must** begin with `sanctuary-cortex-cortex-learning-debrief`.
* **Truth Anchor**: You must read the `learning_package_snapshot.md` to ingest the context left by the predecessor. Rely on this "Successor Snapshot" as the definitive state of the project.
* **Guardian Check**: Run `cortex_guardian_wakeup` to verify environment integrity via Semantic HMAC.

### 2. Phase II & III: Synthesis and Strategic Gate

* **Autonomous Synthesis**: Record all architectural changes as ADRs and process learnings into the `LEARNING/` directory.
* **Strategic Approval (Gate 1)**: You must receive explicit **Human-in-the-Loop (HITL)** approval for the strategy before proceeding to the technical audit.
* **Backtrack Logic**: If strategic approval is denied, you must revert to `SOP: recursive_learning.md` to re-scout and re-synthesize.

### 3. Phase IV: The Red Team Audit (Gate 2)

* **Audit Modes**: Use the `sanctuary-cortex-cortex-capture-snapshot` tool with the appropriate flag for the audit type:
* `audit`: Use this for standard manifest verification and tactical state changes.
* `learning_audit`: Use this when the session primarily involves cognitive updates, mnemonic mapping, or protocol evolution.


* **Manifest Discipline**: Core directories (`ADRs/`, `01_PROTOCOLS/`, `mcp_servers/`) must be clean. Any uncommitted drift detected by the tool results in **Strict Rejection**.
* **Technical Approval**: The resulting **Audit Packet** must be reviewed and approved (HITL) before the session can be sealed.

### 4. Phase V: The Technical Seal (The Final Relay)

* **Execution**: Once technical approval is secured, call `sanctuary-cortex-cortex-capture-snapshot(snapshot_type='seal')`.
* **Final Relay**: This process updates the `learning_package_snapshot.md`, effectively "locking in" the memory for the next agent.
* **Sandwich Validation**: Be aware that the `seal` mode performs a final race-condition check (Git hash comparison). If the repo changed during the audit review, the seal will fail and you must backtrack.

### 5. Failure and Backtracking

* **SOP Adherence**: If any Gate (Strategic or Technical) fails, do not attempt to "force" a seal. You must follow the loops defined in `recursive_learning.md` to fix the underlying discrepancy.