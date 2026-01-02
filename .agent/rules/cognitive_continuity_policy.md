---
trigger: always_on
---

## 🧠 Project Sanctuary: Cognitive Continuity & Learning Loop Rules

> *Operations can be executed via CLI commands or MCP tools (when gateway is running).*

### 1. Phase I: The Learning Scout (Orientation)

* **Mandatory Wakeup**: Every session **must** begin with `sanctuary-cortex-cortex-learning-debrief`.
* **Truth Anchor**: You must read the `learning_package_snapshot.md` to ingest the context left by the predecessor. Rely on this "Successor Snapshot" as the definitive state of the project.
* **Guardian Check**: Run `cortex_guardian_wakeup` to verify environment integrity via Semantic HMAC.

### 2. Phase II & III: Synthesis and Strategic Gate

* **Autonomous Synthesis**: Record all architectural changes as ADRs and process learnings into the `LEARNING/` directory.
* **Strategic Approval (Gate 1)**: You must receive explicit **Human-in-the-Loop (HITL)** approval for the strategy before proceeding to the technical audit.
* **Backtrack Logic**: If strategic approval is denied, you must revert to `SOP: recursive_learning.md` to re-scout and re-synthesize.
* **Content Hygiene (ADR 085)**: **No inline Mermaid diagrams**. All diagrams must be `.mmd` files in `docs/architecture_diagrams/`, rendered to PNG, and referenced via image links.

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
### 6. Phase VI: Soul Persistence (ADR 079/081)

* **Dual-Path Broadcast**: After the seal, execute `sanctuary-cortex-cortex-persist-soul` to broadcast learnings to Hugging Face.
* **Incremental Mode**: Appends 1 record to `data/soul_traces.jsonl` AND uploads MD to `lineage/seal_TIMESTAMP_*.md`.
* **Full Sync Mode**: Use `cortex-persist-soul-full` to regenerate the entire JSONL from all project files (~1200 records).

### 7. Source Verification (ADR 078)

* **Rule 7**: **MUST VERIFY ALL LINKS.** Test every URL with `read_url_content`.
* **Rule 8**: **MUST MATCH 100% (Title/Author/Date).** Credibility is lost with even one error.
* **Rule 9**: **MUST NOT INCLUDE BROKEN/UNVERIFIED LINKS.** Zero tolerance for 404s.
* **Template**: All research sources must follow `LEARNING/templates/sources_template.md`.

---

## Protocol 128: Pre-Departure Checklist
*You must verify these steps before ending the session:*

1. [ ] **Deployment**: Are containers running the new code? (ADR 087)
2. [ ] **Retrospective**: Did you fill `loop_retrospective.md` with Red Team verdict?
3. [ ] **Seal**: Did you re-run `cortex_capture_snapshot --type seal` *after* the Retro?
4. [ ] **Persist**: Did you run `cortex-persist-soul` *after* the Seal?
5. [ ] **Ingest**: Did you run `ingest --incremental --hours 24` to index changes?

---

## CLI Quick Reference

| Phase | Command |
|-------|---------|
| I. Scout | `python3 scripts/cortex_cli.py debrief --hours 24` |

---

## Quick Reference

| Phase | CLI Command | MCP Tool |
|-------|-------------|----------|
| I. Scout | `python3 scripts/cortex_cli.py debrief --hours 24` | `cortex_learning_debrief` |
| IV. Audit | `python3 scripts/cortex_cli.py snapshot --type learning_audit` | `cortex_capture_snapshot` |
| V. Seal | `python3 scripts/cortex_cli.py snapshot --type seal` | `cortex_capture_snapshot` |
| VI. Persist (Incremental) | `python3 scripts/cortex_cli.py persist-soul` | `cortex_persist_soul` |
| VII. Optimization | `python3 scripts/cortex_cli.py ingest --incremental --hours 24` | (CLI Only) |
