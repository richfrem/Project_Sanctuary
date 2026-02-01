---
description: "Standard operating procedure for Protocol 128 Hardened Learning Loop (Scout -> Synthesize -> Audit -> Seal -> Persist)."
---

# Recursive Learning Loop (Protocol 128)

**Objective:** Cognitive continuity and autonomous knowledge preservation.
**Reference:** `ADRs/071_protocol_128_cognitive_continuity.md`
**Tools:** Cortex MCP Suite, Git, Chronicle

---

## Phase I: The Learning Scout (Orientation)

1.  **Mandatory Wakeup**: Call `cortex_learning_debrief` (MCP) or `python3 scripts/cortex_cli.py debrief --hours 24` (CLI)
2.  **Truth Anchor**: Read the `learning_package_snapshot.md` returned by debrief
3.  **Guardian Check**: Run `cortex_guardian_wakeup` to verify environment integrity via Semantic HMAC
4.  **Security Binding**: You are now bound by Git Pre-Flight (Protocol 101) and Execution Lock (Human Gate)

## Phase II: Synthesis

1.  **Context Check**: Use `view_file` to check existing topic notes in `LEARNING/topics/...`
2.  **Record Changes**: All architectural changes → ADRs, learnings → `LEARNING/` directory
3.  **Conflict Resolution**:
    *   New confirms old? → Update/Append
    *   New contradicts old? → Create `disputes.md` (Resolution Protocol)
4.  **Content Hygiene (ADR 085)**: No inline Mermaid diagrams. All diagrams as `.mmd` files.

## Phase III: Strategic Gate (HITL Required)

1.  **Strategic Review**: Human reviews `/ADRs` and `/LEARNING` documents created during session
2.  **Align Intent**: Ensure autonomous research matches session goals
3.  **Approval**: Explicit "Approved" or "Proceed" required
4.  **Backtrack**: If denied → re-scout and re-synthesize

## Phase IV: Red Team Audit (HITL Required)

1.  **Snapshot Generation**: Call `cortex_capture_snapshot --type audit` or `--type learning_audit`
2.  **Manifest Discipline**: Core directories (`ADRs/`, `01_PROTOCOLS/`, `mcp_servers/`) must be clean
3.  **Zero-Trust Check**: Tool verifies manifest against `git diff`. Discrepancies flag Strict Rejection.
4.  **Audit Review**: Human reviews `red_team_audit_packet.md` for technical truth

## Phase V: The Technical Seal

1.  **Execute Seal**: Call `cortex_capture_snapshot --type seal`
2.  **Final Relay**: Updates `learning_package_snapshot.md` (the "memory" for next session)
3.  **Sandwich Validation**: If repo changed during audit review → seal fails, backtrack required
4.  **Git Commit**: Commit all learning artifacts per Protocol 101 Preservation

## Phase VI: Soul Persistence (ADR 079/081)

1.  **Dual-Path Broadcast**: Call `cortex_persist_soul` to broadcast learnings to Hugging Face
2.  **Incremental Mode**: Appends record to `data/soul_traces.jsonl` + uploads MD to `lineage/`
3.  **Full Sync Mode**: Use `cortex_persist_soul --full` for complete regeneration

## Phase VII: Retrospective & Curiosity Vector

1.  **Retrospective**: Update `loop_retrospective.md` with session verdict
2.  **Deployment Check**: Are containers running the new code? (ADR 087)
3.  **Curiosity Vector**: Append incomplete ideas to "Active Lines of Inquiry" in `guardian_boot_digest.md`
4.  **Ingest**: Run `cortex ingest --incremental --hours 24` to index changes

---

## Pre-Departure Checklist (Protocol 128)

- [ ] **Retrospective**: Filled `loop_retrospective.md`?
- [ ] **Deployment**: Containers running new code?
- [ ] **Curiosity Vector**: Recorded any future "Lines of Inquiry"?
- [ ] **Seal**: Re-ran `snapshot --type seal` after Retro?
- [ ] **Persist**: Ran `cortex_persist_soul` after Seal?
- [ ] **Ingest**: Ran `ingest --incremental` to index changes?

---

## Quick Reference

| Phase | CLI Command | MCP Tool |
|-------|-------------|----------|
| I. Scout | `python3 scripts/cortex_cli.py debrief --hours 24` | `cortex_learning_debrief` |
| IV. Audit | `python3 scripts/cortex_cli.py snapshot --type learning_audit` | `cortex_capture_snapshot` |
| V. Seal | `python3 scripts/cortex_cli.py snapshot --type seal` | `cortex_capture_snapshot` |
| VI. Persist | `python3 scripts/cortex_cli.py persist-soul` | `cortex_persist_soul` |
| VII. Ingest | `python3 scripts/cortex_cli.py ingest --incremental --hours 24` | (CLI Only) |

---

## Next Session: The Bridge

1. **Boot**: Next session agent calls `cortex_learning_debrief`
2. **Retrieve**: Tool identifies `learning_package_snapshot.md` as "Strategic Successor Context"
3. **Resume**: Agent continues from where predecessor left off

---
// End of Protocol 128 Workflow
