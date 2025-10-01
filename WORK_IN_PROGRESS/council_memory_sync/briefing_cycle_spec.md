# Council Briefing Cycle Specification

## Purpose
Establishes a shared mnemonic context at the start of each Council deliberation.

---

## Sequence

1. **Temporal Anchors**
   - Retrieve the two most recent consecutive entries in `Living_Chronicle.md`.
   - Verify continuity (see continuity_check_module.py).
   - Present anchor titles + checksums to all Council roles.

2. **Prior Directives**
   - Gather the last 1–2 Council directives relevant to the task domain.
   - Summarize their key outputs into a `briefing_packet.json`.

3. **Unified Briefing Distribution**
   - Inject `briefing_packet.json` into the initial state of Coordinator, Strategist, and Auditor.
   - Confirm that all roles start with the same context before deliberation.

---

## Notes
- Steward remains query proxy. All Cortex pulls must be validated and executed by Steward.
- Logs for each briefing cycle should be written to:
  `WORK_IN_PROGRESS/council_memory_sync/briefing_logs_<timestamp>.md`

---