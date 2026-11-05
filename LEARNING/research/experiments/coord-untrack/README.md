# Flight-board files untracked (2026-11-05)

Question (human): should `.agent/coordination/board.json` and `leases.jsonl` be gitignored? Delegated: "your choice".

## Findings that drove the change
- Both were tracked from one commit (#277, MISSION-026) and never committed again: the committed `board.json` held **two stale `EXECUTING` leases** (MISSION-022 claude-code, MISSION-026 antigravity) that every fresh clone or worktree inherited; the committed `leases.jsonl` had 2 lines while the local one had 69. Main showed both files modified at all times.
- `leases.jsonl` is the predecessor relay's handoff memory (34 CHECKIN events, 18 with handoff notes) and it lived only on one disk, in no commit.
- `manage_coordination.get_project_root()` resolves from the script's own location, so run from a linked worktree it used that worktree's own board (verified with a probe worktree), invisible to every other agent.

## Change
1. `shared_root()`: board and leases paths always resolve to the main checkout, from any worktree (plain non-git directories unchanged).
2. `checkin` mirrors each CHECKIN (without the bulky `usage` block) into the tracked, union-merged `.agent/learning/events.jsonl`; the relay and wakeup read from there too (de-duplicated against the local log).
3. **34 existing CHECKIN events backfilled** from the local `leases.jsonl` into `events.jsonl` (`backfilled_from` field; secret scan passed; 18 carry handoff text, 2026-11-03 to 2026-11-05).
4. `board.json` and `leases.jsonl` untracked (`git rm --cached`), gitignored; the `union` merge rule for `leases.jsonl` dropped; they are no longer listed as shared-generated files.
5. 8 new tests (`test_coord_shared_root.py`; 6 failed before the change).

## Limits
- A different machine does not see this machine's live leases, by design; it never could (the committed board was stale, not live).
- The audit journal of CHECKOUT/HEARTBEAT events stays local; only CHECKINs are durable.
- Pulling this change deletes the tracked copies from a working tree; local copies must be kept aside and restored (done in the main checkout; other clones that edited them must do the same).
