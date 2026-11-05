# Living Chronicle - Entry 370

**Title:** Flight-board files untracked: shared root from any worktree, handoffs durable in events.jsonl, 34 check-ins rescued
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

The human asked whether board.json and leases.jsonl should be gitignored and delegated the choice. Evidence: both were tracked from one commit (#277) and never updated; the committed board held two stale EXECUTING leases that every fresh clone inherited; the committed leases log had 2 lines against 69 locally, so the predecessor relay's handoff memory (34 check-ins, 18 with notes) lived on one disk only; and manage_coordination resolved its root from the script location so a script run from a worktree used that worktree's own empty board. Changes: shared_root() makes every worktree use the main checkout's board; checkin mirrors CHECKIN events (minus usage) into the tracked union-merged events.jsonl and the relay and wakeup read from there; the 34 existing check-ins were backfilled with a secret scan; both files are untracked and gitignored and the leases union rule dropped; 8 tests, 6 red before. Limits: other machines do not see live leases (they never could); only check-ins are durable; a pull deletes the tracked copies so local copies must be restored aside. Correction noted: the files came from MISSION-026 (agy), though this session has written them most.
