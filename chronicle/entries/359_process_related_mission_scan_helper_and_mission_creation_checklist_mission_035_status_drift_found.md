# Living Chronicle - Entry 359

**Title:** Process: related-mission scan helper and mission-creation checklist; MISSION-035 status drift found
**Date:** 2026-11-04
**Author:** claude-code
**Status:** CANONICAL
**Classification:** internal

---

Added manage_backlog.py 'related' (lexical TF-IDF + tag overlap, link-state per candidate, --id or --text for undrafted ideas) and a required mission-creation checklist (agent-native-backlog.md 4a) pointed to from AGENTS.md item 7 and Protocol 128. Pre-registered recall@8 >= 0.8 on MISSION-035's known related set was MISSED (3/5; 031 is a mechanism-level relation lexical search cannot see; 028 at rank 9); noise and runtime predictions met; default top raised to 10 post-hoc (recall@10 4/5), disclosed. Also found and fixed: MISSION-035's PR 299 shipped the flight-board IN_PROGRESS marker to main (restored to PROPOSED; defect captured for the board code). Earlier in the day the 035 PR was left unmerged and worked in a plain worktree without a lease; both corrected.
