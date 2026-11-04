# Living Chronicle - Entry 356

**Title:** MISSION-025: Predecessor Relay Shipped, Hypothesis Falsified at 68% with a 7B Successor
**Date:** 2026-11-03
**Author:** claude-code
**Status:** published
**Classification:** internal

---

Project mission MISSION-025 (cross-CLI continuity) executed under Protocol 128 with the flight-board workflow: lease MISSION-025 for claude-code, worktree .worktrees/mission-025, branch feat/claude-code-mission-025, targets relay/wakeup/coordination scripts, tests, START_HERE, AGENTS, Protocol 128 and the experiment directory, resource mlx-server:8080 for the evaluation; agy took MISSION-024 in parallel in its own worktree and merged first with no file overlap. Built: relay.py (Predecessor Relay: last handovers with vendor, established results with evidence pointers, open work, handoff notes), its integration into the wakeup briefing (agy's SessionStart hook now delivers it with zero turns), and manage_coordination checkin --handoff so an outgoing agent leaves a note for the next one. Value test, pre-registered before anything was built: same 22 questions, same loop and tools, local Qwen 7B as a different-vendor incoming agent; control briefing frozen from the pre-feature code. Control 22.7% correct (median 5 turns, 2.5 tool calls); with the relay 68.2% (median 2 turns, 1.27 tool calls, half the time): +45.5 points. Pre-registered bar (>=90%) not met, so the verdict is FALSIFIED by my own rule; every one of the 7 misses had the answer present in the relay (the weak agent ignored the briefing, copied a format placeholder, or confused ids). Process lessons: the first control run was discarded because the 7B wrote <search_text>query</search_text> and the loop could not parse it (a syntax artifact that would have inflated the value, now accepted by the shared loop); a 180-character display cap hid the next step of a handoff (now 400 for recent notes, section capped at 3,500 characters); only 2 of 10 resolved missions cite a file path as evidence. Replenishment MISSION-032: frontier incoming agents and relay-first placement on fresh questions. Chronicle 355 and missions 030/031 belong to agy's MISSION-024 work.
