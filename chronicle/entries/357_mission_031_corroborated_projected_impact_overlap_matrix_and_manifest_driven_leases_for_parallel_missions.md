# Living Chronicle - Entry 357

**Title:** MISSION-031 Corroborated: Projected Impact, Overlap Matrix and Manifest-Driven Leases for Parallel Missions
**Date:** 2026-11-03
**Author:** claude-code
**Status:** published
**Classification:** internal

---

Project mission MISSION-031 (highest open priority, 400) executed under Protocol 128 and the flight-board workflow while the human slept: lease MISSION-031 for claude-code, worktree .worktrees/mission-031, branch feat/claude-code-mission-031, with a wide target list covering the coordination scripts, relay, tests, backlog manager, flight-board reference and the mission and experiment files. Pre-registered before any code: 5 predictions and a decision rule. Defect reproduced on the OLD code loaded from git: checkout without --targets takes no locks (3/3 RED tests) and a second overlapping mission checks out silently in 284 of 284 overlapping random pairs. Built: impact.py (conservative component-wise overlap, leaf rule, shared generated files never locked), projected_impact in mission manifests (validated by manage_backlog validate), checkout that takes its locks from the manifest and refuses an undeclared lock-free lease, matrix/suggest/verify commands, the dispatchable line in the Predecessor Relay, the IN_PROGRESS marker written into the new worktree not the main tree, non-zero exit codes for blocked checkouts in JSON mode, and 20 backfilled declarations (a local-model-memory resource models the one-model-at-a-time limit). Results: 500 pairs 0 false negatives, 6.9% false positives; registry p95 5.6 ms; real git 60 branch pairs 0 unpredicted conflicts, verify flagged 20/20 planted undeclared edits; suggest returns 3 disjoint missions, 13 can run in parallel. Honest boundary: index.jsonl conflicted in 60/60 merges because every mission regenerates it; the board flags it, it does not eliminate it. Dogfooding verify on this mission caught 1 of 33 files outside its own declaration (test_relay.py, also declared by MISSION-032). Verdict CORROBORATED. Replenishment MISSION-034: merge-time regeneration of the shared generated files. MISSION-033 (orchestrator) depends on this work.
