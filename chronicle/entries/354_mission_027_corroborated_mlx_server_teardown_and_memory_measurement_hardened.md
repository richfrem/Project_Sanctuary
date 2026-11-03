# Living Chronicle - Entry 354

**Title:** MISSION-027 Corroborated: MLX Server Teardown and Memory Measurement Hardened
**Date:** 2026-11-03
**Author:** claude-code
**Status:** published
**Classification:** internal

---

Project mission MISSION-027 (any CLI agent may take project missions) was executed under Protocol 128 with the flight-board workflow. Checkout: manage_coordination.py checkout --mission MISSION-027 --agent claude-code, targets mlx_server_manager.py, guardian_wakeup.py, the experiment dir and the mission file, resource mlx-server:8099 (spare port; the persistent :8080 server was never touched), worktree .worktrees/mission-027, branch feat/claude-code-mission-027, ttl 240 min; checked in at the end with the PR number. Stage 1: orientation read the code and measured the live server read-only (launcher uvx pid with 4.8 MB RSS, python worker with 10 MB ps RSS but 4,711 MB footprint, one process group). Stage 2: pre-registered predictions and decision rule before any run; RED tests on the old code reproduced 4 of 4 defects (discovery returned a client PID; stop left the worker listening; stop reported stopped while a SIGTERM-ignoring worker survived; status reported the launcher); fix: LISTEN-only discovery, whole-tree teardown with verification (process group only if every member is an MLX process, else tree kill), worker footprint via footprint -p, reaping limited to managed servers and hooked into the Protocol 128 boot and closure. Results: 20/20 stub cycles and 8/8 real-model cycles (port 8099) verified, 0 survivors, 0 client PIDs, warm footprint 4.30-4.36 GB, free memory about 24% -> 52% after stop. Verdict CORROBORATED with recorded deviations: the mission's literal lsof command is invalid on macOS; ps RSS is unusable for MLX; two of my numeric predictions missed (fresh footprint and ps RSS) because the model loads lazily, and the warm measurement point was added after the miss and labelled as an amendment. Stage 3: lessons recorded; replenishment MISSION-029 (cold-start cost vs memory reclamation, idle-reap policy). Stage 4: 19+ tests, cognitive score, this entry.
