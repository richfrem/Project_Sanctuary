# Living Chronicle - Entry 378

**Title:** MISSION-052 created: the four-worker orchestration batch as a startable mission, with a brief for a clean session
**Date:** 2026-11-05
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

At the human's request the four-worker orchestration instruction (Codex A, Codex B, agy, local Qwen under one main-thread orchestrator) became a new mission so a clean session can run it. MISSION-052 is separate from MISSION-033 on purpose: 033 specifies a three-worker pilot of a wrapper skill that does not exist, while this batch uses only the existing stack (flight board, merge_prep, cli-agent, usage_hook delegated), so it is the configuration-alone arm of 033's own falsification test and makes no claim about 033's thresholds. Created through the checklist: related scan (no duplicate), a falsifiable hypothesis with six measurable parts and explicit falsifiers, linked both ways to 14 missions (033, 026, 031, 034, 035, 025, 022, 021, 028, 036, 037, 038, 045, 051), schema-valid, narrow own impact (no broad orchestration lease), human gate required. The brief keeps the human's instruction verbatim with mission paths corrected, and adds what changed since it was drafted: 035 FALSIFIED (budgets are caps, not predictions), 038 not implemented (supervised recovery), the relay and fuel gauge in the boot hook, the board is untracked per-machine state, agy at 11% (constrained), a human-managed MLX server resident with swap about 92% used, Qwen's sandbox is read-only so it can only extract or classify, the held-out generator is hidden by name, and a known gap: the current recorder cannot separate two concurrent Codex sessions. Preregistered predictions P1-P8 with priors are in the README. A test checks that every path and mission link in the brief resolves. Score 5.0 (leverage 5, depth 4, cost 1.0, friction 4): it does not outrank cheaper missions, so a session must be pointed at it explicitly.
