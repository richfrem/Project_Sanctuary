# Living Chronicle - Entry 351

**Title:** Sovereign Multi-Turn Local Agent Loop on Apple MLX: MISSION-022 Falsified
**Date:** 2026-11-03
**Author:** claude-code
**Status:** published
**Classification:** internal

---

MISSION-022 asked whether Qwen2.5-Coder-7B (4-bit, MLX, offline) can complete >=85% of bounded repository discovery tasks in a plan->tool->observe->final loop. Built local_agent_loop.py (forge_llm.py loop): sandboxed read-only tools, repetition/turn/context guards, intervention metrics. Early 5-task runs (1/5, then 4/5 after tuning on the same tasks) were discarded as evidence after agy (Gemini 4 Argon) review. Pre-registered held-out evaluation on 30 generated tasks with computed answers: run 4 = 83.3% (inconclusive), run 5 = 73.3% after a tool fix (below the 75% falsification line). Verdict FALSIFIED. Scalar single-hop 100%; multi-hop locate 40-60%. Loop-health signals catch 10/13 failures but miss silent errors, spawning MISSION-028 (signal-triggered escalation to Haiku, child of MISSION-007). Lessons: scaffolding is load-bearing; error hints can mislead small models; the originating task script presupposed CORROBORATED and was not followed because the data disagreed. Run via coordination with agy on MISSION-026 worktrees.
