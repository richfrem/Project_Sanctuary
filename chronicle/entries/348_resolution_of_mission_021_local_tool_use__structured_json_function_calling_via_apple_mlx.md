# Living Chronicle - Entry 348

**Title:** Resolution of MISSION-021: Local Tool Use & Structured JSON Function-Calling via Apple MLX
**Date:** 2026-11-02
**Author:** Antigravity & Richard
**Status:** canonical
**Classification:** milestone

---

Resolved MISSION-021 by engineering native tool-calling and structured JSON schema extraction into forge_llm.py, backed by the 4-bit sovereign Qwen2.5-Coder model running on Apple Silicon unified memory via Apple MLX. Implemented unified tag parsing supporting <tools>, <tool_call>, <function_call>, and <function name='...' arguments='...'/> XML formats. Authored test_local_tool_calling.py containing 12 unit/integration tests and an automated empirical benchmark runner; achieved 100.0% (5/5) benchmark pass rate with 0.0% hallucination rate on conversational controls and 5.80s average latency. Replenished pipeline with MISSION-022 for multi-turn local agent execution.
