# Living Chronicle - Entry 347

**Title:** Resolution of MISSION-019: On-Demand MLX Local Server Lifecycle Controller & Zero-Latency Local Calling
**Date:** 2026-11-02
**Author:** Antigravity & Richard
**Status:** canonical
**Classification:** internal

---

Implemented on-demand MLX local server lifecycle controller in operating-sanctuary-guardian (mlx_server_manager.py). Provides background daemon spawning, attached-instance recognition (preventing port collisions), memory telemetry (RSS MB), and idle memory reclamation (reap_if_idle). Wired into forge_llm.py (server start/stop/status/ensure/reap subcommands and query --auto-start) and guardian_wakeup.py HOLISTIC boot check. Added 7 unit tests (100% pass) in test_mlx_server_manager.py. Replenished research pipeline with MISSION-021 for local tool use and structured JSON function-calling.
