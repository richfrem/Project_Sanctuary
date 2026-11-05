# Living Chronicle - Entry 365

**Title:** MISSION-029 corroborated: idle-reap default cut from 15 to 5 minutes; cold start p95 4.97 s, 3.6 GiB returned
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

Measured 20 stop/start cycles of a managed MLX server (Qwen2.5-Coder-7B 4-bit) on a 16 GB M1 while a human-owned server was resident on :8080 and swap was saturated; each cycle ended with the real reap --only-managed path. 0 failed requests; first token median 4.0 s, p95 4.97 s, max 5.77 s; 3.6 GiB resident per server; system-available memory recovered a median 4.34 GiB. Default idle threshold is now 300 s (was 900 s). Two predictions were wrong: I expected default reap to kill an idle unmanaged decoy (85%) but unmanaged servers report idle_sec 0.0 so it cannot; and I expected swap saturation to hide the memory recovery (40% for >= 3 GiB) but it did not. My first swap-based safety guard tripped on a 2-cycle pilot because macOS grows swap on demand; replaced with memory-free percent and disk. Limits: cold process with warm page cache, one machine and model, 2 s test threshold, 1-token requests. The human's :8080 server was never touched.
