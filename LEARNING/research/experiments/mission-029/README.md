# MISSION-029 experiment: idle-reap policy for the local MLX server

Pre-registered 2026-11-05 by claude-code, before any measurement.

Setup constraints found first: a human-owned MLX server is already listening on :8080 (pid 74300, `managed: false`) and swap is 12.0 of 13.0 GB used. So: all cold starts run on a separate managed port (:8091), the :8080 server is never touched, every start is guarded (abort if free swap < 500 MB), and the "reap must not kill an unmanaged server" falsifier is tested on a decoy listener on :8092, never on :8080.

## Predictions (each can fail)
- P1. p95 of (start command -> first generated token) over 20 cold starts is <= 10 s. Prior 85% (MISSION-027 boot was 1.6-2.3 s with a hot cache; memory pressure here is the risk).
- P2. 0 failed requests across 20 stop/ensure/request cycles. Prior 80%.
- P3. The managed server's resident memory at stop is >= 3 GiB (70%), but system-wide available memory recovering by >= 3 GiB after stop is less likely (40%) because swap is saturated and the human's server is competing.
- P4. `reap` run with its DEFAULT flags on an idle unmanaged decoy kills it (85%), i.e. the hypothesis' fourth falsifier already fires for the current CLI default; with `--only-managed` the decoy survives (95%).
- P5. The swap guard never trips (70%).

## Results
Run: `python3 LEARNING/research/experiments/mission-029/run_cold_starts.py` (raw output: `results.json`). 20 cycles of: managed start on :8091 -> one 1-token chat request -> 3 s idle -> real `reap --only-managed --timeout 2` -> settle. The human-owned :8080 server (pid 74300) was untouched throughout and still running afterwards.

| Prediction | Result |
|---|---|
| P1 p95 start->first token <= 10 s | **Held**: median 4.0 s, p95 4.97 s, max 5.77 s (boot median 1.78 s; two slower boots 2.8 and 3.8 s) |
| P2 0 failed requests in 20 cycles | **Held**: 20/20 ok, 20/20 reaped by the real `reap` path, no leftover listeners |
| P3 managed RSS >= 3 GiB (70%); system-available recovery >= 3 GiB only 40% likely | **Held, and the pessimistic half was wrong**: RSS median 3.6 GiB; system-available memory recovered a median 4.34 GiB after each reap. I expected swap saturation to hide it; it did not. (4.34 > 3.6 because inactive pages elsewhere were also released in the interval; treat 3.6 GiB as the process figure.) |
| P4 default `reap` kills an idle unmanaged decoy (85%) | **Missed**: the decoy survived. An unmanaged server reports `idle_sec: 0.0`, so even `--timeout 1` with default flags leaves it running; `--only-managed` also left it alone (`status: unmanaged`). The hypothesis' fourth falsifier does not fire. The protection is real but partly incidental (no idle tracking for unmanaged servers), so scripts should still pass `--only-managed`. |
| P5 guard never trips (70%) | **Held after a change I made**: my first guard (free swap < 500 MB) tripped in a 2-cycle pilot because macOS grows swap on demand (free swap fell 1.35 GB -> 0.41 GB while the machine was fine). I replaced it with memory-free % >= 20 and disk free >= 20 GB and logged swap only; the final run had no aborts. Swap used rose from 12.0 to 13.0 GB over the experiment. |

Change delivered: the default idle threshold is now 300 s (was 900 s) in `mlx_server_manager.py` and `forge_llm.py`, with two tests and the reference doc updated.

Limits: "cold" is a cold process with the model files in the OS page cache; a first load after eviction or reboot was not measured (MISSION-027's boot numbers were also hot-cache). One machine, one model, one memory situation. The reap was triggered with a 2 s threshold to keep the run short; the 300 s default itself was not waited out 20 times. Request length was 1 token, so this is time to first token, not generation speed.
