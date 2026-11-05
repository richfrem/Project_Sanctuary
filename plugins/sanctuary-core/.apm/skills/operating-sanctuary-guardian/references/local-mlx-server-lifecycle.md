# On-Demand MLX Local Server Lifecycle & Memory Management

Comprehensive operational runbook for managing local sovereign inference servers (`mlx_lm.server`), background lifecycle control, and automated idle memory reclamation on Apple Silicon (M1/M2/M3/M4) unified memory.

## Contents
- [Executive Purpose & Architecture](#1-executive-purpose--architecture)
- [Server Lifecycle CLI Commands](#2-server-lifecycle-cli-commands)
- [Querying & Verification Runbooks](#3-querying--verification-runbooks)
- [Bootloader Telemetry Integration](#4-bootloader-telemetry-integration)
- [Hardening Notes (MISSION-027)](#5-hardening-notes-mission-027)

---

## 1. Executive Purpose & Architecture

On Apple Silicon Macs with 16GB of unified memory, running local LLMs presents an operational trade-off:
- **One-Shot CLI Execution (`mlx_lm.generate`)**: Completely frees RAM upon completion, but incurs an 8–10 second cold-start penalty on every single prompt to re-read and allocate 4.43 GB of weights.
- **Persistent Server (`mlx_lm.server` on `:8080`)**: Delivers sub-3-second responses by keeping model weights resident in unified memory, but risks locking up 4.43 GB of system RAM indefinitely if left abandoned.

**The On-Demand Lifecycle Controller** ([`mlx_server_manager.py`](../scripts/mlx_server_manager.py)) solves this dilemma:
1. **Zero-Latency Availability**: Delivers instant HTTP chat completion responses (`/v1/chat/completions`) when needed.
2. **Safe Port Attachment**: Automatically detects whether a server is already listening on port 8080, safely attaching to existing PIDs without port collisions or duplicate process spawns.
3. **Automated Idle Reclamation**: Every query invocation automatically touches the `last_active` timestamp. A background reaper routine (`reap_if_idle`) terminates servers that remain idle beyond the threshold (default: 5 minutes / 300s; MISSION-029 measured a cold start to first token at p95 4.97 s and 3.6 GiB returned per reap), fully releasing the Metal memory buffer back to macOS.

---

## 2. Server Lifecycle CLI Commands

The lifecycle controller is wired directly into [`forge_llm.py`](../scripts/forge_llm.py) and [`mlx_server_manager.py`](../scripts/mlx_server_manager.py):

### Check Server Status & Memory Telemetry
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server status
```
**Sample Output:**
```json
{
  "is_active": true,
  "pid": 74300,
  "host": "127.0.0.1",
  "port": 8080,
  "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "endpoint": "http://127.0.0.1:8080/v1",
  "uptime_sec": 342.1,
  "idle_sec": 12.4,
  "idle_timeout_sec": 300,
  "launcher_pid": null,
  "managed": false,
  "rss_mb": 10.1,
  "footprint_mb": 4711.0
}
```
`pid` and the memory figures describe the **listening worker**. `footprint_mb` is the number to trust (it includes Metal/unified memory); `rss_mb` is `ps` RSS, which can read a few MB for a worker holding the whole model (see section 5). `managed` is true only for a server this manager started.

### Ensure Ready (Start on Demand)
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server ensure
```
*If inactive, spawns the server in the background and polls `/v1/models` until responsive.*

### Start Server Manually
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server start --port 8080 --idle-timeout 300
```

### Stop Server & Reclaim Memory
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server stop
```
*Tears down the whole server tree (the `uvx` launcher and the worker that listens on the port): `SIGTERM` to the process group when every member is an MLX process, otherwise to the worker, launcher and their children only. It then verifies that nothing is alive and nothing listens on the port, and returns `status: error` with the survivors (not "stopped") if anything remains; `--force` escalates to `SIGKILL`. Reports `reclaimed_footprint_mb`.*

### Reap If Idle
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server reap --timeout 300
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py reap --only-managed
```
`--only-managed` reaps only a server this manager started and leaves an attached or hand-launched persistent daemon alone.

---

## 3. Querying & Verification Runbooks

### Direct Fast-Query with Auto-Start
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py query "In one sentence, what is Project Sanctuary?" --auto-start
```

### Offline Protocol 128 Pre-Completion Verification Gate
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py verify
```
**Sample Output:**
```json
{
  "status": "success",
  "backend": "mlx-server",
  "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "response": "{\n  \"status\": \"PASS\",\n  \"summary\": \"Project Sanctuary git tree is clean, 4 core context files valid, all tests pass. Pre-Completion Verification Gate is satisfied.\"\n}",
  "latency_sec": 2.913,
  "prompt_tokens": 79,
  "completion_tokens": 45,
  "total_tokens": 124
}
```

---

## 4. Bootloader Telemetry Integration

During Holistic Wakeup (`guardian_wakeup.py --mode HOLISTIC`), the local server daemon status is automatically audited under `runtime_checks`:
```json
{
  "component": "Local MLX Server Daemon (:8080)",
  "status": "ACTIVE",
  "description": "Responding at http://127.0.0.1:8080/v1 (zero-latency ready)"
}
```
*(If inactive, returns status `STANDBY` without throwing an error).*

---

## 5. Hardening Notes (MISSION-027)

Measured on an M1 with 16 GB and `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (details: `LEARNING/research/experiments/mission-027/`).

- **Port discovery lists LISTEN sockets only.** The correct macOS command is `lsof -nP -iTCP:<port> -sTCP:LISTEN -t`. (`lsof -ti -sTCP:LISTEN :<port>` errors, and plain `lsof -ti :<port>` also returns client connections, so it could return a client's PID.)
- **Process shape.** `uv tool uvx ...` (launcher, tiny) starts a python worker (listener, the model). Both share one process group. Signalling only the launcher left the worker listening and the old check reported "stopped" anyway.
- **Memory.** Use `footprint -p <pid>` (the worker's physical footprint). `ps` RSS depends on paging and compression and read about 10 MB for a worker with 4.7 GB resident. The footprint of a freshly started server varies (about 0.5-3.3 GB) because weights load lazily; after one small generation it was 4.30-4.36 GB (4.71 GB on a server that had been serving for hours).
- **Reclamation.** Stopping returned about 4.3 GB: system free memory went from about 24% (server running) to about 52%.
- **Reaping policy.** Idle reaping applies only to servers this manager started (`managed`). Holistic wakeup calls `reap_if_idle(only_managed=True)` at boot, and the closure step of Protocol 128 should run `mlx_server_manager.py reap --only-managed` (or `stop` for a server you started). A persistent daemon you launched yourself is never reaped.
- **Not covered.** macOS SIP permission errors when signalling are not handled (none occurred); processes owned by another user cannot be signalled.

## Idle-reap policy (MISSION-029)

Measured on a 16 GB M1 with a human-owned server already resident and swap saturated: 20 stop/start cycles of a managed
`Qwen2.5-Coder-7B-Instruct-4bit` server, each ended by the real `reap` path. 0 failed requests; boot median 1.8 s; first
generated token median 4.0 s, p95 4.97 s, max 5.77 s; 3.6 GiB resident per server; system-available memory recovered a
median 4.34 GiB after a reap. So the default idle threshold is 300 s: restarting costs about 4 s, a resident idle server costs 3.6 GiB.

- "Cold" here means a cold process with the model files still in the OS page cache. A first load after the files are evicted
  (or a reboot) was not measured and will be slower.
- A server the manager did not start reports `idle_sec: 0.0` and `managed: false`, so even `reap --timeout 1` with default flags
  leaves it running. Still pass `--only-managed` in scripts so that does not depend on idle tracking.
- Evidence: `LEARNING/research/experiments/mission-029/`.
