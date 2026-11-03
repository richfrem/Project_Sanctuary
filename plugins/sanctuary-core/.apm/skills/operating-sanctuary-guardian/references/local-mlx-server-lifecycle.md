# On-Demand MLX Local Server Lifecycle & Memory Management

Comprehensive operational runbook for managing local sovereign inference servers (`mlx_lm.server`), background lifecycle control, and automated idle memory reclamation on Apple Silicon (M1/M2/M3/M4) unified memory.

## Contents
- [Executive Purpose & Architecture](#1-executive-purpose--architecture)
- [Server Lifecycle CLI Commands](#2-server-lifecycle-cli-commands)
- [Querying & Verification Runbooks](#3-querying--verification-runbooks)
- [Bootloader Telemetry Integration](#4-bootloader-telemetry-integration)

---

## 1. Executive Purpose & Architecture

On Apple Silicon Macs with 16GB of unified memory, running local LLMs presents an operational trade-off:
- **One-Shot CLI Execution (`mlx_lm.generate`)**: Completely frees RAM upon completion, but incurs an 8–10 second cold-start penalty on every single prompt to re-read and allocate 4.43 GB of weights.
- **Persistent Server (`mlx_lm.server` on `:8080`)**: Delivers sub-3-second responses by keeping model weights resident in unified memory, but risks locking up 4.43 GB of system RAM indefinitely if left abandoned.

**The On-Demand Lifecycle Controller** ([`mlx_server_manager.py`](../scripts/mlx_server_manager.py)) solves this dilemma:
1. **Zero-Latency Availability**: Delivers instant HTTP chat completion responses (`/v1/chat/completions`) when needed.
2. **Safe Port Attachment**: Automatically detects whether a server is already listening on port 8080, safely attaching to existing PIDs without port collisions or duplicate process spawns.
3. **Automated Idle Reclamation**: Every query invocation automatically touches the `last_active` timestamp. A background reaper routine (`reap_if_idle`) terminates servers that remain idle beyond the threshold (default: 15 minutes / 900s), fully releasing the Metal memory buffer back to macOS.

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
  "idle_timeout_sec": 900,
  "rss_mb": 24.81
}
```

### Ensure Ready (Start on Demand)
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server ensure
```
*If inactive, spawns the server in the background and polls `/v1/models` until responsive.*

### Start Server Manually
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server start --port 8080 --idle-timeout 900
```

### Stop Server & Reclaim Memory
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server stop
```
*Sends `SIGTERM`, verifies process termination, removes runtime state files, and logs memory reclaimed.*

### Reap If Idle
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py server reap --timeout 900
```

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
