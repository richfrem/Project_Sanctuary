# Safe Agent Zero: Operational Workflows

**Status**: Draft
**Version**: 1.0

## 1. Boot Sequence (Strict Ordering)

To ensure the "Guard" is always active before the Agent can communicate, the startup sequence is critical.

1.  **Network Up**: Create `frontend-net` (Public) and `control-net` (Internal).
2.  **Guard Up**: Start Nginx Guard. Healthcheck URL must return 200.
3.  **Agent Up**: Start OpenClaw container.
    *   *Check*: Verify it connects to Guard via WebSocket.
4.  **Scout Up**: Start Browser (Scout) container.
    *   *Check*: Agent Zero connects to Scout via CDP.

**Command**: `docker compose up -d` (Depends_on clauses in Docker Compose handle this ordering).

## 2. Standard Operation

### A. Assigning a Task
*   **Via UI**: User types prompt into Sanctuary Frontend.
*   **Flow**: Frontend -> Guard (proxy) -> Agent (ACP `prompt`).

### B. Monitoring
*   **Live Logs**: `docker compose logs -f agent`
*   **Session View**: The Frontend polls the Guard for the Agent's "Thought Stream".

## 3. Emergency Stop (Kill Switch)

If the Agent behaves erratically or gets stuck in a loop:

**Command**: `docker compose stop agent` (Graceful)
**Command**: `docker compose kill agent` (Immediate)

*   **Impact**: Agent process dies instantly. Browser (Scout) remains running but idle.
*   **Recovery**: `docker compose start agent` (Resumes from fresh state, previous context may be lost depending on persistence config).

## 4. Audit & forensics

All agent actions are logged.

*   **Session Logs**: Located in volume `agent-sessions`.
*   **Access**:
    ```bash
    # View latest session log
    docker compose exec agent cat /home/node/.openclaw/sessions/latest/log.jsonl
    ```
*   **Forensic Artifacts**: Screenshots and traces are stored in the session directory.

## 5. Maintenance

### A. Updating Approvals
1.  Edit `config/exec-approvals.json` on host.
2.  Restart Agent: `docker compose restart agent`.
    *   *Note*: Hot-reloading via SIGHUP is planned for V2.

### B. Cleaning Data
To wipe all agent memory/history:

```bash
docker compose down -v
# This removes the 'agent-sessions' volume
```

## 6. Security Verification (Pre-Flight)

Before every run, the "Sanctum" system (CLI or script) should verify:
1.  **Read-Only Root**: Is the container root FS read-only?
2.  **Network Isolation**: specific network drivers active?
3.  **User ID**: Running as non-root (UID != 0)?
