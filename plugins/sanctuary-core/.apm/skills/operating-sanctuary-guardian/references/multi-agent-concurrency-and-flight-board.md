# Multi-Agent Concurrency Control & Flight Board Protocol

## Contents
- [Purpose](#purpose)
- [Architecture & Components](#architecture--components)
- [CLI Reference](#cli-reference)
- [Upstream Lineage & Doctrine](#upstream-lineage--doctrine)

## Purpose
Provides decentralized, zero-daemon concurrency control, optimistic file leases, and git worktree isolation for heterogeneous autonomous AI agents (Antigravity/Gemini, Claude Code, OpenAI Codex, and local MLX models) operating on the Project Sanctuary repository.

---

## Architecture & Components

```
                ┌───────────────────────────────────────────────┐
                │          Agent Coordination Flight Board       │
                │        (.agent/coordination/board.json)       │
                └──────────────────────┬────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     [Antigravity / AGY]        [Claude Code]              [OpenAI Codex]
            │                          │                          │
            ▼                          ▼                          ▼
   Worktree Isolation         Worktree Isolation         Worktree Isolation
  .worktrees/agy-026/       .worktrees/claude-022/     .worktrees/codex-027/
            │                          │                          │
            ▼                          ▼                          ▼
  Feature Branch PR          Feature Branch PR          Feature Branch PR
  (feat/agy-mission-026)    (feat/claude-mission-022)   (feat/codex-mission-027)
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                             Squash-Merge to main
                               (origin/main)
```

### 1. Zero-Daemon Coordination Registry
- **Board State** (`.agent/coordination/board.json`): Active leases, holding agents, locked target paths, exclusive resources (e.g. `:8080`), worktree locations, and TTLs.
- **Audit Event Journal** (`.agent/coordination/leases.jsonl`): Immutable append-only log of every `CHECKOUT`, `CHECKIN`, `HEARTBEAT`, and `EXPIRE` event.

### 2. Worktree Isolation Standard
- Concurrent agents NEVER mutate the root working tree simultaneously.
- When an agent checks out a non-trivial mission, it operates within an isolated git worktree:
  ```bash
  python3 manage_coordination.py checkout --mission MISSION-026 --agent antigravity --create-worktree
  ```
- This branches cleanly from `origin/main` into `.worktrees/<agent>-<mission>/`, ensuring uncommitted experimental files never bleed across agent workspaces.

### 3. Collision Detection & Optimistic Locking
- Before acquiring a lease, `manage_coordination.py` evaluates:
  1. **Path Overlap**: Ensures no other active, unexpired lease locks the target files or parent directories.
  2. **Resource Overlap**: Ensures exclusive system resources (e.g. `mlx-server:8080`, serial port, or local container) are not concurrently contested.
- If a conflict is detected, checkout is hard-blocked with the holding agent's details:
  ```
  ❌ CHECKOUT BLOCKED by conflict:
  [
    {
      "holding_agent": "claude-code",
      "holding_mission": "MISSION-022",
      "conflicting_targets": ["forge_llm.py"],
      "worktree": "main",
      "expires_in_sec": 3240.5
    }
  ]
  ```

---

## CLI Reference

### 1. Check Out a Mission
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py checkout \
  --mission MISSION-026 \
  --agent antigravity \
  --targets "scripts/manage_coordination.py" \
  --resources "mlx-server:8080" \
  --worktree ".worktrees/mission-026" \
  --ttl 60
```

### 2. Inspect Flight Board
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py board
```

### 3. Send Heartbeat
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py heartbeat \
  --mission MISSION-026 \
  --agent antigravity \
  --extend 30
```

### 4. Check In (Release Locks)
```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py checkin \
  --mission MISSION-026 \
  --agent antigravity \
  --status COMPLETED \
  --pr 278 \
  --handoff "Did X; decided Y over Z because ...; do NOT re-run benchmark B (see experiments/mission-026); next: run T-9 before merging."
```
`--handoff` is optional (capped at 600 characters, whitespace collapsed). It is stored on the CHECKIN event and shown to the next agent, whatever its vendor, in the **Predecessor Relay** section of the boot briefing (`relay.py show`). Write what the successor cannot read off the code: decisions and why, results not to repeat, and the next step.

---

## Upstream Lineage & Doctrine
This protocol synthesizes battle-tested patterns from `github.com/richfrem/agent-plugins-skills`:
- **`running-control-plane`**: Pre-worktree dirty state reconciliation (`worktree-reconciliation-and-multi-worktree-practices.md`), clean base branching from `origin/main`, and non-overlapping path ownership.
- **`orchestrating-agent-work`**: Confinement and redirection, isolated worktree provisioning (`.worktrees/task-<id>`), and supervisor verification gates.
