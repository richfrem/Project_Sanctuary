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

**Where the state lives (untracked).** `board.json` and `leases.jsonl` are per-machine runtime state: they are gitignored (they used to be tracked, which left the working tree permanently dirty and shipped two stale `EXECUTING` leases to every fresh clone). The scripts always read and write the copy in the **main checkout**, even when run from a worktree (`manage_coordination.shared_root`), so every agent on the machine sees one board. The durable part, each `CHECKIN` with its handoff note, is also appended (`mirror_checkin_to_events`) to the tracked, union-merged `.agent/learning/events.jsonl` of the checkout the script runs from, NOT the main checkout: run from a worktree, it is an uncommitted edit in that worktree and must be committed with that branch. That file is what the predecessor relay reads after a clone. A different machine does not see this machine's live leases by design.

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

---

## Projected impact, the overlap matrix and parallel dispatch (MISSION-031)

A mission declares what it will touch in its manifest, so the board can lock exactly that without the agent typing `--targets`:

```json
"projected_impact": {
  "paths": ["dir/", "dir/file.py", "dir/**/*.md"],
  "resources": ["local-model-memory"],
  "shared_generated": []
}
```

- **Checkout takes the locks from the manifest.** `manage_coordination.py checkout --mission ID --agent you --create-worktree` needs no `--targets` for a registered mission; `--targets`/`--resources` are added on top. A mission with no declaration and no targets or resources is **refused** (an unlocked lease can never conflict, so it protects nobody); `--allow-unlocked` overrides. The lease records `impact_source` (`manifest`, `manual`, `manifest+manual`, `unlocked`).
- **Conservative overlap.** Files, directories and globs are compared component by component; ambiguity counts as overlap (a mission waits) and never the other way, except that a path that names a file (has an extension) has no children. Extension-less names may be directories.
- **Shared generated files are not locks.** `index.jsonl`, `backlog.md`, `cognitive_ledger.json`, `events.jsonl`, `chronicle/ledger.json` and the coordination files are rewritten by every mission. They are reported after checkout as `regenerate_after_merge`: when merging, regenerate them (`manage_backlog.py compile`, ledger reconciliation), do not hand-merge. They WILL still conflict in git when two branches both rewrite them; the board flags this, it does not remove it.
- **`suggest --count N`** lists up to N open missions that are free of active leases and mutually disjoint, by priority (the incoming agent also sees this in the briefing's Predecessor Relay). **`matrix`** prints which open missions conflict with which; `unregistered` lists missions without a declaration.
- **`verify --mission ID`** compares the declaration with the files the worktree actually changed (committed against `origin/main`, staged, unstaged, untracked). A mission's own manifest and shared generated files are implicitly allowed. Run it before opening a PR; an undeclared edit means the declaration was wrong and the no-conflict guarantee does not hold for that file.
- **Checkout marks the manifest `IN_PROGRESS` in the new worktree's copy**, not in the main tree, so the main tree stays clean and `git pull` keeps working.
- `projected_impact` is validated by `manage_backlog.py validate` (non-empty repo-relative `paths`, list-typed `resources`/`shared_generated`).

