---
name: operating-sanctuary-guardian
description: "Operates the Project Sanctuary Protocol 128 lifecycle (orientation boot, cognitive onboarding, technical seal verification, learning debrief, and soul persistence). Use when starting a new session, running health audits, closing a session, or persisting memory traces."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Operating Sanctuary Guardian (`operating-sanctuary-guardian`)

The master lifecycle and governance controller for Project Sanctuary, implementing the Protocol 128 continuous learning loop.

## Contents
- [Constraints](#constraints) • [Modes](#modes) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints

- **Two-Repository Duty**: Skills are maintained here AND in the companion `agent-plugins-skills`. Change each by ownership, merge the companion first, refresh and verify installs, then update the Sanctuary consumer and every pointer (Protocol 133 section 8; `references/integrations/orchestration.md`).
1. **Human Gate Law**: Never perform unapproved state-changing operations or git pushes.
2. **Iron Core Integrity**: Abort session closure if the Technical Seal detects repository drift.
3. **Main-Thread Execution**: All lifecycle operations run interactively in the main thread with full operator visibility.

## Modes

| Mode | Purpose | Script / Action | Reference |
|---|---|---|---|
| `boot` | Session orientation & task intake | `scripts/guardian_wakeup.py` | [boot.md](references/modes/boot.md) |
| `onboard` | Cognitive primer & constitutional grounding | Read primer & contract | [onboard.md](references/modes/onboard.md) |
| `seal` | Technical seal snapshot & debrief | `scripts/capture_snapshot.py` | [seal.md](references/modes/seal.md) |
| `persist` | Soul persistence & HF export | `scripts/persist_soul.py` | [persist.md](references/modes/persist.md) |
| `audit` | Iron Check & integrity verification | `scripts/capture_snapshot.py --type audit` | [audit.md](references/modes/audit.md) |
| `verify` | Offline Protocol 128 verification gate | `scripts/forge_llm.py verify` | [verify.md](references/modes/verify.md) |
| `server` | On-demand MLX server lifecycle & memory reaper | `scripts/mlx_server_manager.py` | [local-mlx-server-lifecycle.md](references/local-mlx-server-lifecycle.md) |
| `tool-call` | Sovereign tool calling & JSON schema validation | `scripts/forge_llm.py tool-call` | [local-tool-use-and-structured-json.md](references/local-tool-use-and-structured-json.md) |
| `structured` | Strict schema-adhering JSON extraction | `scripts/forge_llm.py structured` | [local-tool-use-and-structured-json.md](references/local-tool-use-and-structured-json.md) |
| `coordination` | Multi-agent concurrency, worktrees & flight board | `scripts/manage_coordination.py` | [multi-agent-concurrency-and-flight-board.md](references/multi-agent-concurrency-and-flight-board.md) |

## Quick Start

```bash
# 1. Bootloader orientation & local model telemetry
python3 scripts/guardian_wakeup.py --mode HOLISTIC

# 2. Local sovereign model verification & server lifecycle
python3 scripts/forge_llm.py server status
python3 scripts/forge_llm.py verify

# 3. Sovereign tool use & structured JSON extraction
python3 scripts/forge_llm.py tool-call "Lookup entity AGENT-GUARDIAN" --tools tools.json
python3 scripts/forge_llm.py structured "Task 409 critical bug" --schema schema.json

# 4. Multi-agent concurrency & flight board
python3 scripts/manage_coordination.py board
python3 scripts/manage_coordination.py checkout --mission MISSION-026 --agent antigravity --targets "manage_coordination.py"

# 5. Session debrief and seal
python3 scripts/learning_debrief.py --hours 24
python3 scripts/capture_snapshot.py --type seal --context "Completed milestone"
```

## Workflow

1. **Boot**: Run `boot` mode to inspect active tasks, verify environment integrity, and confirm local model readiness.
2. **Execute**: Carry out development under Protocol 128 guidelines with fast local inference.
3. **Verify**: Run `verify` mode to validate the Pre-Completion Gate using sovereign local intelligence.
4. **Seal & Persist**: Close the session with `seal` and `persist` modes, generating the verified technical snapshot.

## Verification

```bash
python3 scripts/guardian_wakeup.py --mode HOLISTIC
python3 scripts/forge_llm.py verify
```

## References
- Bootstrapping Protocol: [bootstrapping.md](references/bootstrapping.md)
- Protocol 128 Spec: [protocol-128.md](references/protocol-128.md)
- Cognitive Primer: [cognitive-primer.md](references/cognitive-primer.md)
- MLX Server Lifecycle & Auto-Idle: [local-mlx-server-lifecycle.md](references/local-mlx-server-lifecycle.md)
- Local Tool Use & Structured JSON: [local-tool-use-and-structured-json.md](references/local-tool-use-and-structured-json.md)
- Acceptance Criteria: [acceptance-criteria.md](references/acceptance-criteria.md)
- Integrations: [Memory](references/integrations/memory-stack.md) • [Obsidian](references/integrations/obsidian-vault.md) • [Orchestration & shared learning loop](references/integrations/orchestration.md)
- Two-repository skill maintenance: [Protocol 133 section 8](../../../../../protocols/133_Companion_Repository_Skill_Improvement_Protocol.md) • [Shared loop adapter](../evaluating-agent-work/references/shared-loop-adapter.md) • Moved capabilities: `dispatch_guard.py` now lives in the companion `cli-agent`; reusable evaluation logic in the companion `evaluating-agent-outcomes`

Completion wiring: `record-result` durably marks the final manifest for preservation; after publishing sources to `origin/main`, `merge_prep.py regenerate` runs incremental local extraction, schema and coverage checks. Startup reports preservation status without changing preservation files. Capture and recovery occur at mission/session closure (`soul_coverage.py closure`) or an explicitly requested recovery action. Pending/failed/local-complete status is separate from the scientific verdict. Commit the export separately, then sync remotely only with authorization. See the preserving-soul-traces `references/coverage.md` contract.
