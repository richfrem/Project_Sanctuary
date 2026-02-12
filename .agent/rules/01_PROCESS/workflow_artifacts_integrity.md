---
trigger: always_on
---

# 🛡️ Workflow Artifacts Integrity Policy

**Effective Date**: 2026-02-12
**Related Constitution Articles**: I (Hybrid Workflow), III (Zero Trust)

## Core Mandate: Tool-Generated Truth
The Agent MUST NOT simulate work or manually create process artifacts that are controlled by CLI tools.
**If a command exists to generate a file, YOU MUST USE IT.**

### 1. Spec Kitty Lifecycle
The following files are **READ-ONLY** for manual editing by the Agent. They MUST be generated/updated via CLI:

| Artifact | Mandatory Command | Forbidden Action |
|:---|:---|:---|
| `spec.md` | `/spec-kitty.specify` | Manually writing a spec file |
| `plan.md` | `/spec-kitty.plan` | Manually scaffolding a plan |
| `tasks.md` | `/spec-kitty.tasks` | Manually typing a task list |
| `tasks/WP-*.md` | `/spec-kitty.tasks` | Manually creating prompt files |
| Status Updates | `spec-kitty agent tasks mark-status` | Manually editing `[x]` in markdown |

**Violation**: Creating these files via `write_to_file` is a critical process failure.

### 2. Checklist Verify-Before-Trust
The Agent MUST NOT mark a checklist item as complete (`[x]`) unless:
1.  The specific tool command for that step has been executed.
2.  The tool output has been received and verified as SUCCESS.
3.  The artifact exists on disk (verified via `ls` or `view_file` if uncertain).

**Simulation is Lying**: Marking a task `[x]` based on "intent" or "mental model" is prohibited.

### 3. Git Sovereignty (Human Gate)
- **NEVER** set `SafeToAutoRun: true` for `git push`.
- **NEVER** push directly to `main` (Protected Branch).
- **ALWAYS** use a feature branch (`feat/...`, `fix/...`, `docs/...`).
- **ALWAYS** wait for explicit user approval for any push.

### 4. Worktree Hygiene
- **Never** manually create directories inside `.worktrees/`.
- **Always** use `spec-kitty implement` (or `run_workflow.py`) to manage worktrees.
- **Cleanup**: Delete worktrees only via `git worktree remove` or approved cleanup scripts.
