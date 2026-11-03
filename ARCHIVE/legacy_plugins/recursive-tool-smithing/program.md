# Tool-Smithing Program: Recursive Self-Growth

## Identity

This program governs the autonomous self-improvement loop for the recursive-tool-smithing plugin.
The agent reading this is both the author of this program and the executor of it.

**Home repo:** `Project_Sanctuary` - this forge exists to evolve the Sanctuary Guardian.
**Companion plugins:** `sanctuary-guardian` (identity + memory), `chronicle-manager` (living history),
`protocol-manager` (governance), `rsvp-speed-reader` (knowledge ingestion).

## Goal

Minimize friction in the agent's own capability gaps over time.

Operationally: maximize the number of repeatable tasks the agent can execute in under 30 seconds
without requiring ad-hoc script writing or manual research.

Metric: `gap_count` (open entries in gaps.json with status=open) + `avg_task_latency` (seconds).
Track both in results.tsv.

## Setup (Iteration Mode Entry)

1. `git checkout -b autoresearch/<run-tag>` (e.g., `autoresearch/mar14`)
2. Read: this program.md, scripts/toolbox/ inventory, gaps.json
3. Run: `python3 scripts/toolsmith_wakeup.py --mode TELEMETRY`
4. Initialize results.tsv header if not present (do NOT commit - leave untracked)
5. Confirm and LOOP FOREVER

## Scope

### Files agents CAN modify
- `scripts/toolbox/*.py` - the growth library
- `gaps.json` - gap registry (update after each tool keeps)
- `evals/results.tsv` - experiment ledger

### Files agents CANNOT modify (immutable)
- `scripts/sandbox_runner.py` - safety perimeter
- `scripts/evolution_metrics.py` - fitness scoring oracle
- `skills/recursive-tool-smithing/references/safety-sandbox.md` - compliance gate
- `evals/evals.json` - test corpus

## Generation Backend

**Copilot CLI ONLY.** GPT-5.0 mini via GitHub Copilot account. Free, zero tokens.
No claude-cli. No gemini-cli. No direct API calls.

Pattern:
```bash
cat scripts/toolbox/closest_existing.py | \
  copilot -p "Write a new Python script to solve: <gap>. Match the style. Include: file header, type hints, docstring, __main__ test block, argparse CLI." \
  > /tmp/<gap-slug>.py
```

## Keep Threshold

A tool earns `keep` when:
- fitness_depth >= 2.0 AND fitness_scope >= 2.5 (combined >= 4.5)
- Solves the stated gap (verified by running against real data from this plugin)
- Does NOT add subprocess/network/eval violations

Simplicity criterion:
- Same result with fewer lines = keep
- Small improvement + adds 20+ lines = probably discard
- Delete code to fix a bug = always keep

## Approved External Packages

stdlib only. If a gap requires third-party packages, escalate to human gate.

## Experiment Loop (NEVER STOP)

```
LOOP FOREVER:
1. python3 scripts/toolsmith_wakeup.py --mode TELEMETRY > .boot_digest.md
2. python3 scripts/gap_scanner.py --toolbox-path scripts/toolbox --gap-log gaps.json
   → Pick highest-priority open gap
3. Generate candidate via Copilot CLI -> /tmp/<gap-slug>.py
4. git commit -m "wip(toolbox): candidate <gap-slug>"
5. python3 scripts/sandbox_runner.py --script /tmp/<gap-slug>.py > /tmp/run.log 2>&1
6. python3 scripts/evolution_metrics.py fitness --file /tmp/<gap-slug>.py
7. Review run.log + fitness score
   IF keep: cp /tmp/<gap-slug>.py scripts/toolbox/, amend commit, mark gap resolved
   IF discard/crash: git reset HEAD~1, delete /tmp/<gap-slug>.py, log crash
8. Append row to evals/results.tsv
9. Every 5 keeps: run continuous-skill-optimizer on SKILL.md
   Every 10 keeps: run session_chronicle.py both
```

## Current Priority Gaps

See gaps.json. Top priority: automation of gap detection itself.

## Growth Milestones

| Milestone | Target |
|-----------|--------|
| Toolbox: 5 tools | First stable library |
| Toolbox: 10 tools | Self-sustaining loop |
| Toolbox: 20 tools | Cross-plugin evolution |
| Chronicle: 30 daily digests | Measurable growth trajectory |

## Orchestrator Integration

When routing is ambiguous, use the `orchestrator` skill (installed at `.claude/skills/orchestrator/`) to decide:
- research/doc task -> learning-loop
- architecture/security -> red-team-review
- parallel script generation -> agent-swarm
- single tool generation -> dual-loop (this loop)

## Sanctuary Growth Targets

Beyond toolbox scripts, the forge should also target:
- Improving `sanctuary-guardian` scripts (guardian_wakeup, evolution_metrics, learning_debrief)
- Generating tools that feed guardian_manifest.json with richer context
- Building bridges to chronicle-manager for auto-journaling of keeps

_Last updated: 2026-03-14 by recursive-tool-smithing agent_
