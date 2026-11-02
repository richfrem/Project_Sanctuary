# Evolution Loop Specification (Dual-Mode Meta-Skill)

## Contents
- [Mode 1: Bootstrap](#mode-1-bootstrap) • [Mode 2: Iteration](#mode-2-iteration-gap-fill-loop) • [Program Template](#programmd-template-karpathy-style) • [Single-Hypothesis Discipline](#single-hypothesis-discipline)

## Mode 1: Bootstrap

Triggered when: `program.md` does not exist OR `scripts/toolbox/` is empty.

Steps:
1. Run Discovery Phase interview (see SKILL.md).
2. Create `program.md` from answers.
3. Create `scripts/toolbox/` directory.
4. Create `scripts/toolbox/requirements.txt` (stdlib only to start).
5. Create `evals/evals.json` baseline (if not provided).
6. Run one baseline measurement (gap count = N, avg task time = T).
7. Log baseline to `evals/results.tsv` as iteration 0.
8. Switch to Iteration Mode.

## Mode 2: Iteration (Gap-Fill Loop)

Triggered when: `program.md` exists AND `scripts/toolbox/` has >= 1 script.

Each cycle:
```
Load program.md strategy
  |
  v
Attempt target task
  |
  +-- All tools available? --> Execute normally --> Log SUCCESS
  |
  +-- Capability gap found? --> Log GAP entry
        |
        v
      Generate candidate script (one hypothesis per iteration)
        |
        v
      Sandbox validation (5 min, constrained)
        |
        +-- Pass + improvement? --> keep + commit + ledger
        +-- Pass, no improvement? --> discard + log + next hypothesis
        +-- Crash / timeout? --> crash + log failure + narrow scope
```

## program.md Template (Karpathy-style)

This mirrors the structure of Karpathy's autoresearch `program.md`.
Humans program this file; the agent reads and follows it.

```markdown
# Tool-Smithing Program: <Project Name>

## Setup

1. Agree on a run tag (e.g., `mar14`). Branch: `autoresearch/<tag>` must be fresh.
2. `git checkout -b autoresearch/<tag>`
3. Read in-scope files: this program.md, scripts/toolbox/ inventory, gaps.json
4. Initialize `results.tsv` with header row (do NOT commit - leave untracked)
5. Run toolsmith_wakeup.py to confirm system state
6. Confirm and go

## Goal

Minimize: <metric> for <domain>
Example: "Minimize average query latency (ms) for semiconductor patent research tasks"

## Scope

What this toolbox covers:
- <task type 1>
- <task type 2>

Files agents CAN modify: `scripts/toolbox/*.py`
Files agents CANNOT modify: `scripts/sandbox_runner.py`, `scripts/evolution_metrics.py`, `evals/evals.json`

## Keep Threshold

A new or modified script is worth keeping when:
- Metric improves by >= <N>% (e.g., 10%) AND
- Code is not significantly more complex

Simplicity criterion (from Karpathy):
- Small improvement + adds 20+ lines = probably NOT worth it
- Same/better performance by DELETING code = definitely keep
- All else equal: simpler is better

## Generation Backend

Use Copilot CLI for M1 Mac efficiency:
```bash
cat scripts/toolbox/similar_tool.py | copilot -p "Write a new Python tool to <gap description>. Follow the same patterns. Include type hints, docstring, __main__ test block." > scripts/toolbox/<gap-slug>.py
```

## Approved External Packages

(stdlib only unless listed here)
- <package>=<version>

## Experiment Loop

LOOP FOREVER (do not pause mid-loop):
1. `python3 scripts/toolsmith_wakeup.py > .boot_digest.md`
2. `python3 scripts/gap_scanner.py --toolbox-path scripts/toolbox --gap-log gaps.json`
3. Generate candidate (Copilot CLI or direct)
4. `git commit -m "wip(toolbox): candidate <gap-slug>"`
5. `python3 scripts/sandbox_runner.py --script scripts/toolbox/<gap-slug>.py > run.log 2>&1`
6. `python3 scripts/evolution_metrics.py fitness --file scripts/toolbox/<gap-slug>.py`
7. If keep: amend commit with proper message, advance branch
8. If discard/crash: `git reset HEAD~1`, delete file, log to results.tsv
9. Append row to results.tsv (NOT committed)

## Results TSV Format

Columns (tab-separated, NOT comma-separated):
```
commit  metric_before  metric_after  memory_mb  fitness_depth  fitness_scope  status  description
```

Status values: `keep`, `discard`, `crash`

## Current Gaps (agent updates this table each cycle)

| Priority | Slug | Description | Status |
|----------|------|-------------|--------|
| - | - | - | open |
```

## Single-Hypothesis Discipline

Each iteration changes EXACTLY ONE of:
- Script algorithm / approach
- Input data structure
- Caching strategy
- Output format / schema
- Error handling breadth

Never change two variables in the same iteration.

## Copilot CLI Integration

On M1 Mac with limited compute, use Copilot CLI (GPT-5.0 mini, free tier - zero tokens) for:

**Script generation:**
```bash
cat scripts/toolbox/closest_existing.py | \
  copilot -p "Write a new Python script to solve: <gap description>. Match the style of the piped file. Include: file header, type hints, __main__ test block, argparse CLI." \
  > scripts/toolbox/<gap-slug>.py
```

**Pre-sandbox code review (Security Auditor persona):**
```bash
cat scripts/toolbox/<gap-slug>.py | \
  copilot -p "Review this Python script for security issues. Do NOT use tools. Check for: subprocess misuse, file path escapes, eval/exec, hardcoded credentials. Format findings as: CRITICAL / MODERATE / MINOR." \
  > /tmp/security_review.md
```

**Post-run fitness check (Architect persona):**
```bash
cat run.log scripts/toolbox/<gap-slug>.py | \
  copilot -p "Review this tool and its run log. Does it meet the goal? Is it simpler than the approach it replaces? Recommend: keep or discard with one-line reason." \
  > /tmp/architect_review.md
```
