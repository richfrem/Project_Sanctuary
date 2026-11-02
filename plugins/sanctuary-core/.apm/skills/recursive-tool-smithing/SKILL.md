---
name: recursive-tool-smithing
description: "Autonomous seed-agent engine that scans for capability gaps in the agent toolbox, generates Python tools, validates in a restricted sandbox, and commits passing tools. Use when automating capability evolution or running iterative tool development loops."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Recursive Tool-Smithing (`recursive-tool-smithing`)

Autonomous engine implementing the Seed Agent pattern: scan capability gaps, forge scripts, validate inside a restricted sandbox, and permanently commit passing tools.

## Contents
- [Constraints](#constraints) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints
1. **Safety Sandbox**: Candidates must execute within a restricted 5-minute timeout sandbox before being accepted.
2. **Blast Radius**: Generated tools are confined to `scripts/toolbox/`.
3. **Rollback**: Failed tools are discarded via git revert/reset immediately.

## Quick Start

```bash
# Scan for open capability gaps
python3 scripts/gap_scanner.py --toolbox-path scripts/toolbox --gap-log gaps.json

# Run sandbox validation on a candidate script
python3 scripts/sandbox_runner.py --script scripts/toolbox/example_tool.py

# Evaluate fitness metrics
python3 scripts/evolution_metrics.py fitness --file scripts/toolbox/example_tool.py
```

## Workflow

1. **Scout**: Run `gap_scanner.py` to identify missing functionality.
2. **Forge**: Write the tool candidate into `scripts/toolbox/` with strict type annotations and docstrings.
3. **Sandbox Gate**: Execute `sandbox_runner.py` to verify safety and non-crashing execution.
4. **Decide & Commit**: If passing fitness thresholds, commit via `commit_tool.py`; otherwise reset.

## Verification

```bash
python3 scripts/toolsmith_wakeup.py --mode TELEMETRY
```

## References
- Safety Sandbox: [safety-sandbox.md](references/safety-sandbox.md)
- Evolution Loop: [evolution-loop.md](references/evolution-loop.md)
- Rollback Contract: [rollback-contract.md](references/rollback-contract.md)
- Discovery & Setup: [discovery-guide.md](references/discovery-guide.md)
- Acceptance Criteria: [acceptance-criteria.md](references/acceptance-criteria.md)
- Fallback Tree: [fallback-tree.md](references/fallback-tree.md)
