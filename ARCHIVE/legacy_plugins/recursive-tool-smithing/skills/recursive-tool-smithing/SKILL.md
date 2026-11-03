---
name: recursive-tool-smithing
description: >
  Autonomous seed-agent that identifies capability gaps in an agent's toolbox,
  writes Python scripts to fill them, validates each script in a restricted 5-minute
  sandbox, and permanently commits passing tools to the agent's growing library.
  Auto-invoked when the user mentions "capability gap", "seed agent", "tool evolution",
  "Recursive Tool-Smithing", "self-improving agent", or wants to bootstrap a program.md
  strategy file for autonomous tool creation. Also trigger when the user wants an agent
  that writes its own scripts in a loop (Karpathy-style autoresearch pattern).
  Do NOT trigger for one-off script writing, standard automation, or general Python help.
disable-model-invocation: false
---

# Recursive Tool-Smithing
> The "Seed Agent" pattern: identify gaps, forge scripts, test in sandbox, commit winners.

[See acceptance criteria](references/acceptance-criteria.md) | [Safety sandbox rules](references/safety-sandbox.md) | [Evolution loop spec](references/evolution-loop.md) | [Rollback contract](references/rollback-contract.md) | [Fallback tree](references/fallback-tree.md)

## Pre-Execution Commitment Diagram

Before any execution, display this commitment map:

```
+-----------------------------------------------------------------------+
|              RECURSIVE TOOL-SMITHING PROTOCOL v2                      |
+-----------------------------------------------------------------------+
| [ ] Phase 0: Boot     - toolsmith_wakeup.py + program.md init        |
| [ ] Phase 1: Branch   - git checkout -b autoresearch/<run-tag>        |
| [ ] Phase 2: Scout    - gap_scanner.py + pre-flight brief             |
| [ ] Phase 3: Forge    - write candidate to scripts/toolbox/           |
| [ ] Phase 4: Sandbox  - 5-min restricted run > run.log 2>&1          |
| [ ] Phase 5: Decide   - keep (advance) / discard (git reset) / crash  |
| [ ] Phase 6: Ledger   - append commit+metrics to results.tsv          |
+-----------------------------------------------------------------------+
LOOP FOREVER until: budget exhausted | 3 crashes | GAP_NONE x2
```

## Mode Detection (Dual-Mode Meta-Skill)

Inspect the target directory before proceeding:

- **Bootstrap Mode** (no `program.md` or empty `scripts/toolbox/`): Run Discovery Phase, then call `scripts/toolsmith_wakeup.py --bootstrap`.
- **Iteration Mode** (`program.md` exists + toolbox has >= 1 script): Run `scripts/toolsmith_wakeup.py --mode TELEMETRY`, load boot digest, proceed to Phase 1.

## Discovery Phase (Bootstrap Mode only)

Ask the following in a single grouped question set:

1. **Run tag**: Short identifier for this session (e.g., `mar14`). Branch will be `autoresearch/<tag>`.
2. **Project scope**: What is this agent's domain? (e.g., InvestmentToolkit, Project Sanctuary)
3. **Self-improvement goal**: What single metric should the agent minimize? (e.g., "query latency ms", "memory tokens used", "API calls per task")
4. **Success threshold**: What % improvement makes a new script worth keeping? (default: 10%)
5. **Eval harness**: Existing benchmark path, or generate a default one?
6. **Safety level**: Blast radius limit (default: `scripts/toolbox/` read/write only)

## Recap (Bootstrap Mode only)

```
Run tag        : autoresearch/<tag>
Project scope  : <answer>
Strategy goal  : minimize <metric>
Keep threshold : >= <N>% improvement OR code deletion maintaining performance
Toolbox path   : scripts/toolbox/
Safety limit   : scripts/toolbox/ read/write only
```

"Does this look right? (yes / adjust)"

## Execution

### Phase 0: Boot (every session)

Run the boot script to generate session context:

```bash
python3 scripts/toolsmith_wakeup.py --mode TELEMETRY > .boot_digest.md
```

Read `.boot_digest.md` for: toolbox inventory, last session metrics, open gaps from prior run.

### Phase 1: Branch Setup (Bootstrap or new run-tag only)

```bash
git checkout -b autoresearch/<run-tag>   # must be a fresh branch
```

Initialize `results.tsv` with header (do NOT commit - leave untracked):
```
commit  metric_before  metric_after  memory_mb  fitness_depth  fitness_scope  status  description
```

### Phase 2: Gap Scout

1. Run `python3 scripts/gap_scanner.py --toolbox-path scripts/toolbox --gap-log gaps.json`
2. **Pre-flight brief**: Before generating, search the toolbox for semantically similar scripts. If a similar tool exists (>80% overlap in purpose), extend it rather than creating a duplicate.
3. Log the top-priority gap. One gap per cycle. No parallel gap resolution.

### Phase 3: Forge Candidate Script

1. Read `program.md` to anchor on the strategy goal.
2. Write the candidate to `scripts/toolbox/<gap-slug>.py`.
3. Apply the Anti-Pattern Vaccination - see [fallback-tree.md](references/fallback-tree.md#anti-patterns).
4. Script MUST include: file header, type hints, input validation, `__main__` test block.
5. **git commit the candidate immediately** (even before sandbox) so we have a hash to revert from.

> **MODE-INVARIANT GATE** - See [safety-sandbox.md](references/safety-sandbox.md). Runs on EVERY iteration. No bypass permitted.

### Phase 4: Sandbox Validation

```bash
python3 scripts/sandbox_runner.py --script scripts/toolbox/<gap-slug>.py > run.log 2>&1
```

Extract result:
```bash
grep '"status"' run.log    # pass / crash / timeout / violation
grep '"elapsed_seconds"' run.log
```

Measure fitness scores:
```bash
python3 scripts/evolution_metrics.py fitness --file scripts/toolbox/<gap-slug>.py
```

### Phase 5: Keep / Discard / Crash Decision

**Simplicity Criterion** (from Karpathy autoresearch):
- Small improvement + adds significant complexity = `discard`
- Equal/worse performance + adds lines = `discard`
- Improvement achieved by *deleting* code = **definitely keep**
- Clear improvement (>= threshold) with modest additions = `keep`

| Outcome | Action |
|---------|--------|
| `pass` + metric improved >= threshold | `keep` - advance branch |
| `pass` + metric improved but adds complexity | evaluate simplicity criterion |
| `pass` + no improvement OR worse | `discard` - **`git reset HEAD~1`** to revert |
| `crash` / `timeout` / `violation` | `crash` - **`git reset HEAD~1`**, log, narrow hypothesis |

**On discard/crash**: `git reset HEAD~1` (not just deleting the file). This is the Karpathy discipline - the branch only ever contains commits that improved the metric.

**Graduated Autonomy for commits:**
- New standalone tool, < 100 lines, no new deps: auto-commit + advance.
- Modifies existing tool OR adds new dependency: **HUMAN GATE** - show diff, wait for "Proceed".

Commit message (keep only):
```
feat(toolbox): add <gap-slug> - <one-line rationale>

Strategy: minimize <metric> (<before> -> <after>)
Fitness: depth=<N> scope=<N>
```

### Phase 6: Ledger Update

Append one TSV row (tabs, NOT commas):
```
<commit-hash>  <metric_before>  <metric_after>  <memory_mb>  <fitness_depth>  <fitness_scope>  <keep|discard|crash>  <description>
```

Every 5 cycles, invoke continuous-skill-optimizer:
```bash
python3 ${plugins}/agent-scaffolders/skills/continuous-skill-optimizer/scripts/execute.py \
  --skill-path . --eval-set evals/evals.json --results-dir evals/experiments
```

**LOOP FOREVER** - do not pause for human approval mid-loop. The loop runs until a halt condition is met.

## Halt Conditions

- Iteration budget (from `program.md`) exhausted
- 3 consecutive `crash` classifications
- `GAP_NONE` on 2 consecutive cycles (toolbox is sufficient for current strategy goal)
- Any trigger in [rollback-contract.md](references/rollback-contract.md)

## Output

```
## Tool-Smithing Session Report
- Run tag         : autoresearch/<tag>
- Branch          : <branch>
- Cycles run      : <N>
- Scripts kept    : <N> (commits advanced)
- Discarded       : <N> (git reset)
- Crashes         : <N>
- Best metric     : <before> -> <after> (<delta>%)
- Halt reason     : <reason>
- Next open gap   : <description or "none">
```

**Sources Checked:** `.boot_digest.md`, `program.md`, `scripts/toolbox/` (inventory), `evals/results.tsv`
**Sources Unavailable:** [list any missing]

## Next Actions

1. Continue loop: re-invoke with same run-tag to resume on the same branch.
2. Visualize ledger: `python3 scripts/eval-viewer/generate_review.py evals/experiments`
3. Optimize triggers: invoke `continuous-skill-optimizer` on this skill.
4. Audit compliance: run `audit-plugin` against `plugins/recursive-tool-smithing/`.
5. Graduate to agent: run `create-agentic-workflow` on this skill.
