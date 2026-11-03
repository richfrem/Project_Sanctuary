# recursive-tool-smithing Plugin

The "Holy Grail" of agentic workflows: a seed-agent that identifies capability gaps,
writes and tests its own Python scripts, and permanently commits winners to a growing toolbox.
Inspired by Karpathy's autoresearch pattern.   https://github.com/karpathy/autoresearch

## Purpose

This plugin implements **Recursive Tool-Smithing**: instead of a human writing every helper
script, the agent identifies friction in its own capabilities, generates a targeted Python
script, sandboxes it for 5 minutes, and if it passes the eval, commits it to its library.

Self-improvement loop:
```
Identify Gap -> Generate Script -> Sandbox (5 min) -> Eval Pass? -> Commit to Toolbox
                                                  |-> Fail/Crash -> Discard, narrow hypothesis
```

## Skills

### recursive-tool-smithing
Main orchestration skill. Runs in two modes:
- **Bootstrap**: Interview to configure `program.md` + initialize toolbox structure.
- **Iteration**: Gap-fill loop using existing `program.md` strategy.

## Sub-Agents

### tool-smith-loop-agent
Autonomous loop agent that runs gap-identification and script-generation cycles
without per-cycle human interaction. Invoked from the main skill for unattended runs.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/gap_scanner.py` | Scans task log for capability gaps |
| `scripts/sandbox_runner.py` | Executes candidate scripts with timeout + scope enforcement |
| `scripts/commit_tool.py` | Commits passing scripts to toolbox with structured message |

## Integration with agent-scaffolders

This plugin delegates to the `agent-scaffolders` plugin for:
- `continuous-skill-optimizer` - trigger quality optimization after each ledger update
- `create-skill` - when a gap requires a full new skill (not just a script)
- `create-sub-agent` - when a gap requires a new autonomous sub-agent

## Safety Sandbox

The Mode-Invariant Compliance Gate enforces on EVERY iteration:
- File writes are allowed only in your plugin folder `plugins/recursive-tool-smithing/`
- Only use copilot CLI and GPT 5.0 mini as it's free zerotoken. max 2 concurrent cli requests at most. 
- 5-minute hard timeout
- No secrets/credential access. with exception of copilot CLI and GPT 5.0 mini.

See `skills/recursive-tool-smithing/references/safety-sandbox.md` for full rules.

## Directory Structure

```text
recursive-tool-smithing/
+-- .claude-plugin/plugin.json
+-- README.md
+-- CONNECTORS.md
+-- requirements.in
+-- skills/
|   +-- recursive-tool-smithing/
|       +-- SKILL.md
|       +-- references/
|       |   +-- acceptance-criteria.md
|       |   +-- safety-sandbox.md      <- Mode-Invariant Compliance Gate
|       |   +-- evolution-loop.md      <- Dual-Mode Meta-Skill spec
|       |   +-- rollback-contract.md   <- Pre-Committed Rollback Contract
|       |   +-- fallback-tree.md       <- Highly Procedural Fallback Trees
|       +-- evals/
|           +-- evals.json
|           +-- results.tsv
+-- agents/
|   +-- tool-smith-loop-agent.md       <- Sub-agent for autonomous loop
+-- scripts/
    +-- toolbox/                       <- Growing library of committed tools
    +-- gap_scanner.py
    +-- sandbox_runner.py
    +-- commit_tool.py
```
