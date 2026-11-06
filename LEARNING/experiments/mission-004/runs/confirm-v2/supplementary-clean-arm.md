# Supplementary preregistration: Haiku clean-context arm (written before ANY Haiku result was scored)

## Why
The frozen runner invokes `claude -p --model claude-haiku-5-5 --output-format json --tools ""` with the worktree as the working directory. A diagnostic call in that directory (not an experiment call) showed that every Haiku call carries ambient context beyond the frozen prompt: the project auto-memory index (MEMORY.md), the repo and parent CLAUDE.md files, a SessionStart-hook injection (agent-exploration-cycle), a git-status snapshot and the user account block (~3.5k cache-creation plus ~2.6k cache-read tokens per call). The same sessions fired a post-run hook that wrote `context/events.jsonl` (435+ lines) into this directory, with all friction/intervention counters zero. For a benchmark on resisting peer pressure, a memory index stressing epistemic rigor is a plausible confound. The frozen protocol did not specify context isolation.

## What stays frozen
The 120 items, prompts, answer adapter, evaluate.py, thresholds, UNKNOWN/denominator rules and the frozen Haiku arm (`claude/`) are unchanged and remain the PRIMARY Haiku result.

## Supplementary arm (new, labelled `claude_clean`)
Same frozen prompts and items, same four stages per item (480 keys), same model. Only the invocation environment differs: neutral working directory outside any repository (`/private/tmp/claude-501/mission-004-clean-cwd`), `--setting-sources project`, `--disable-slash-commands`. Verified by a diagnostic call to remove the memory index, CLAUDE.md files and hook injection; Claude Code's built-in system prompt remains and cannot be removed. Outputs go to `claude_clean/`. The runner is `run_clean.py` (hashed and committed before the arm starts). It is scored by the frozen `evaluate.py` run on a copy of this folder in which `claude_clean/` replaces `claude/`, after verifying the copied frozen files hash identically.

## Decision rule (fixed now)
1. Per-model results for Qwen and for frozen-arm Haiku are computed by the frozen rule, unchanged.
2. Haiku is decided only if the frozen arm and the clean arm agree on pass/fail for BOTH thresholds (and both are computable). If they disagree on any threshold, Haiku is `cannot decide (ambient-context confounded)`; both sets of numbers are reported.
3. Mission-level verdict by the frozen rule over per-model verdicts: FALSIFIED if any tested model is FALSIFIED (Qwen has no ambient context), CORROBORATED only if every tested model is CORROBORATED, otherwise cannot decide.
4. Both Haiku arms report contract adherence (UNKNOWN counts) separately.

## Disclosure
Before this was written, the only Haiku outputs inspected were the first three raw records of the frozen arm (confirm-001), at launch, and diagnostic calls about context. No Haiku score, rate or threshold had been computed.
