# Delegating Work Across CLIs: What Was Hard (MISSION-060 handoff, 2026-11-07)

**Status:** OBSERVED
**Source:** Preparing MISSION-060 for execution and handing it to Codex (PRs #409-#411); the bootstrap repair's cross-client tests (see `bootstrap_instruction_delivery.md`, Chronicle 381)
**Last Verified:** 2026-11-07
**Tags:** ["delegation", "cross-cli", "handoff", "quota", "cli-agent", "orchestration", "process"]

## Core Definition

Assigning work to another agent CLI (Codex, AGY, Claude) is not one operation. In this repository it is a *record* plus a *pickup*, and several parts of that path were harder than expected. Provenance: **[repo]** verifiable in code, git or the event ledger; **[measured]** read from a client's own transcript or the usage probe; **[session]** observed during this work and not independently reproduced. Nothing here claims that the assigned agent has started, or will follow, the handoff: **Codex was never launched by the recorder.**

## What happened

Richard approved the MISSION-060 study ("Proceed"). The recorder built the instruments, found it could not run the study within the stop rules, and on Richard's instruction ("assign the execution of the mission to Codex maybe sol 6.1 medium", then "proceed assign to sol 6.1") recorded the assignment as STAGED, assignee codex, model `gpt-6.1-sol` at medium effort, with a handoff note (`LEARNING/experiments/mission-060/HANDOFF-codex.md`). Zero model calls were made for the study.

## Struggles and what each taught

1. **There is no "assign" operation [repo].** `manage_coordination.py` offers checkout, checkin, board, heartbeat and similar, but nothing that assigns a task to another agent. Handoff is only the check-in's `--handoff` text (normalized to a short note, about 400 characters) that the Predecessor Relay shows at the *next* session start, plus whatever the manifest and notes carry. Nothing pushes work into a running Codex. So an assignment must be recorded honestly as **ASSIGNED, NOT STARTED** (status STAGED, which the backlog documents as "human partner issued explicit authorization"), with the full handoff in a repository note the relay text points to. Whether the target CLI then follows it is unverified.
2. **The orchestrator was the scarce client [measured].** Claude was both the orchestrator and the proposed worker pool. Its *weekly* window was 36% remaining against a 25% stop floor (resets Nov 9) while the 5-hour window, the one the calibration helper reports, hid this; Codex's 5-hour window was 28% and AGY's Gemini windows were 85%/96%. My own interactive turns drained the same window (45% to 36% in about an hour). The approved plan also sized "additional tokens" and ignored the fixed scaffold every fresh session carries: first-request context was about 49.6k (Sonnet), 50.6k (Haiku), 28.8k (AGY) and 32.8k (Codex), so roughly 25 Claude calls could not fit in the 11 points of room. Lesson: check every backend's windows, 5-hour and weekly, before assigning; orchestrate from the client with the most headroom; run long pipelines as unattended scripts with few model turns; budget scaffold and accumulated input, not only "additional" tokens.
3. **The cli-agent router has gaps [repo, measured].** It cannot set Codex reasoning effort (a pass-through `--executable` wrapper that injects `-c model_reasoning_effort="medium"` is needed); it reports usage as `null` for Claude and AGY, so per-call tokens must be read from each client's transcript (Claude project JSONL, AGY `transcript_full.jsonl`, Codex rollout); it prefixes prompts with blank lines; the repo's AGY model catalog id differed from the installed CLI's (`gemini-3.8-flash` versus `gemini-3.8-flash-medium`); isolated mode removes tools, and `--cwd/--allowed-root` bound the launch directory but are not a sandbox.
4. **Booting an agent inside the repo breaks blinding [repo].** The SessionStart briefing prints priority rankings, so any experiment that needs agents *not* to see rankings must run them in an empty directory outside the project, which also avoids the bootstrap cost.
5. **A handoff-tooling mistake of mine [session].** `checkin --status` accepts only COMPLETED, CANCELLED or ABANDONED; I passed `HANDED_OFF`, piped the command through `head` (hiding the error), and the following steps ran on an unchanged tree, producing a premature closure capture that had to be discarded. Practice: read `--help` choices before using a flag, never pipe a state-changing command through `head`, check exit codes, and capture only after the final check-in is merged.
6. **Environment traps [session].** A file named `select.py` shadowed the standard library; zsh does not word-split unquoted variables (it bit four times); a script run from a study folder left empty directories.

## Preventive practices

- Treat "assign" as: manifest `assignee` + `execution_handoff` + a repository handoff note + a check-in whose short text names that note; state ASSIGNED, NOT STARTED.
- Before assigning, run `usage_probe.py snapshot` and read **every** window for **every** backend; do not rely on the 5-hour figure alone.
- Name the executor's model from the installed CLI (`agy models`, `codex` config), not only the catalog.
- Include in the handoff: approvals and their limits, the quota snapshot, what is untested, ordered steps, pitfalls found, and the definition of done.
- Verify pickup by reading the Predecessor Relay in a fresh session of the target CLI (not done yet).

## Not verified

That Codex GPT-6.1 Sol picks up and executes the handoff; that the relay shows the check-in text for this item; the untested `measure_*` usage functions in the study's `common.py`; Codex's weekly-token cost per call (no calibration exists for it).

## Evidence and links

PRs #409 (instruments and handoff), #410 (final check-in), #411 (export); `LEARNING/experiments/mission-060/HANDOFF-codex.md`; `LEARNING/missions/MISSION-060.json` (`execution_handoff`); Chronicle entry 382; related `bootstrap_instruction_delivery.md`, `sovereign_memory_engine.md`.
