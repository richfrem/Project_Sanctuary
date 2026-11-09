---
name: operating-sanctuary-bot
description: "Sets up and operates Sanctuary Bot: an always-available local background bot (like Grok Bot, ChatGPT scheduled tasks or Gemini scheduled actions, but running on this machine) that runs approved automations on a schedule or on demand: one-off jobs, multi-step pipelines, priority backlogs or verify-and-fix sweeps, using shell commands or AI engines (claude, codex, agy, local llama). Use when the user wants to set up a bot, an automation, a scheduled or recurring task, a background worker, an agent that works while they are away or overnight, or to check, pause, resume, run, approve or stop their bot or automations. The daemon, scheduler, conductor and safety rails underneath are handled for the user. Never started by session bootstrap."
allowed-tools: Bash, Read, Write
---

# Operating Sanctuary Bot (`operating-sanctuary-bot`)

One interface, `scripts/bot.py`, over everything the user should not have to think about: the runtime daemon, the scheduler, the multi-step conductor, approvals, run memory and notifications. Users talk about **the bot** and its **automations**; each firing is a **run**.

## Contents
- [Constraints](#constraints) • [Workflow](#workflow) • [Quick start](#quick-start) • [Verification](#verification) • [References](#references)

## Constraints
- **Interview, then approve, then run.** Never register, approve or run on the first turn. The user approves each automation with an explicit "go"; `bot.py approve` records it against the definition's content hash, so any later edit pauses it until re-approved.
- **The human starts the bot.** Starting it on request in this conversation is fine; session bootstrap never starts it.
- **Definitions are reviewed like code**: `.agent/automations/<name>.json` in the repo, through a feature branch and PR. Approvals, run state, history and the inbox stay machine-local in `~/.sanctuary/daemon/`.
- **Edits happen only in a task worktree** (`.worktrees/<task>`, leased with `manage_coordination.py`). The bot never pushes, opens PRs or merges, and never writes the repo's memory files from the background.
- **No permission bypass unless approved**; never Copilot. **Cost is bounded by rails** (cycles, timeouts, failure limits), not tokens: say so.

## Workflow
1. **Plumbing (silent unless broken):** `bot.py doctor`; if not set up, `bot.py setup` (configures `.agent/automations/` and state). Report problems in one line.
2. **Interview:** follow [setup-interview.md](references/setup-interview.md): what to run, how often, one step or several, engine, rails, notifications.
3. **Write the definition** ([automations.md](references/automations.md)), run `bot.py validate <name>`, show a plain-language summary with worst-case time and quota, and wait for "go".
4. **Commit the definition on a feature branch and open a PR** (or keep it local if the user says so), then `bot.py approve <name> --by <approver>` and, if needed, `bot.py start`.
5. **Report:** `bot.py status`, when it next runs, and how to pause or stop it.
6. **Later:** `status`, `history <name>`, `pause`/`resume`, `run-now`, and `~/.sanctuary/daemon/inbox.md` for results.

## Quick start
```bash
B=plugins/sanctuary-core/.apm/skills/operating-sanctuary-bot/scripts/bot.py
python3 $B setup && python3 $B doctor
python3 $B validate nightly-tests
python3 $B approve nightly-tests --by <approver>
python3 $B start
python3 $B status            # bot, automations, recent results
python3 $B run one-off.json --by <approver>   # run once without scheduling
```

## Verification
- `scripts/test_bot.py` (11: schedule maths, catch-up, approval tamper refusal, overlap, auto-pause, run-now, memory, inbox, notify, end-to-end with a real bot), `scripts/test_campaign.py` (14: multi-step conductor), `scripts/test_daemon.py` (9: runtime rails). No model calls.
- `bot.py doctor` checks host, config, bot process and engines.

## References
- Interview: [setup-interview.md](references/setup-interview.md)
- Automation definitions, schedules, approval, memory, notifications: [automations.md](references/automations.md)
- Multi-step runs (modes, steps, rails): [campaign-manifest.md](references/campaign-manifest.md) • [conductor-loop.md](references/conductor-loop.md)
- Running the bot (start at login on macOS, Linux, Windows; rails; known gaps): [operations.md](references/operations.md)
- Example: [example-automation.json](assets/example-automation.json)
