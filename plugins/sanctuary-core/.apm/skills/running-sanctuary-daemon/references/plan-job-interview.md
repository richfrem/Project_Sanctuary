# Plan a daemon job: interactive interview

Use this before anything touches `~/.sanctuary/daemon/queue/`. It turns a plain-language request ("clean up the auth module overnight") into one or more bounded, approved `enqueue` commands.

Adapted from Richard's 2026-11-09 draft of a `plan-sanctuary-daemon-job` skill, folded into this skill so two skills do not compete for the same requests.

## Operating principles
1. **Ask before queueing.** Never enqueue on the first turn of a request.
2. **Plain language over flags.** Explain what each choice protects against ("stop it after 15 minutes so a stuck run cannot burn quota all night"), then show the flag.
3. **Human Gate.** Enqueue only after an explicit go-ahead such as "go", "proceed" or "enqueue it". "Sounds good" or "looks fine" is not approval: ask once more.
4. **Say what the daemon cannot do.** Do not offer a capability the daemon lacks (see "Not supported yet").

## Phase 0: Check the ground (no questions yet)
Run these, read-only:
```bash
D=plugins/sanctuary-core/.apm/skills/running-sanctuary-daemon/scripts/sanctuary_daemon.py
python3 $D status                                   # daemon_status, pid, existing jobs
ps -p "$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.sanctuary/daemon/state.json"))).get("pid",0))')" >/dev/null 2>&1 && echo running || echo "not running"
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py board
```
- Daemon not running: say so, and tell the human how to start it (`python3 $D run`, or the launchd agent in [operations.md](operations.md)). Do not start it yourself.
- `BLOCKED_ON_HUMAN`: show the last failures (`~/.sanctuary/daemon/logs/`), and ask whether to `reset` before planning new work.
- A lease already covers the target files: plan around it or ask.

## Phase 1: The four questions
Ask them together when the request is vague; skip any the human already answered. Offer the recommended option first.

### 1. What kind of work is it?
- **Check or script** (tests, lint, build): shell runner, no model, zero tokens.
- **Bounded code change** (one feature, one refactor, tests for one module): an agent engine, in a task worktree, usually with "must commit".
- **Open-ended investigation**: best split into a planning job and later jobs, or done interactively instead. The daemon is good at bounded jobs, not open exploration.

### 2. One run, or a pipeline?
- **Single pass**: one job, verified, done.
- **Pipeline of 2-3 steps**: separate jobs chained with `--after <previous-job-id>`. A step runs only if the previous one COMPLETED; otherwise it is SKIPPED, so a failed step never triggers the next one.
- **Not supported yet: "keep going until tests pass".** The daemon has no retry-until-green loop. Offer a fixed pipeline (for example: fix, then test), and note that three consecutive failures stop the daemon until a human runs `reset`.

### 3. Which engine?
| Choice | Use for | Cost | Notes |
|---|---|---|---|
| Shell (no AI) | `make test`, `pytest`, scripts | zero tokens | Deterministic. |
| Local Qwen (`--cli llama`) | small drafting or summaries | local compute only | Needs the local model server; weakest at multi-file edits. |
| Codex (`--cli codex`) | quick scaffolding, patches | ChatGPT Plus quota | Codex's default sandbox for `exec` was not verified here; check it can write before relying on it. |
| Antigravity (`--cli agy`) | multi-file edits | Gemini quota | Edits need the bypass opt-in (question 4). |
| Claude Opus 5.5 (`--cli claude --model claude-opus-5-5`) | architecture, hard refactors | Claude quota (most expensive) | Edits need the bypass opt-in (question 4). Cheaper: `claude-sonnet-5-5`, `claude-haiku-5-5`. |

Copilot is never offered (work repositories only). Check headroom before an AI job: `python3 plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/usage_probe.py headroom`.

### 4. Which safety rails?
- **Time limit.** Default 60 s for shell checks, 600-900 s for agent jobs. After the limit the job's whole process group gets SIGTERM, then SIGKILL 2 s later.
- **Proof of work.** "Must commit" (`--require-commit`): the job counts as COMPLETED only if the worktree's HEAD moved.
- **Where it runs.** A task worktree (`.worktrees/<task>`), leased with `manage_coordination.py checkout --create-worktree`, so the human's branch is untouched. Strongly recommended for any agent job.
- **Edit permission (agent jobs that change files).** By default jobs run isolated: the engine gets no permission bypass, which for `claude -p` and `agy` means it cannot edit files unattended. To allow edits, the job needs `--allow-permission-bypass`. Say plainly what that means: the engine can run tools and edit files without asking, limited only by the worktree, the timeout and the circuit breaker. Recommend it only together with a dedicated worktree and a timeout, and ask for it explicitly.

## Phase 2: Plain-language summary
Show this, filled in, and wait:

```markdown
### Proposed daemon job
- **Mission:** MISSION-0NN (or task name)
- **Goal:** <one sentence>
- **Engine:** <Shell / Local Qwen / Codex / Antigravity / Claude model>
- **Runs in:** `.worktrees/<task>` (leased) | repo root
- **Steps:** 1 job | pipeline: <job-a> -> <job-b> (b runs only if a completes)
- **Rails:** stop after <N> min; must commit: <yes/no>; edit permission: <isolated (no edits) / bypass approved>
- **Cost:** <zero tokens / which quota, current headroom>

Command(s):
python3 $D enqueue --mission MISSION-0NN --authorized-by <name> --id <job-a> --timeout <sec> \
  [--require-commit] (--shell "<cmd>" | --cli <engine> [--model <model>] --cwd .worktrees/<task> \
  --instruction "<bounded instruction>" [--allow-permission-bypass]) [--after <previous-id>]

Reply "go" to queue it, or tell me what to change.
```

Instruction-writing rules for agent jobs: one bounded goal, the files or area in scope, the check to run, and the exact commit message. No open-ended "improve everything".

## Phase 3: Enqueue after explicit approval
1. Make sure the worktree and lease exist (`manage_coordination.py checkout ... --create-worktree`).
2. Run the `enqueue` command(s), in pipeline order, with `--authorized-by` set to the approving human.
3. Confirm with `python3 $D status`.
4. Tell the human how to watch it: `tail -f ~/.sanctuary/daemon/logs/<job-id>.log`, and `status` for the result.
5. When it finishes, review the diff and continue the normal PR and closeout workflow. The daemon does not push, open PRs or merge.

## Example
> "Set up a background job to clean up the auth module."

Phase 0 shows the daemon running and the circuit closed. Then:
> I can set that up. Four quick things:
> 1. **Scope:** refactor the existing code, or write tests for it first?
> 2. **Steps:** one run, or two chained jobs (write tests, then refactor only if the tests job completes)?
> 3. **Engine:** Claude Opus 5.5 for a careful refactor, Sonnet to save quota, or Codex for something quicker?
> 4. **Rails:** 15-minute limit each, in a new `.worktrees/auth-cleanup` worktree, must commit? Editing needs the permission-bypass opt-in; with the worktree and time limit as the boundary, is that OK?
