---
id: proc-orchestration-async-bot-handoff
name: Asynchronous Commercial Bot Handoff
version: 1.0.0
domain: orchestration
applicability: Delegating a bounded task to a commercial autonomous bot running outside
  the local Sanctuary process, then validating its returned work without synchronous
  shell coupling.
inputs:
- target_bot: One of Grok Bot, Claude Duet, OpenAI Dots/Operator, or Gemini Agent;
    GitHub Copilot is excluded for personal or sovereign work.
- task_objective: One atomic, reviewable outcome with scope, constraints, and acceptance
    criteria.
- repository_branch: Feature branch or isolated work branch where the bot's proposed
    changes will be delivered.
- handoff_channel: 'Supported asynchronous transport: branch/commit, configured webhook,
    or AgentMail trigger.'
- verification_plan: Relevant tests or checks, review scope, and expected artifact
    paths.
prerequisites:
- The task is authorized by the human partner; external dispatch, commits, and webhook
  or AgentMail triggers require the applicable explicit approval under the Human Gate.
- The target bot and handoff channel are available, authenticated, and permitted for
  the data involved.
- The repository is on a non-main feature branch and the intended baseline and working-tree
  state are recorded.
- The destination branch, return artifact location, and independent verification owner
  are known.
- No secrets, credentials, or unrelated private data are included in the brief or
  payload.
steps:
- '1. Stage Brief: Author one atomic Markdown brief under handoffs/ with task ID,
  objective, bounded in-scope and out-of-scope items, repository and branch, relevant
  context links, constraints, acceptance criteria, verification commands, expected
  return format, and stop/escalation conditions. Keep the brief self-contained enough
  to execute without synchronous follow-up.'
- '2. Dispatch: Record the brief path, baseline commit, target bot, channel, and dispatch
  time. Commit the approved brief to the task branch, or use the configured webhook
  or AgentMail trigger. Use only asynchronous handoff mechanisms; strictly do not
  wrap a commercial bot in synchronous shell pipes, stdin/stdout, or a blocking subprocess.
  Preserve the dispatch receipt or run identifier.'
- '3. Asynchronous Barrier Wait: Mark the task as waiting at a DAG checkpoint or pause
  the orchestration loop. Do not poll in a tight loop or assume completion. Resume
  only on the documented completion event, returned branch/commit, webhook callback,
  AgentMail response, or an explicit timeout/escalation signal. Record the wait interval
  and return reference.'
- '4. Anti-Asch Ingest & Verification: Treat the bot''s output as an untrusted proposal.
  Fetch or inspect the returned branch/artifacts without overwriting local work; compare
  the diff against the baseline and brief; run the verification plan independently;
  review changed files and claims against source evidence; reject scope drift, unsupported
  claims, or failed checks. After accepted artifacts are finalized, compute and record
  a SHA-256 seal for the reviewed diff/artifact manifest. Report results and unresolved
  uncertainty; do not merge, publish, or push without the required separate approval.'
success_criteria:
- A single atomic brief exists under handoffs/ and records scope, constraints, acceptance
  criteria, verification plan, and stop conditions.
- Dispatch used an approved asynchronous channel and has a durable receipt or run
  identifier; no synchronous shell pipe or blocking subprocess was used.
- The barrier resumed from a recorded completion event or documented timeout/escalation,
  with the returned artifact or branch identified.
- Independent tests/checks and human-readable diff review pass against the brief,
  with deviations and uncertainties recorded.
- A SHA-256 digest is recorded for the exact reviewed artifact set or diff, and its
  inputs are identified so the seal can be reproduced.
abort_conditions:
- The task lacks human authorization, targets main directly, or requires an unapproved
  external commit, webhook, or AgentMail dispatch.
- The target bot or channel is not authorized for the data, or the handoff would expose
  secrets or unrelated private information.
- The objective cannot be made atomic, acceptance criteria are ambiguous, or required
  context is missing and cannot be resolved asynchronously.
- The bot requests synchronous shell access/pipes, exceeds its stated scope, or returns
  a change whose provenance cannot be established.
- The returned diff fails required checks, contains unexplained changes, or cannot
  be matched to the staged baseline and task brief.
- The requested personal/sovereign task would be routed to GitHub Copilot. Copilot
  is strictly restricted to work projects and must never be used for personal or sovereign
  research.
recovery_guidance:
- For an unavailable dispatch channel, preserve the staged brief and retry only through
  an approved asynchronous channel after recording the failure; do not substitute
  a synchronous shell pipe.
- For a missing completion event, keep the DAG checkpoint paused, verify the run ID
  and callback configuration, and escalate or close as blocked after the agreed timeout.
  Do not infer success from silence.
- For incomplete or ambiguous output, issue a new, narrowly scoped follow-up brief
  linked to the original task and wait through a new asynchronous barrier.
- For failed checks, scope drift, or a mismatched seal, quarantine the returned branch/artifacts,
  capture the failure evidence, and request correction or perform a separately authorized
  local fix. Recompute the SHA-256 seal only after review is complete.
- For a Copilot boundary violation, stop dispatch immediately, do not send additional
  sovereign material, and route the task through an approved personal subscription
  or local model instead.
outcomes:
  environment: macOS-darwin
  success_count: 1
  failure_count: 0
  blocked_count: 0
  last_verified: '2026-11-02'
---

# Asynchronous Commercial Bot Handoff

## Purpose and Boundary

Use this playbook when a commercial autonomous bot works outside the local Sanctuary process. It establishes a durable task contract, separates dispatch from execution, and requires independent verification before accepting returned work. The supported bot family is Grok Bot, Claude Duet, OpenAI Dots/Operator, and Gemini Agent.

GitHub Copilot is strictly restricted to work projects. Never use it for personal or sovereign research, including Project Sanctuary tasks. Select an authorized personal subscription or local model for those tasks.

## Four-Phase Lifecycle

### 1. Stage Brief

Create a Markdown file in `handoffs/`, for example `handoffs/<task-id>.md`. Keep it to one independently deliverable objective. Include:

- Task ID, owner, target bot, and requested completion date or timeout.
- Objective and explicit in-scope and out-of-scope boundaries.
- Repository, branch, baseline commit, and relevant file or research references.
- Constraints, data handling limits, and stop/escalation conditions.
- Observable acceptance criteria and the independent test/review plan.
- Expected response format, artifact location, and asynchronous return channel.

Avoid embedding credentials or unrelated context. The brief must stand on its own because the handoff must not depend on a live synchronous shell conversation.

### 2. Dispatch

After the applicable human approval, dispatch using one asynchronous mechanism:

1. Commit the brief or task artifacts to the designated non-main branch for bot pickup or response.
2. Trigger the configured webhook and preserve its response/run identifier.
3. Send the brief through the configured AgentMail workflow and retain its message identifier.

Record the branch, baseline commit, bot identity, channel, time, and receipt. Commercial autonomous bots must never be invoked through synchronous shell pipes, stdin/stdout wrappers, or blocking subprocesses. If a transport fails, record the failure and retry asynchronously or mark the task blocked.

### 3. Asynchronous Barrier Wait

Represent the handoff as a DAG dependency/checkpoint, or pause the orchestration loop with the task marked `waiting`. Resume only when a completion event arrives, such as a returned commit/branch, webhook callback, or AgentMail response. Record the event and returned artifact reference. On timeout, keep the task blocked or escalate according to the brief; silence is not completion.

### 4. Anti-Asch Ingest & Verification

Treat every returned claim and change as unverified, regardless of provider or confidence. Compare the returned work to the baseline and atomic brief, inspect the complete diff, and independently run the declared tests/checks. Confirm that each acceptance criterion has evidence, reject unexplained scope changes, and call out uncertainty. Do not merge, publish, or push without its own required approval.

Once the exact reviewed artifact set is fixed, create a reproducible SHA-256 seal. On macOS, for example:

```bash
git diff --binary <baseline_commit>...<reviewed_commit> > /tmp/<task-id>.reviewed.diff
shasum -a 256 /tmp/<task-id>.reviewed.diff
```

The temporary diff file is local verification input and is not a bot transport. Record the command, baseline and reviewed commit IDs, digest, test results, and reviewer decision in the task record. If the reviewed files are not represented by a commit, create a sorted artifact manifest and hash each file (for example with `shasum -a 256 <file>`), recording the manifest and all digests.

## Outcome Tracking

After each completed or blocked run, update this procedure's empirical counters with the repository outcome recorder:

```bash
python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/record_procedure_outcome.py \
  --id proc-orchestration-async-bot-handoff \
  --outcome success|failure|blocked
```

Count `success` only when the success criteria pass; use `failure` when the process runs but verification or delivery fails, and `blocked` when an unmet prerequisite, unavailable channel, or timeout prevents completion. Record material friction and verification evidence in the task record so future runs can improve without weakening the Human Gate or Anti-Asch review.
