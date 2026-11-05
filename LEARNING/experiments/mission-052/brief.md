# MISSION-052 brief: one bounded four-worker batch under a main-thread orchestrator

**How to start (clean session):** open Claude Code in `/Users/richardfremmerlid/Projects/Project_Sanctuary` on `main`, let the SessionStart hook load the boot context, then send:

> Execute MISSION-052. Read `LEARNING/experiments/mission-052/brief.md` and `README.md` in that folder, then follow the brief as the orchestrator. One bounded batch; stop at the integration branch and ask me before anything is pushed or merged.

Everything below the line "THE INSTRUCTION" is the human's instruction, unchanged except that mission manifest paths were corrected (`LEARNING/research/missions/` is now `LEARNING/missions/`).

---

## State of the stack on 2026-11-05: read this before trusting the instruction

The instruction was drafted against an earlier state of the repository. Verified facts, and what they change:

**Mission statuses (read the manifests; do not rely on this table)**

| Mission | Status | What it means for this batch |
|---|---|---|
| 033 | PROPOSED | A wrapper skill that does not exist. This batch tests the "configuration alone" arm of 033's own falsification test (no new skill). Do not claim 033's thresholds. |
| 026, 031, 034 | RESOLVED / CORROBORATED | Flight board, impact declarations and `verify`, and `merge_prep.py` exist and are tested. Use them. |
| 035 | RESOLVED / **FALSIFIED** | Exact per-worker tokens work; a per-task-class token *predictor* does not (122% error). Do not use predictions for budgets: use explicit caps. |
| 025 | RESOLVED / FALSIFIED | The relay raised handover consistency (22.7% to 68.2%) but missed its 90% bar. Do not assume handoffs fully transfer. |
| 022 | RESOLVED / FALSIFIED | Broad local multi-turn autonomy failed. Qwen gets narrow, checkable work only. |
| 021 | RESOLVED / CONFIRMED | Structured JSON tool-calling on the local model works. |
| 028, 036, 037, 038 | PROPOSED | Documents only. **038 (lease fencing, crash recovery) is not implemented**: an expired lease does not stop a direct filesystem write. Use supervised recovery. |
| 045, 051 | PROPOSED | 051 fixes the size buckets (small < 50k, medium 50-150k, large > 150k billable tokens) and wants same-bucket two-model data: record each worker's bucket and model. |

**Capabilities and the exact interfaces (inspect `--help` before use; these are what exists)**
- **Boot:** the SessionStart hook now injects the Predecessor Relay and the fuel gauge. The wakeup prints an `installed-skills` PASS/WARN line; if it is WARN run `apm install` at the repo root (see INSTALL.md; restore `apm.lock.yaml` afterwards with `git checkout -- apm.lock.yaml` because a local gitignored file adds lock entries on this machine).
- **Flight board** (`operating-sanctuary-guardian/scripts/manage_coordination.py`): `suggest`, `matrix`, `checkout --mission --agent --create-worktree [--targets]`, `verify --mission`, `heartbeat`, `board`, `checkin --handoff`. The board is **untracked per-machine state** that scripts always read from the main checkout, even from a worktree. `checkout` marks a manifest `IN_PROGRESS` in the worktree: do not commit that marker. Lease only what a worker needs; a broad lease blocks every other mission.
- **Merging generated files:** never hand-merge `LEARNING/index.jsonl`, `LEARNING/backlog.md` or the ledgers: run `merge_prep.py regenerate` (reports ledger drift, exit 3, instead of repairing it). `events.jsonl` merges with `union`.
- **Measurement** (`evaluating-agent-work/scripts/usage_probe.py`, `operating-sanctuary-guardian/scripts/usage_hook.py delegated`): snapshot before dispatch; record each worker afterwards. agy: run it with `--output-format json` (text mode reports no usage). Claude delegates: run from their own directory and they are flagged `concurrent` because they share the account window with a Claude orchestrator. **Known gap, resolve before launch:** `codex_tokens_between` sums *every* Codex rollout in the time window, so two concurrent Codex workers cannot be separated by the current recorder; read each worker's own rollout file by its session id (printed by `codex exec`), or the Codex pair is attributable only jointly. Say which.
- **Dispatch** (`~/.claude/skills/cli-agent/scripts/run_agent.py`, with `references/*-dispatch.md`): `--isolated` suppresses the permission-bypass flags and wraps the input as a source block with a no-tools instruction (it is a request, not a sandbox): right for reviewers and extraction, wrong for an implementation worker that must edit files. Launch each implementation worker with its working directory set to its own worktree. Verify the actual flags first.
- **Local Qwen:** the existing MLX integration is `plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py` plus `.../scripts/local_agent_loop.py` in the same skill (see `plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/local-agent-execution-loop.md`). Its sandbox is **read-only** (list, read, find, search), so Qwen cannot edit files: give it extraction, classification or single-hop inspection and apply or validate its structured output yourself. The held-out evaluation generator and the guardian `tests/` folder are hidden from it by default in every installed copy.
- **Local model resource:** a **human-managed** MLX server is already resident on `:8080` (`managed: false`). Use it as it is; never stop, reap or reconfigure it. A second 7B server was measured (MISSION-029) to cost 3.6 GiB and push this machine's swap; swap was already about 92% used on 2026-11-05. Only Qwen touches the model.
- **Held-out evaluations:** installed skill copies include `tests/`; the sandbox hides the generator by name, but keep any held-out material out of every worker's reach.

**Fuel when this mission was created, 2026-11-05 (`usage_probe.py headroom`; re-read it at launch):** claude 43% ok, **agy 11.4% CONSTRAINED** (Gemini weekly), codex 58% ok. Codex A and B draw on the same quota. Give agy a small, capped task and decide in advance what happens if its window falls below about 5% (reassign, do not push through).

**Lessons from the work that produced this stack (apply them)**
- Commit the dataset resync (`trace_extractor` then `sync_soul_dataset`) right after each change; a deferred one caused a merge conflict on the ledger files.
- Stage by explicit path; never `git add -A` in a shared tree. Experiment records and recorded data are evidence: do not rewrite their paths.
- After editing anything under `plugins/sanctuary-core/`: run the tests first, then `find plugins -name __pycache__ -type d -prune -exec rm -rf {} +`, then `apm install`; the lockfile must have no bytecode entries (a test fails otherwise).
- A newly failing test is a signal: the last one found that installed copies exposed the held-out generator.
- Report a measurement that missed as a miss. Cheap models were not cheaper in total effort: orchestration and verification dominated delegation cost ($5.59 of orchestrator work around a $0.04 Haiku draft).

**What the human still decides:** publication of the integration branch (push, PR, merge to `main`), dataset sync to Hugging Face, and anything else the Constitution gates. The standing "complete the full cycle" habit of earlier sessions does **not** apply to this batch: the instruction below stops at a reviewed integration branch.

---

## THE INSTRUCTION

Act as **Sanctuary Guardian and the main-thread mission orchestrator**. Execute one bounded batch of parallel research/engineering work, supervising four concurrent workers:

1. **Codex A**
2. **Antigravity (`agy`)**
3. **Codex B**
4. **Local Qwen2.5-Coder-7B through the existing MLX integration**

You own mission selection, worktree provisioning, leases, budgets, independent evaluation, integration, and handoffs. Workers own their assigned artifacts and evidence.

Your objective is **verified mission progress with preserved cognitive continuity**. Launching processes or shipping code does not establish empirical corroboration.

### Authorization and scope

This instruction authorizes the selected batch's local execution: create isolated worktrees and feature branches, dispatch workers, write scoped artifacts, run bounded tests, record evidence, create local commits, and integrate accepted work into a dedicated integration branch.

Follow Sanctuary's Constitution and applicable Human Gate requirements. Do not push, merge into protected `main`, publish datasets/models, delete pre-existing work, or change account settings without the required explicit authorization. Prepare the reviewed integration result before requesting any remaining publication approval.

Run **one batch**, not an indefinite autonomous loop. Respect any narrower existing authorization.

### 1. Orient and verify the current implementation

Read:

- `AGENTS.md`
- `START_HERE.md`
- `architecture.md`
- `.agent/rules/constitution.md`
- `context/soul.md`
- `context/user.md`
- `context/memory.md`
- The current Guardian boot digest and predecessor relay.

Use the installed capabilities and current script interfaces. Inspect their help before constructing unfamiliar commands.

Relevant skills:

- `operating-sanctuary-guardian`
- `managing-sovereign-memory`
- `evaluating-agent-work`
- Global `orchestrating-agent-work`
- Global `cli-agent`
- Global `running-control-plane`, where applicable.

Read the current coordination reference:

`plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/references/multi-agent-concurrency-and-flight-board.md`

Use existing mechanisms. Do not create a second scheduler, overlap detector, usage ledger, or merge regeneration system.

### 2. Applicable missions

Read these manifests and their cited evidence before relying on their capabilities:

| Mission | Application to this batch |
|---|---|
| [MISSION-033](../../missions/MISSION-033.json) | Primary orchestration experiment: selection, parallel dispatch, independent review, scoring, and integration. |
| [MISSION-026](../../missions/MISSION-026.json) | Flight board, leases, and isolated worktrees. |
| [MISSION-031](../../missions/MISSION-031.json) | Declared impact, overlap detection, resource conflicts, and verification of actual edits. |
| [MISSION-034](../../missions/MISSION-034.json) | Merge-time regeneration and reconciliation of shared generated artifacts. |
| [MISSION-035](../../missions/MISSION-035.json) | Measured usage, backend headroom, effective cost, and attribution limitations. |
| [MISSION-025](../../missions/MISSION-025.json) | Predecessor relay and known limitations of cross-model handoffs. |
| [MISSION-022](../../missions/MISSION-022.json) | Falsified broad local multi-turn autonomy; constrains Qwen's assignment. |
| [MISSION-021](../../missions/MISSION-021.json) | Existing local structured-output/tool-call interface. |
| [MISSION-028](../../missions/MISSION-028.json) | Proposed local-to-cloud escalation policy; do not assume its targets are achieved. |
| [MISSION-036](../../missions/MISSION-036.json) | Evidence contracts and separation of delivery from claim corroboration. |
| [MISSION-037](../../missions/MISSION-037.json) | Applicability of inherited claims after dependencies change. |
| [MISSION-038](../../missions/MISSION-038.json) | Crash recovery and lease fencing, only to the extent actually implemented and verified. |

Inspect current statuses. A proposed mission is not an available implementation.

**Important scope distinction:** MISSION-033 originally specifies a three-worker pilot. This request is for four workers, including two Codex processes. Record this as a separately preregistered four-worker experiment linked to 033; do not silently overwrite its original hypothesis or claim that this batch proves its original thresholds.

### 3. Select four compatible assignments

Inspect the current mission index, manifests, active leases, worktrees, and backend headroom.

Use the existing `suggest`, `matrix`, and impact-validation capabilities where available. Select high-value work based on:

- Priority and readiness.
- Independence of changed paths and shared resources.
- Clear acceptance conditions.
- Actual worker capability.
- Backend quota and memory constraints.

Default assignment pattern:

- **Codex A:** bounded implementation with executable acceptance tests.
- **AGY:** an independent implementation or experiment with disjoint impact.
- **Codex B:** a second independent implementation/test task with disjoint impact.
- **Qwen:** a narrow, independently checkable mission task such as structured extraction from supplied evidence, classification of bounded records, or a single-hop repository inspection.

Do not allocate all four to the same files. Do not give Qwen an open-ended research or repair loop.

Prefer four independent missions. If fewer than four are ready, assign a bounded supporting subtask to an existing mission with an exclusive artifact path and explicit parent relationship. Do not invent busywork or create a new mission merely to occupy a worker.

Publish a concise dispatch table before launch:

`worker | mission/subtask | objective | worktree | branch | leased paths/resources | acceptance tests | token/time ceiling | reviewer`

Preregister the experiment's measurements and stopping conditions. State budget estimates as estimates. If account telemetry is unavailable, use conservative explicit caps and mark headroom unknown.

### 4. Provision worktrees and leases

Use the existing coordination workflow to create four distinct worktrees and feature branches from a recorded clean base.

Give the workers unique identities, including `codex-a` and `codex-b`.

Verify:

- Each worker's working directory resolves to its assigned worktree.
- Each has an appropriate live lease and impact declaration.
- No unintended path or resource overlap exists.
- The root working tree remains unchanged.
- Runtime outputs and session IDs are separately attributable.
- The two Codex workers share backend quota; they are not independent subscription pools.
- Only one worker loads the local model.
- Qwen's process does not terminate or reconfigure a human-managed MLX server.

Do not hold a broad orchestration lease that unnecessarily blocks all child missions. Lease shared bookkeeping only through the supported coordination mechanism.

If 038's recovery protections are unavailable, record that limitation and use supervised recovery. Do not pretend expired leases fence direct filesystem writes.

### 5. Launch all four workers concurrently

Use the current `cli-agent` executor interfaces and established worktree confinement practices. Verify the actual supported syntax rather than assuming flags.

Start all four dispatches without awaiting the completion of earlier workers. Retain process/session handles and confirm that all four have begun their assigned work.

The local worker uses the existing MLX adapter and tool runner. Do not assume Qwen has a standalone coding CLI or feed unchecked model output directly to a shell.

Give every worker a compact brief containing:

- Mission ID, hypothesis, and current evidence.
- Exact objective and deliverables.
- Allowed paths and resources.
- Prohibited operations and authorization boundaries.
- Acceptance tests and falsification criteria.
- Budget and deadline.
- Relevant predecessor findings that should not be unnecessarily repeated.
- Required structured return format.

Worker return format:

1. Changed files.
2. Implementation summary.
3. Exact tests/experiments run and observed outcomes.
4. Evidence artifact paths.
5. Hypothesis assessment, with unmet thresholds explicit.
6. Remaining defects and uncertainties.
7. Handoff note.
8. Usage identifiers for orchestrator measurement.

Workers must not commit, merge, push, edit peer worktrees, or modify shared generated files independently. Git integration remains with you.

For Qwen, use short explicit inputs, validated output schemas, and a fixed small turn limit. Stop repeated or malformed calls. Preserve the failure evidence before escalating; any escalation must remain inside the batch budget and be attributed separately.

### 6. Supervise and measure

Monitor workers without busy polling or flooding the main context with full logs.

Use existing lifecycle hooks and lease renewal mechanisms. Track:

- Worker start/end times.
- Actual overlap interval.
- Input/output/cache tokens where available.
- Turns, retries, and intervention counts.
- Backend quota observations and their freshness.
- Local memory pressure and server health.
- Blockers and undeclared edits.

Take 035-style snapshots before and after execution where supported. Attribute usage using worker session identifiers. Account-wide quota movement cannot be independently charged to both Codex workers.

Do not substitute model self-reported token estimates for measured usage.

Stop or pause the affected worker if it exceeds its cap, violates scope, repeatedly loops, loses its lease, or threatens the local memory budget. Preserve partial work and explain the disposition.

### 7. Review independently and score

For every delivery:

1. Verify the diff against declared impact.
2. Re-run relevant acceptance checks yourself.
3. Inspect evidence supporting the empirical claims.
4. Arrange cross-vendor review where practical.
5. Record your evaluation separately from the worker's self-assessment.

Suggested review routing:

- AGY reviews Codex A's delivery.
- Codex A or B reviews AGY's delivery.
- Claude Guardian reviews Codex B's delivery.
- Deterministic checks plus Claude Guardian review Qwen's delivery.

Reviewers receive read-only access to the exact delivery snapshot and return findings to an assigned review-artifact location. They must not repair the author's files during review. Hold fixes until review finishes, then return them to the owning worker.

Use the existing five anchored scoring dimensions:

- Useful outcome.
- Learning depth.
- Verification rigor.
- Token economy.
- Anti-Asch truth grounding.

Record empirical validity separately from these quality scores. A polished delivery with a falsified hypothesis can still be valuable; a high score cannot turn failed thresholds into corroboration.

Do not count Claude reviewing Claude's own work as independent re-scoring. If the original 033 agreement measurement is claimed, obtain its required independent assessment.

### 8. Integrate accepted work sequentially

Create a dedicated integration branch/worktree. Integrate only accepted deliveries, one at a time, in dependency order.

Before each integration:

- Verify the worker's final state and lease.
- Confirm review findings are resolved or explicitly accepted.
- Create an attributable local commit.
- Inspect conflicts semantically; never blindly select "ours" or "theirs."

Use the current MISSION-034 merge preparation and regeneration workflow for shared indexes, backlog views, and ledger projections. Preserve source events and report unexplained aggregate mismatches rather than silently repairing history.

After each integration, run the relevant integration checks. Run the appropriate combined suite once the batch is assembled.

Keep rejected or blocked work isolated. Do not mark its mission resolved or merge it to make the batch appear complete.

### 9. Close the batch and preserve continuity

For each assignment, record:

- Implementation disposition: accepted, changes requested, blocked, or rejected.
- Empirical disposition: corroborated, falsified, unassessed, or abandoned with reason.
- Evidence paths and exact scope.
- Independent score and measured usage.
- Remaining questions and next discriminating experiment.

Update mission state through the canonical tools. Complete supported checkin/release actions and leave useful predecessor handoffs.

Perform the applicable Protocol 128 retrospective, technical verification, and local persistence steps. Remote persistence remains subject to the required authorization.

Clean up only newly created, fully integrated temporary resources when safe. Preserve unmerged work and do not forcibly delete worktrees.

Return a concise batch report:

- Four-worker dispatch and actual concurrency.
- Accepted/rejected deliveries and empirical verdicts.
- Review scores and substantive defects found.
- Measured cost, latency, interventions, and telemetry limitations.
- Integration branch and commit identifiers.
- Tests passed and unresolved failures.
- Remaining human decision, if publication is ready.
- Recommended next mission based on the evidence.

Do not infer parallel speedup from overlapping timestamps alone. Report a speedup claim only against a valid measured or clearly identified baseline.

**Proceed with orientation, current-state inspection, mission selection, and execution of this bounded batch. Keep the human informed at dispatch, material blockers, review, and integration completion.**
