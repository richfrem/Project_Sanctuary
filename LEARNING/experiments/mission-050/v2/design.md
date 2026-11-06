# MISSION-050 v2 preregistration

Frozen before any v2 measured run. V1 evidence and v1 policy remain unchanged.
Authority: MISSION-050 manifest; 20% completion gain, <=5% acknowledged loss,
<=10% checkpoint inference-token overhead, <=3% interruption-free throughput loss.
Fail the candidate if any primary threshold misses. Report simulation separately.
No further candidate tuning after held-out evaluation. No favorable seed selection.

Prediction: fixing zero-hazard cold start will cause earlier checkpoints, but the
9-cost overhead floor probably still exposes too much progress to meet <=5% loss;
20% gain over the strongest fixed baseline is unlikely. The real arm may also miss
the control throughput gate. These are falsifiable predictions, not measurements.

## Simulation
30 tasks each development, held-out and interruption-free control; retain v1's
1200 ticks, 300 goals, six hazard profiles, 17 fixed baselines and acknowledgment,
atomic commit, final flush and independent SHA-256 receipt semantics. Development
task seeds 110000..110029, schedules 120000..120029. Held-out task seeds
130000..130029, schedules 140000..140029 (all distinct from v1). Controls reuse
held-out workloads without interruptions. Prior is a stated pseudo-count, NOT fit:
rate=(observed_interruptions+1)/(elapsed_ticks+40). V2 otherwise retains v1 formula
max(sqrt(2*cost*value_rate/rate),9*cost*value_rate). Immutable observation has ONLY
rate, pending value, value rate and checkpoint cost. Policy receives no schedule,
seed, task, split or elapsed clock. Pooled and worst-task diagnostics as v1; best
fixed selected on development, held-out retrospective best also disclosed.
Synthetic overhead is ticks and cannot establish token overhead or model recovery.

## Real-agent arm
30 independently seeded multi-stage tasks PER split: development 210000..210029,
held-out 230000..230029, controls reuse held-out. Each task provides 100 sequential
subgoals, cycling modular addition, reversing two digits, and sorting two digits.
Initial state and operation operands come from task seed. Each subgoal uses exactly
one actual Qwen model call. Checker recomputes the answer from seed, stage and prior
verified state, requiring exactly two digits (whitespace stripped). Incorrect calls
consume budget, do not acknowledge progress, retry same subgoal on later calls.
All verified subgoals have equal value 1. Keep raw requests, responses and usage.
No memoization or replay as a substitute for actual policy calls.

Model: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit, human-managed shared endpoint
http://127.0.0.1:8080/v1. Every request, including discovery, exclusively locks
/tmp/sanctuary-mlx.lock using Python fcntl.flock; serialize calls; never manage server.
Temperature 0, max_tokens 24. System work prompt: 'Follow the instruction. Reply
with exactly two digits and nothing else.' User work prompt: 'State: {state}. {op}
Write the resulting two-digit state, padding with a leading zero when necessary.'
Operations: 'Add {operand} modulo 100.' / 'Reverse the two digits.' / 'Sort the two
digits in ascending order.' Explicit state in every request is agent working memory.
Checkpoint system prompt: 'Copy the requested checkpoint exactly. No explanation.'
User: 'Return exactly this JSON: {compact JSON with cursor and state}'. Checkpoint
succeeds only on JSON equality; host independently validates all saved receipts and
atomically replaces a per-trial durable snapshot file. Model need not invent hashes.

Per-trial budget 1000 measured prompt+completion inference tokens, no free checkpoint.
Policies: end-only; fixed steps 1,2,4; fixed-time 240,480 consumed token intervals;
adaptive v2. In this arm 'time' is resource-token time, not wall seconds (shared
server contention makes elapsed wall time an unsuitable matched resource).
Best tested fixed policy selected by total verified durable completions on development,
then lower checkpoint tokens, then lexical name; freeze for held-out. Also report
retrospective strongest fixed per split, without feeding it to policy.
Adaptive prior: one interruption per 400 inference tokens; observed rate
(events_seen+1)/(measured_tokens_spent+400); checkpoint cost initial 100 tokens,
then ceiling of mean actual checkpoint call cost; progress rate = acknowledged
subgoals / actual work-call tokens (including retries), pending value = volatile
verified subgoals beyond durable cursor. Same immutable observation and formula.
These are stated pseudo-counts/constants, not estimates from held-out schedules.

Schedules frozen before runs using schedule seeds 220000..220029 development,
240000..240029 held-out; four token boundaries at 120,270,420,570 plus seeded
jitter [-30,30]. Event kinds rotate by task: context_reset, worker_termination,
quota_denial, vendor_handoff. All four kinds occur once in each interrupted trial.
Events trigger when actual consumed tokens cross a boundary, after response
acknowledgment and before checkpoint commit. Events beyond trial end do not fire.
Multiple crossed boundaries process in order. The agent has no future schedule.
Context reset restores only durable state and discards volatile progress. Worker
termination actually terminates the disposable agent subprocess, then launches a
fresh one restored from disk. Quota denial rejects one next logical request locally,
with zero server tokens and rollback; actual quota outages are not induced. Vendor
handoff launches a fresh agent subprocess with durable state and an alternate
system-role label, using the SAME Qwen model; this is simulated vendor handoff,
NOT evidence for cross-vendor recovery. Never kill/reconfigure the server. Recovery
occurs at completed-call safe points, not inside live generation.

Resource admission uses conservative per-request bound: UTF-8 bytes of message
contents + 128 wrapper allowance + max_tokens. Execute only when remaining budget
covers this bound. If pending and remaining <= work_bound + checkpoint_bound, try
one final checkpoint if affordable, then stop. All unused budget stays in denominator;
report it. Bounds are admission guards, NOT measured token counts; flag any actual
bound/budget overrun. Token overhead = all checkpoint prompt+completion usage /
all actual inference usage, including aborted checkpoints; also report allocated
budget ratio. Verified throughput = durable verified subgoals / identical allocated
budget; control loss versus strongest tested fixed control and dev-selected fixed.
Acknowledgment loss = rollback+terminal unpersisted completions / all successful
acknowledgments, counting redone successes again. Check conservation identities.

Run all development trials, freeze comparator file, then held-out and controls.
630 trial ceiling * 1000 tokens = 630000 measured tokens; total mission ceiling 1M.
Log actual total server usage; never self-report supervisor session tokens. Record
wall time separately. Preserve append-only call logs and per-trial results; resume
only completed trials after verifying frozen hashes. Crash mid-trial requires explicit
partial-trial disclosure; do not silently rerun or choose favorable output. Stop on
unmetered calls or endpoint failure; finish every independent offline portion.

Inference supports a finite benchmark of this local Qwen driver, exact verifier and
safe-point recovery; no claim about all agent tasks, actual vendor changes, live
mid-generation kills, production storage crashes or long quota outages. Simulator
cannot test semantic model errors or inference overhead. Neither arm can establish
universal optimality over all possible fixed intervals. No bootstrap CI required;
report exact n and descriptive finite-sample ratios without population claims.

## Scope and lifecycle
Only experiment directory and three authorized guardian files may change. No git
mutation, manifests, generated indices, ledgers, APM install or coordination writes.
Orchestrator handles integration/ROI ledger. Record prediction versus measured result
here. Control-plane status was INCONSISTENT (missing DB/hooks); no repair authorized.
