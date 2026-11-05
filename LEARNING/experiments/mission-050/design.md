# MISSION-050 simulation plan (frozen before the first benchmark)

This is a synthetic checkpoint/recovery experiment, not evidence of real-agent
autonomy, inference-token usage, or vendor handoff reliability. No model dispatch
or network call is involved. Do not infer token counts from simulated units.

Prediction: the adaptive policy will probably miss the 20% improvement threshold
against the strongest fixed baseline, and may miss the 5% acknowledged-loss
threshold. A 10% checkpoint-overhead floor constraint should limit overhead in
most runs, but is not a guarantee under repeated interrupted checkpoints. With
zero observed interruptions, adaptive will do only the common final flush and
should meet the 3% control-throughput threshold. These are predictions, not results.

Freeze: 30 independently seeded tasks per split, 1,200 resource ticks per task,
300 sequential subgoals each, work 3–12 ticks, value 1–5, checkpoint cost 2–6 ticks.
Development seeds: tasks 10000–10029, schedules 20000–20029. Held-out seeds:
tasks 30000–30029, schedules 40000–40029. Controls reuse held-out tasks with
empty schedules. All policies see identical tasks and schedules within a split.
Six schedule profiles rotate across the 30 tasks: stationary low/medium/high,
rising hazard, falling hazard, and alternating bursts. Event labels rotate among
context reset, worker termination, quota denial, and vendor handoff; all have the
same abstract rollback effect. The model does not implement actual vendor calls.

Baselines: end-only; fixed wall-time intervals 8, 16, 24, 32, 48, 64, 96, 128,
256, 512; fixed completed-subgoal intervals 1, 2, 4, 8, 16, 32. Choose the best
fixed policy on development verified completion count (then value, lower cost,
lexical name) and freeze it for held-out comparison. Also disclose the stronger
per-split best fixed comparator, with held-out selection labeled retrospective.
No adaptive parameter tuning, seed search, or schedule-dependent decisions.

Adaptive input boundary: an immutable record of measured past interruption rate,
unpersisted progress value, measured progress-value rate, and checkpoint cost.
The policy cannot receive a task, schedule, clock, RNG, event type, or split name.
It checkpoints when unpersisted value reaches
`max(sqrt(2 * checkpoint_cost * value_rate / interruption_rate),
9 * checkpoint_cost * value_rate)`. If no interruptions have been observed it
waits for the common final flush. The square-root term balances a stationary
hazard estimate with checkpoint cost; the other term targets at most 10% cost.
Neither assumption is guaranteed for these nonstationary schedules.

Each work tick earns proportional subgoal value; a completed subgoal emits an
acknowledgment even before durable persistence. Independent receipt validation
recomputes canonical SHA-256 receipts and checks ordered completion identities.
Interrupted checkpoints retain the previous atomic snapshot, including partial
work. Interruption on the checkpoint's final tick wins over commit. Repeated
completion attempts after rollback are counted again in acknowledgments and
losses. Lost acknowledged count and value include terminal unpersisted receipts;
partial work losses are reported separately. This avoids redefining acknowledgment
as persistence or hiding losses through retries. All final flushing and aborted
checkpoints consume the same budget. There is no free final save.

Primary thresholds: >=20% more verified completed subgoals versus best fixed;
<=5% lost acknowledged count AND value; <=10% checkpoint resource share;
<=3% verified throughput loss in controls versus best fixed. Report pooled ratios
and worst-task values; require every split and every task to pass loss/overhead
gates for a conservative all-threshold verdict. Per-task zero denominators with
zero loss are defined as zero. A split with zero fixed verified completions makes
relative gain undefined and fails the improvement gate. No inference-token or
real-agent threshold can be corroborated by simulated ticks.

Evaluation uses reproducible raw task-level results, schedule/task fixtures,
hashes, policy decision traces, and stdlib verification. Record actual wall time
as runtime expenditure only; token accounting remains the supervisor's job.
No ledger, mission manifest, APM installation, or coordination write is permitted
in this worker's scope. The handoff will call out those deferred integration steps.
