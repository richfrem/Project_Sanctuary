# Operational notes (not changes to the frozen design)

- Freeze completed 2026-11-06T05:02:29Z; real run began at 05:03:01Z.
- Initial inference was serialized by the required shared lock. The first 24
  calls measured 177.13292025122792 seconds of aggregate lock wait and 1,429
  inference tokens. Mean request wall time was 7.952236751831758 seconds.
- Around 05:29Z the shared lock cleared. No server lifecycle action was taken;
  the existing frozen run continued and accelerated. Remaining feasibility must
  be assessed from the new measured rate, not the early contention rate.
- Operational deadline guard is separate from design/source hashes. It creates
  an explicitly marked stop file for the next trial, using the frozen runner's
  snapshot-exists guard. This stops before another model call; completed trial
  results and raw evidence are intact. Remove only this marker before resume.
  Never remove a real incomplete snapshot and silently rerun its trial.
- `ps` inventory was unavailable (`operation not permitted`). This did not affect
  experiment execution; workers manage their own client children and never
  discover, terminate or change the human-owned server.
- At the first two fully matched task blocks, adaptive had no early checkpoint
  requests and behaved like end-only. This is a measured design limitation,
  not a reason to alter the frozen prior, prompt, budget or thresholds.
- The 9-cost floor requires pending progress / value_rate >= 9*checkpoint_cost.
  Because pending successful acknowledgments <= all successful acknowledgments,
  and value_rate = acknowledgments / work_tokens, a first adaptive request
  necessarily requires work_tokens >= 9*initial_cost = 900 tokens (when there
  has been no previous checkpoint). The 1,000-token trial's conservative final
  admission/flush guards stop useful work earlier. Thus this short-budget arm
  is poorly suited to distinguish adaptive early scheduling from end-only.
  This algebra is a post-freeze interpretation, NOT a modified run rule.
- Real inference exposes semantic errors excluded from simulation. One raw
  response repeatedly sorted state '49' as '04'. All such failed calls are
  metered, rejected by the recomputing checker and preserved, including retries.
  No prompt editing, retry deletion or best-output selection is permitted.
- Fixed-time here is consumed-token time. This is a resource-matched abstraction,
  not a wall-clock interval policy. Handoff uses a new client and role, not a
  second vendor. Quota denial is injected locally, not an actual quota outage.

Follow-up candidate for the orchestrator (do not create a manifest here):
Derive and test the feasible hazard/checkpoint-cost region for simultaneous
<=5% acknowledged loss and <=10% checkpoint-token overhead. Compare host-written
verified snapshots with LLM-mediated copy checkpoints using a separately frozen,
adequately long budget and fresh held-out schedules. Preserve these failed v2
results; do not optimize this run using its held-out measurements.

Additional interpretation limits:
- V2 simulation uses fresh seeds, so its difference from v1 is not a paired causal
  estimate of the cold-start repair's effect. The defensible direct conclusion is
  that the frozen v2 candidate still misses its own thresholds.
- The simulation receipt payload retains the v1 proof-schema version for exact
  compatibility. Experiment metadata says v2; policy is `adaptive-v2`; fresh task
  seeds and full fixture hashes distinguish the new evidence.
- Inference token accounting includes all prompt tokens in server usage, including
  cached prompt tokens. It does not estimate hardware FLOPs, energy or dollar cost.
- Actual costs are measured for every call; adaptive scheduling initially uses
  the frozen 100-token checkpoint-cost prior. If final flush is its first checkpoint,
  it cannot learn an actual checkpoint cost and use it for an early decision in
  that trial. Therefore the real arm tests this prior-initialized implementation,
  not an ideal calibrated adaptive scheduler.
- The agent is a sequential request driver with explicit state in each prompt.
  Context recovery concerns that driver's verified progress, not arbitrary long
  conversational context, tool environments or a live model KV-cache reset.

## Orchestrator takeover after the worker hit its usage limit (2026-11-06)
- The resumed codex-b session ended at 2026-11-06T06:49:41Z with "usage limit" (no final report). 519 of 630 trials were complete and unique (development 210, held_out 210, control 99); no process was left running.
- The trial in flight when the session died, control-230014-fixed-steps-1, left a partial snapshot (initial state plus 1 receipt, 113 measured tokens, 2 logged calls in real-calls.jsonl, 1 trial_start event). The runner refuses to resume over a partial snapshot. The snapshot was PRESERVED (not deleted) at partial-preserved/control-230014-fixed-steps-1.snapshot.json (sha256 96e6dc42d88eb16c2556fea3b365f696d024a08be13f805c4e1b3d20a27db19d); its 2 calls and 113 tokens remain in the append-only logs and are an interrupted-attempt overhead, excluded from that trial's own totals. The trial was then rerun from scratch by the frozen runner (deterministic at temperature 0; no design, prompt, seed, prior, budget or threshold changed). Interruption cause: external quota exhaustion of the worker session, not an outcome-driven stop.
- The remaining 111 control trials were run by the orchestrator with the same frozen resume command and the same exclusive flock per request.
