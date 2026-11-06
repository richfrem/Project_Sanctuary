# Adaptive interruption policy: MISSION-050 simulation

`../scripts/adaptive_checkpoint.py` is an offline stdlib experiment. It is not
connected to the guardian lifecycle, coordination board, quota probes, or any
model backend. Resource ticks are synthetic and must never be reported as tokens.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/adaptive_checkpoint.py --output-dir LEARNING/experiments/mission-050
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_adaptive_checkpoint.py --basetemp=LEARNING/experiments/mission-050/pytest-tmp
```

Read `LEARNING/experiments/mission-050/design.md` for the frozen prediction and
parameters, `results.md` for observed thresholds, `summary.json` for machine-readable
gates, and `evidence.json.gz` for full fixtures and task-level evidence. Decompress
with stdlib `gzip.open(path, 'rt')` and `json.load`. The gzip timestamp is zero for
reproducibility. `execution.json` measures actual benchmark wall time and hashes.

The runner owns task and event fixtures. `AdaptivePolicy.should_checkpoint`
accepts **only** a frozen, slotted `AdaptiveObservation` of four numeric fields:

- interruptions observed / resource ticks elapsed;
- unpersisted progress value, including proportionate partial-subgoal work;
- cumulative value earned / work ticks executed, counting retried work;
- known synthetic checkpoint cost in ticks (constant within each task, not a
  claim of empirical cost calibration).

No clock, event list, future task, RNG, split identifier, or runner reference is
passed to adaptive code. The policy is stateless, and tests inspect its field and
global access plus compare identical pasts under different future schedules.
This is an architectural interface boundary, not a security sandbox for hostile
Python code. The deterministic task builder and schedule generator use distinct
local RNGs and disjoint development/held-out seeds. Development selects the fixed
baseline; adaptive has no tuned parameter or prior hazard estimate.

For past hazard `h`, progress value rate `v`, cost `c`, and unpersisted value `p`,
the policy requests persistence if `p >= max(sqrt(2*c*v/h), 9*c*v)`. For `h=0`
or `v=0` it waits. The square-root heuristic assumes stationary hazard; the cost
floor targets 10% overhead but does not guarantee it. Changing values, hazard
shifts, and aborted checkpoint retries can violate that target. A common runner
rule attempts a metered final flush near budget exhaustion for **every** policy.
That rule knows the allocated budget; adaptive code never sees it or future events.

Work is sequential. Each completed subgoal immediately emits a volatile
acknowledgment and a canonical receipt. A verifier independently recomputes the
receipt from task facts, rejects malformed order/content, and validates snapshots
at commit, recovery, and final scoring. A checkpoint atomically persists completed
receipts and partial work only after its full cost is paid. Events on the final
checkpoint tick abort the commit. Recovery restores the last committed snapshot;
all acknowledged work discarded since it, including retries, is counted as lost.
Terminal unpersisted acknowledgments also count as lost. The final save costs
ticks and cannot run beyond budget. There is no free end-of-run verification save.

Scores use durable verified subgoal count, not summed value or acknowledgments.
Count and value losses are both disclosed. Checkpoint overhead is spent checkpoint
ticks / allocated ticks; throughput is verified subgoals / allocated ticks. Raw
elapsed/work ticks allow alternate denominators to be audited. Empty acknowledgment
denominators with no losses are zero; a zero baseline makes relative gain undefined
and fails its gate. Pooled results and worst-task diagnostics are reported without
rounding before comparisons. No confidence or real-agent generalization follows
from these fixtures. All event labels currently have identical rollback behaviour;
outage duration, vendor-specific semantics, storage faults, verifier cost and
semantic correctness of agent output remain untested.

Supervisor integration: inspect the evidence before updating mission status.
This worker must not edit manifests, ledgers, shared coordination state or installed
skill copies. APM reinstallation and canonical session accounting remain supervisor
work after review; this experiment creates no replacement runtime checkpoint hook.

## MISSION-050 v2 completion pass (2026-11-05)

`AdaptivePolicyV2` retains the v1 decision formula and immutable four-value input
boundary. The simulation runner supplies the preregistered pseudo-count hazard
`(interruptions + 1) / (elapsed_ticks + 40)`, so zero observed interruptions no
longer prohibit an early checkpoint. `AdaptivePolicy` and its default runner
semantics remain intact for exact v1 replay. Explicit priors must be finite and
positive. The policy still has no task, future schedule, seed or split input.

The new frozen design and evidence are beside v1 in
`LEARNING/experiments/mission-050/v2/`. The full v2 simulation still fails:
development verified-subgoal gain -14.8567%; held-out -11.6423% versus the
**development-selected** fixed-step-2 comparator (held-out split-best differs).
Acknowledged count loss is 47.7307% / 46.2478%; interruption-free throughput loss
is 3.5998%. Synthetic overhead is 5.3806% / 5.1306% of allocated ticks. These are
finite simulation results, not inference-token measurements. Cold-start repair
does not repair the candidate's broader loss/overhead tradeoff.

The separately frozen real-Qwen arm meters server usage, checks deterministic
semantic subgoals and saves raw responses. Consult `v2/results.md` and
`v2/real-summary.json` for completion status and exact scopes; partial trials or
splits cannot support a full-arm verdict. Worker termination targets only the
disposable client process. Context reset, quota denial and handoff are injected
at completed-call safe points; the same Qwen model is used in both handoff roles.
Actual vendor changes and mid-generation recovery remain outside those claims.
Never manage the shared human-owned MLX server. Every request must exclusively
lock `/tmp/sanctuary-mlx.lock` using Python `fcntl.flock`.
