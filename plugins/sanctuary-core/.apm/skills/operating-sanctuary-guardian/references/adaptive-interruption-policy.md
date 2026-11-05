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
