---
status: OBSERVED
observed_at: 2026-11-08
source_mission: MISSION-060
---

# Prompt budgets and post-call guards are different controls

MISSION-060's first Flash tool-enabled worker received a 25,000-token composite
item budget, then used 125,836 additional tokens over 31 requests in 338.7 seconds.
The trusted runner detected the overrun after the call and stopped. It did not
provide an in-generation token interrupt. Successor/reviewer stages never ran.

The frozen metric was final context minus initial context plus all outputs:
91,934 − 12,906 + 46,808 = 125,836. Accumulated input was 1,374,639 tokens and is
reported separately. This is one observed budget failure, not evidence that all
workers overrun, nor a measured full economic cost.

For future bounded experiments, distinguish requested budget, enforced runtime
limit and after-the-fact accounting. Verify a supported stop mechanism on
development cases before confirmatory execution. If the client offers no reliable
in-call limit, design a bounded work unit/request schedule and disclose any
remaining overshoot bound before freezing; do not claim a prompt is enforcement.

Post-call refusal worked correctly. The selection-policy hypothesis remained
UNASSESSED because the required cohort comparison did not complete. An invalid
experiment is not a falsification of that scientific hypothesis. Preserve costs
and partial artifacts, including failed attempts; do not quietly omit them or
retune criteria after seeing outcomes. Model-generated findings from this worker
were not independently reviewed and are not ratified knowledge.

Evidence: [closeout](../experiments/mission-060/CLOSEOUT.md),
[usage](../experiments/mission-060/usage-log.jsonl),
[stop record](../experiments/mission-060/verification/phase5-ITEM-01.txt),
[analysis](../experiments/mission-060/results.json).
