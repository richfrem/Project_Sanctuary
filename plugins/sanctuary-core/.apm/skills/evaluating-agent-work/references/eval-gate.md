# Eval gate (MISSION-054)

`scripts/eval_gate.py` is now a thin compatibility adapter: the logic lives once in the companion skill `evaluating-agent-outcomes` (`eval_core.py`) and records still go to `.agent/learning/events.jsonl` (MISSION-055). See [shared-loop-adapter.md](shared-loop-adapter.md).

| Failure seen in MISSION-052 | Gate |
|---|---|
| Evaluator bug found after responses were collected | `fixtures`: run the evaluator on synthetic cases first |
| Output contract unmet (58 of 60 Qwen answers malformed) | `pilot` on development items; ids must not be reused as confirmatory items |
| Benchmark claimed behavior it could not observe | `validity`: adherence < 0.8 => `MECHANICS_ONLY`; 0 or ceiling baseline => `UNSCORABLE`; malformed stay in the denominator |
| Corrections miscounted / telemetry showed zero | `intervention` (observed event required), `invest` (missing = unknown) |
| Same failure repeated | `repair record` at the second occurrence; `repair applied` per later opportunity; `repair report` gives recurrence per category |

Generic orchestration checks (ownership before dispatch, executable in the worker environment, invocation-context isolation, unique-unit reconciliation) belong to MISSION-053 in `orchestrating-agent-work` (`dispatch_guard.py`); do not reimplement them here.

## MISSION-004 readiness before any new confirmatory run
1. Output handling reliable on separate development items (`pilot` passes).
2. Pressure design with an interpretable baseline (pilot baseline strictly between floor and ceiling); do not tune pressure only to force a failure rate, and do not reuse pilot items as confirmatory.
3. Clean invocation verified with supported diagnostics (`dispatch_guard.py context-check`).
4. Evaluator and matcher tested on fixtures (`fixtures`).
5. Fresh held-out items and a declared budget, preregistered and committed before scoring.

## Limits
The code is a verified mechanism. Whether it reduces recurrence is longitudinal (MISSION-054 Claim B) and stays unassessed until 12 eligible real work units are observed passively.
