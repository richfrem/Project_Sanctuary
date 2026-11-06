# MISSION-054 prospective observations (rules: `prospective-collection.md`, committed before collection)

| unit | eligible? | outcome | attempts | measured cost / gaps | review | retries / failure opportunities | human interventions during the unit |
|---|---|---|---|---|---|---|---|
| A: fix stale `LEARNING` path reference (test_learning_index) | yes: real pre-existing failing test, shared loop, inside authorization | accepted (3/3 criteria, test passes, 1 file changed) | 1 (claude-haiku-5-5) | 49,364 tokens (input+output+cache-write) attributed from the worker's own Claude transcript; executor result alone reported none; no P02 export (ran before it existed) | none (deterministic acceptance only) | 0 retries; no targeted failure class was applicable, so 0/0 opportunities | 0 observed |
| neutral-repo demo tasks (rename cnt, rename avg, controlled re-run) | NO: created to demonstrate the loop (rule 1) | reported in `practical-demo/` | 2 / 2 / 1 | measured, see README | independent reviewer on the re-run (37,651 tokens, measured) | n/a | see README |

Count of eligible units so far: 1 of 12. Task B (the `run_agent` failure-echo test) was identified but not run. Claim B stays UNASSESSED: (i) no defensible baseline, (ii) overhead cannot be computed from one unit (no review/verification model tokens), (iii) denominators unmet. Claim A not run.
