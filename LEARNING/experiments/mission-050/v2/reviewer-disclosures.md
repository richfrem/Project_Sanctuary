
## Reviewer-required disclosures (added after the final independent review; see orchestrator-addendum.md)
- The v2 prior (1 interruption per 40 ticks, 400 tokens in the real arm) was chosen after seeing v1 held-out results: v2 is a post-hoc repair of the cold-start defect, not a clean held-out replication. Fresh task seeds only prevent reuse of v1 held-out instances; the statement above that nothing was tuned after seeing these v2 results remains true.
- About 46.1% of the allocated inference budget is unused (339,573 of 630,000 tokens across 630 completed trials); the runner ends a trial at its first final flush.
- Wrong work calls are mostly deterministic repeats: 1,113 of 1,282 repeat an already-wrong (trial, cursor, state, output).
- Interrupted attempt: 2 calls (113 tokens) of `control-230014-fixed-steps-1` are interrupted-attempt overhead, excluded from that trial's totals and from the checkpoint-overhead figures above; see interrupted-attempts.json.
- Control gates: the adaptive policy was behaviourally identical to `end-only` (identical results in all 90 paired tasks), so the 0.0% real control loss passes by construction; by request service time the loss is 1.09% (pass).
- The real-arm verdict is for this frozen design and candidate. At this budget and schedule even the best fixed policy loses 57-58% of acknowledged progress, so <=5% loss was unreachable for every tested policy.
