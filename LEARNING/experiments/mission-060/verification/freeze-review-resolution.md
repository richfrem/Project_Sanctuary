# Freeze review resolution

Operational review: one AGY Flash session, no held-out or study calls. Raw review
and router result are preserved beside this file. Tokens were not reported by
the router and remain administrative unknown cost, not zero or study evidence.
The reviewer saw concatenated source and inferred some missing dependencies;
its line numbers are not real per-file anchors. Root decisions follow.

| Finding | Resolution and evidence |
| --- | --- |
| Item cap/B1 denominator omit successor/reviewer | Accepted. common.item_usage now sums all item stages; driver stops on composite overruns; analysis derives cohort costs from the complete stage log. Regression fixture has worker 15k + successor 6k + reviewer 5k and correctly stops at 26k, despite worker below 25k. |
| bool("false") counts as true | Accepted. Required reviewer values must be JSON booleans; malformed evidence produces UNASSESSED, not silently false or true. Regression checks both string false and missing review structure. |
| Missing cwd causes blinding breach | Rejected as stated: common.run_model already defaults cwd to NEUTRAL; common.py was not in the peer packet. Explicit cwd and before/after neutral checks were added as defensive clarity, not a claim that this alleged breach occurred. |
| Missing/duplicate declarer candidates silently dropped | Accepted. Validate full non-self coverage, allowed IDs, uniqueness, integer bounds and explanation before persistence and again before selection. Self-proposal rows remain optional/excluded. |
| B1 None bypasses gate; proposed A starts CORROBORATED | The cited B1 path is unreachable with valid cohort/usage gates, and the proposed A tag is wrong: scenarios emits MET or UNSCORABLE. The real unknown-A fall-through was fixed; corroboration now requires recognized positive A and explicit true B claims. Regression verifies unknown A cannot corroborate. |
| Malformed review crashes analysis | Accepted. Contract errors are retained as item errors and make evidence unassessed. No-file attempts intentionally skip review, retain their cost, and count zero SPO. |

Additional pre-freeze controls: freeze hashes all study code, static configuration,
mission manifest and calibration evidence; mutable usage is prefix-hashed. Runtime
dependencies and CLI versions are pinned. Prospective calls and selection require
origin/main parity; held-out generation refuses an existing result before evaluation.
All held-out guard tests use disposable fixtures and mocked evaluators, never the
real held-out generator. Thresholds and calibrated REVIEW/CALIBRATION text did not
change. Worker budget wording now correctly includes later item stages.

The first expanded 29-test run passed with an unclosed-assessment-file warning;
file handling was corrected before the final run. One multi-file apply_patch was
refused for a mistyped analyze.py path; no partial changes were written, and the
correct patch was then applied. No failure was reclassified as pre-existing.
