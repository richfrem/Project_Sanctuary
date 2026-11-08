# MISSION-067: Random change ringing as a shuffle

**Verdict: FALSIFIED** (gate passed; H1 held, H2 failed). Freeze commit `abda842f3` hashes protocol.md, evaluate_ringing.py and discovery.md. Discovery provenance is in `discovery.md`.

## Question
In change ringing each bell moves at most one place per row, so a change is a set of disjoint neighbour swaps. If a band rang *random* changes, how fast would the bells become well mixed, compared with the classic slow shuffle that swaps one random neighbouring pair at a time (lazy)? Does parallelism save work, or only time?

## Result (exact, `results.json`, `dev_results.json`; TV <= 0.25 from rounds)
| n | t_C (changes) | t_A (lazy steps) | speedup R | swaps C | swaps A | C/A swaps |
|---|---|---|---|---|---|---|
| 4 | 5 | 11 | 2.20 | 6.3 | 5.5 | 1.14 |
| 5 | 10 | 25 | 2.50 | 14.3 | 12.5 | 1.14 |
| 6 | 17 | 49 | 2.88 | 28.3 | 24.5 | 1.16 |
| 7 | 24 | 85 | 3.54 | 45.6 | 42.5 | 1.07 |
| 8 | 34 | 135 | 3.97 | 73.2 | 67.5 | 1.08 |

- **H1 held**: the time speedup grows with n (2.88, then 3.54, then 3.97).
- **H2 failed**: random changes use 7-16% *more* total swaps than one-at-a-time swapping. Ringing's parallelism buys time, not work. The speedup roughly tracks swaps per step (s_C = 1.67, 1.90, 2.15 against the baseline's 0.5 per step).

## Surprise
The swap overhead fell from 1.16 (n=6) to 1.07-1.08 (n=7,8). With only two held-out sizes and integer mixing times, I cannot tell whether it tends to 1 or stays above it.

## Limitations
- Exact for n <= 8 only, with no asymptotic claim. Mixing times are integers, so ratios carry granularity error of about 1/t.
- The random chain borrows ringing's move set but not its rules: real ringing forbids repeated rows and uses methods, not chance.
- The comparison is asymmetric by design: chain A is lazy (otherwise periodic), chain C is not.
- No independent model review: the computation is deterministic and gate-checked, and headroom was limited (claude ~33%). The literature on parallel adjacent-swap shuffles was not checked, so this is "new here" only.

## Association assessment
The association was **generative**: ringing's one-place rule plus bee-style parallel recruitment produced the time-vs-work question. It was not tested as a transfer that improves anything. The ringing-to-shuffle analogy breaks at the no-repeat rule (see discovery.md).

## Reproduce
`python3 evaluate_ringing.py --mode heldout` (stdlib, ~5 s CPU).
