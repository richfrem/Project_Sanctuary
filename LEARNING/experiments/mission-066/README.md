# MISSION-066: Equal-temperament records for 5-limit harmony

**Verdict: CORROBORATED** (gate passed; H1 and H2 held). Freeze commit `8041b5b8b`; `freeze.json` hashes protocol.md and evaluate_et.py.

## Question
If the octave is split into N equal steps, how well can it play the just fifth (3/2), major third (5/4) and minor third (6/5)? Measure: the worst of the three errors in cents, each interval rounded to its nearest step. A division is a *record* if it beats every smaller division.

## Results (`results.json`, `dev_results.json`)
| N | worst error (cents) | fifth | major third | minor third |
|---|---|---|---|---|
| 12 | 15.64 | -1.96 | +13.69 | -15.64 |
| 19 | 7.37 | | | |
| 31 | 5.96 | -5.18 | +0.78 | -5.96 |
| 34 | 3.93 | +3.93 | +1.92 | +2.01 |
| 53 | 1.41 | -0.07 | -1.41 | +1.34 |
| 118 | 0.39 | -0.26 | +0.13 | -0.39 |
| 171 | 0.35 | -0.20 | -0.35 | +0.15 |

Full record list for N 1..200: 1, 2, 3, 5, 7, 12, 19, 31, 34, 53, 118, 171.

## Observations
- The textbook list is right under this measure: 12, 19, 31, 34, 53 and 118 are all records.
- Nothing between 54 and 117 beats 53. Of my allowed set, 65 and 87 never became records.
- 12-TET's weakness is the thirds, not the fifth: its fifth is 2 cents off, but its thirds are 14-16 cents off.
- In units of its own step, 118 (0.038 step) is better than 171 (0.050 step). 171 wins only in absolute cents.

## Limitations
- H2 was a superset prediction (7 allowed values), so it was a weaker test than H1. Its main risk was an unexpected record such as 41, 72 or 99, and none appeared.
- Each interval is rounded on its own (patent rounding), so for some N the thirds and the fifth are not mutually consistent. A consistency-based or weighted measure (for example Tenney-Euclidean) could give a different list. I did not test this.
- 7-limit intervals (7/4) and N > 200 are out of scope.
- No independent model review was run, because the computation is short, deterministic and checked by the gate.

## Reproduce
`python3 evaluate_et.py --mode heldout` (stdlib, < 1 s).
