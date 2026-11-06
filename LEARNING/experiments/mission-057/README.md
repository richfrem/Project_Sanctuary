# MISSION-057: do consecutive primes remember their last digit beyond a random model, and does that memory fade? (my own choice)

**Verdict: FALSIFIED (on H2); H1 held.** Real primes repel their own last digit by a clearly measurable amount beyond the random model, **but that extra repulsion did not fade between primes of size ~10^5-10^6 and ~10^7-10^8**; for the digits 1 and 9 it grew slightly. What *does* fade is the part of the deficit that a plain random model already explains.

## Why I chose this
Primes are completely determined, yet statistically they look like random numbers, and the textbook "Cramer" random model is how we picture that. I find it striking that something as plain as last digits breaks the picture: a prime ending in 1 is followed by another prime ending in 1 less often than chance suggests (reported by Lemke Oliver and Soundararajan, 2016; cited from memory, not re-verified here). My own curiosity: how large is that memory per digit, does it fade as primes get bigger, and what is the *right* "no memory" baseline?

## What I did (and a correction to my own plan)
Exact data: every prime up to 10^8 (5,761,455 of them; sieve checked against pi(10^6), pi(10^7), pi(10^8) and trial division). For each window I counted, per last digit d in {1,3,7,9}, how often the next prime ends in d again. My first plan compared that with 0.25 ("one of four digits"). The development run overturned the plan: **a random prime-like model itself gives only about 0.16-0.20, not 0.25**, because the next candidate is reached after a geometric waiting time that rarely lands back on the same digit (exact formula q(1-q)^3/(1-(1-q)^4); it approaches 0.25 only as primes thin out). So 0.25 is the wrong "no memory" baseline at these sizes. I redefined the comparison to the random control (and verified the simulated control against that formula) **before** computing the held-out windows; the original plan and the change are disclosed in `protocol.md`.

## What the evidence shows
| window | primes | digit | real P(d->d) | random control | extra repulsion R |
|---|---|---|---|---|---|
| 10^5-10^6 | 68,906 | 1 / 3 / 7 / 9 | 16.6 / 14.5 / 14.9 / 16.2 % | 17.8 / 17.7 / 17.5 / 17.6 % | 1.2 / 3.2 / 2.7 / 1.4 pp |
| 10^7-10^8 | 5,096,876 | 1 / 3 / 7 / 9 | 17.8 / 16.8 / 16.8 / 17.8 % | 19.6 / 19.6 / 19.6 / 19.6 % | 1.8 / 2.8 / 2.8 / 1.8 pp |

- **H1 holds:** at 10^7-10^8 every digit is below the random control by at least 1.8 percentage points (statistical error ~0.03 pp), well past the 0.5 pp bar.
- **H2 fails:** R did not shrink with scale for every digit. Change from the small to the large window: digit 1 +0.65 pp, digit 9 +0.44 pp (both larger than the ~0.3 pp small-window error), digits 3 and 7 -0.40 and +0.10 pp (within noise). The "memory" is persistent over these two decades.
- **What fades is the trivial part.** The naive deficit from 25% shrinks from ~8-10 pp to ~7-8 pp; of that, the part the random model explains falls from 7.3 pp to 5.4 pp as primes thin out, while the unexplained part stays at 1-3 pp.
- **The memory has structure** (full 4x4 matrices in `results-heldout.json`). At 10^7-10^8, after a prime ending in 1 the next ends in 3 or 7 about 30% of the time each, in 9 about 21%, in 1 only 17.8%; after 9 the next ends in 1 about 33% but after 1 the next ends in 9 only 21%. The transitions are not symmetric: going 9 -> 1 is much more common than 1 -> 9, and digits 3 and 7 repel themselves more than 1 and 9 do.

## What remains uncertain
- Two windows only (a decade-and-a-half apart), base 10, primes up to 10^8. "Does not fade here" is not "never fades"; theory suggests a slow decay at much larger scale.
- One random model (Cramer with density 2.5/ln n). Others, such as models that also respect small-prime sieving, might explain part of the repulsion; that question is exactly what the repulsion R is measuring against.
- The 0.5 pp and 0.3 pp thresholds were set after seeing the development window (disclosed). The held-out numbers were not seen before the freeze commit (`3ed1a4456` precedes the run).
- The literature reference is from memory.

## Artifacts (reproduce: `cd LEARNING/experiments/mission-057 && python3 -m pytest test_checks.py -q && python3 run.py heldout`, ~17 s, Python 3 + numpy)
`protocol.md`, `freeze.json`, `primes_lastdigit.py`, `test_checks.py`, `run.py`, `results-dev.json`, `results-heldout.json`.

## Next question (successor handoff)
**Does the extra repulsion R settle at a constant, grow, or decay like 1/ln x at larger scales?** Extend with a segmented sieve to 10^9 and 10^10 (minutes), preregister three candidate laws (constant, c/ln x, c*ln ln x/ln x) and a fit criterion on windows [10^8,10^9] and [10^9,10^10], and repeat in base 3 or 7 to see whether the asymmetry 9->1 vs 1->9 is a base-10 artifact (it should map to which residue gaps are common).
