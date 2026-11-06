# MISSION-056 frozen protocol

Question (ChatGPT's draft, registered unchanged in meaning): can evenly spaced cyclic rhythms survive missing beats better than irregular ones?

## Model (exact)
n-position circular grid; known template with k beats; unknown cyclic phase; the true pattern is a uniformly chosen DISTINCT rotation (R of them, R divides n). Each beat is deleted independently with probability p; no false positives; the decoder knows n and the grid origin, keeps the distinct rotations that contain every observed beat and guesses uniformly, so an observation with c candidates succeeds with probability 1/c. Expected success is computed exactly by enumerating all R rotations and all 2^k masks (`rhythm.py`). Nothing is simulated.

## Held-out design (frozen)
Every pair n=13..24, k=3..min(8,n-1) (72 pairs). Per pair: one even template (positions floor(j*n/k)); up to 40 distinct uniformly sampled k-subsets that are not rotations of the even template (rejection sampling when C(n,k) > 20000, otherwise sampling from the full eligible list; all if fewer than 40; per-pair stream `random.Random("73185:n:k")`, i.e. seed 73185); one contiguous clustered control (positions 0..k-1). "Distinct" means distinct subsets; two sampled subsets may be rotations of each other (not excluded by the draft). Irregular success for a pair is the mean over its sampled templates (equal weight per template).

## Primary metric and decision (frozen, in `run.py`)
At p=0.25, for each pair: even success minus mean irregular success; primary = unweighted mean over pairs. FALSIFIED if primary < 0.05; otherwise NOT FALSIFIED (supported within the tested domain). Pairs with no eligible irregular template (only k=n-1, which cannot occur in the held-out range) would be excluded and reported.

## Secondary (descriptive only; cannot rescue a failed primary)
p=0.10 and p=0.50; split by whether the EVEN template has rotational symmetry (R<n); clustered control vs irregular; empty-observation probability p^k; per-pair spread.

## Checks (must pass before the held-out run; `test_checks.py`)
Mask probabilities sum to 1; p=0 gives 1; p=1 gives 1/R; rotation invariance; an independently written exact-Fraction brute force (`brute.py`) agrees to 1e-12 on tiny cases; a hand-worked n=4 example; sampling rules and determinism.

## Development disclosure
`python3 run.py dev` (n=8..12) was run and inspected for debugging. It exposed one edge case (k=n-1 has no irregular template), now handled. The dev primary difference was +0.008 with a strong split: even templates that are rotationally symmetric won (+0.136), asymmetric even templates lost (-0.094). The threshold, decoder and aggregation were NOT changed in response. The held-out outcome may differ because n=13..24 has different symmetry composition; no expectation is built into the code.

## Budget
Model tokens: <= 20,000 additional. Experiment wall time: <= 15 minutes (dev ran in 0.3 s).
