# MISSION-074: Truchet tiles as critical percolation

Verdict: CORROBORATED (all pre-set hypotheses held). Scope: this model, a periodic L x L torus, L = 64..1024.

## What was done
Sebastien Truchet's tiles carry two quarter-circle arcs; laid at random on a torus they join into closed loops. Theory says the loops are boundaries of critical percolation clusters, hull dimension 7/4. I simulated it in C (`tilings.c`), checked the loop counter against an independent Python walker (`gate.py`, 200 tilings, 0 mismatches), froze protocol and code (`freeze.json`, `protocol.md`), then ran the held-out experiment (`run_heldout.sh`, ~149 s wall, 129 s CPU, 111,000 tilings). Analysis: `analyze.py` -> `heldout_result.json`.

## Results (held-out)
| Claim | Measured (95% bootstrap) | Pre-set pass window |
|---|---|---|
| H1 uniform p=0.5, longest loop ~ L^d | d = 1.7508 [1.7446, 1.7569] | [1.72, 1.78] |
| H2 p=0.5, L=1024, N(>=s)/L^2 ~ s^-alpha | alpha = 1.1359 [1.1328, 1.1389] | [1.08, 1.20] (8/7 = 1.1429) |
| H3a uniform p=0.35, same fit as H1 | d = 1.7529 [1.7467, 1.7589] | [1.70, 1.80] |
| H3b checker q=0.35, same fit | d = 0.3207 [0.3178, 0.3241] | < 1.2 |

Local doubling exponents at p=0.5 (from the mean longest loop): 1.747, 1.758, 1.749, 1.754 for 64->128 ... 512->1024, so no visible drift. Mean longest loop at L=1024: 347,631 arcs (p=0.5), 359,991 (uniform 0.35, 3.6% larger prefactor, same exponent), 587 (checker 0.35).

## What it shows and does not show
- Shows: these finite tori reproduce hull dimension 7/4 and tail exponent about 8/7 within the pre-set windows, and a uniform bias leaves the exponent unchanged while a checkerboard bias destroys it. The bootstrap intervals describe sampling noise only; they do not bound finite-size or window-choice bias, and the pass windows were set from theory with slack, so agreement is not an independent test of 7/4 beyond roughly +-0.03.
- Does not show: anything about open boundaries, other lattices, or the true infinite-lattice limit. Non-contractible loops on the torus are included in the longest loop, and are not separated out. The percolation mapping is background theory, not tested.

## Surprise and recorded miss
My registered H3 (uniform bias 0.35 would be non-fractal, forecast 0.9) was wrong. A 20-sample development run showed it stays critical, because the two tile orientations are dual to horizontal and vertical bonds with probabilities p and 1-p, which sum to 1 (self-dual critical line). H3 was replaced by H3a/H3b before the freeze; the original is kept in `protocol.md`.

## Disclosures
- Freeze order: `freeze.json` (hashes) was written to disk before the held-out run, and I verified afterwards that no frozen file changed. The git commit of the freeze was BLOCKED by the pre-commit gate (index not recompiled) and my chained command ran the experiment anyway; the freeze commit `9a7538984` therefore comes after the run in git history. The commit gate refusing was correct; the sequencing slip was mine.
- Development exposure: I saw dev-sized estimates (H1 1.746, H2 1.122) before fixing intervals; intervals came from theory (7/4, 8/7) with finite-size slack, not tuned to them.
- Raw held-out outputs (13 MB) are not committed (record limit). They regenerate byte-identically with `run_heldout.sh` after `cc -O2 -o tilings tilings.c`; SHA-256 of each file is in `heldout_raw_sha256.json` (determinism spot-checked on two files).
- Review: one cheap cross-family call, no valid findings (`review.md`).
- Exploratory, unfrozen: `render_svg.py` -> `truchet_L40.svg`, one 40 x 40 tiling with its six largest loops coloured (largest 1,180 arcs).

## Reproduce
`cc -O2 -o tilings tilings.c && python3 gate.py && ./run_heldout.sh`
