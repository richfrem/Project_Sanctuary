# MISSION-056: can evenly spaced rhythms survive missing beats better than irregular ones? (proposed by ChatGPT)

**Verdict: FALSIFIED within the tested domain.** At deletion probability 0.25 the evenly spaced rhythm was, on average, **9.7 percentage points WORSE** than a random irregular rhythm at being reconstructed (primary difference **-0.0972**; the hypothesis needed **at least +0.05**). The even rhythm won in only 23 of 72 (n,k) pairs, and individually cleared the +0.05 bar in just 9.

## What was asked and why
A rhythm is beats on a circular grid of n slots. If a listener knows the rhythm but not where in the cycle it started, and some beats go missing, can they tell where they are? Even spacing is simple, but its repetition might make positions ambiguous; irregular spacing is messy but distinctive. Which wins in a precisely defined model? (Question and interest statement from ChatGPT; registered as MISSION-056.)

## What was done
Exact computation, no simulation: for every template, every distinct rotation and all 2^k deletion patterns, weighted by p^d(1-p)^(k-d); the decoder keeps the rotations consistent with the surviving beats and picks one uniformly (success = 1 / number of candidates). Held-out: all 72 pairs n=13..24, k=3..min(8,n-1); per pair 1 even template, 40 sampled irregular templates (seed 73185, per-pair streams), 1 clustered control. All five pre-registered checks passed first (including an independently written exact-Fraction brute force agreeing to 1e-12, and a hand-worked n=4 case). Held-out run: 1.6 s. Model tokens for the whole mission: small (well under the 20,000 budget; not instrumented separately).

## What the evidence shows
| | mean even minus irregular at p=0.25 | pairs |
|---|---|---|
| all pairs (PRIMARY) | **-0.0972** | 72 |
| even template rotationally symmetric (R<n) | +0.0380 (range -0.047..+0.124) | 28 |
| even template NOT symmetric | **-0.1832** (every pair negative, -0.234..-0.095) | 44 |

Secondary (descriptive, cannot rescue the primary): p=0.10 mean -0.0505, p=0.50 mean -0.0674; the contiguous clustered control is worse than irregular by 0.184 on average (about as bad as even spacing for the asymmetric pairs).

**Concrete example (n=13, k=3).** Even rhythm `X...X...X....` (beats 0,4,8; gaps 4,4,5) versus irregular `XX..X........` (beats 0,1,4; gaps 1,3,9). If only beats 0 and 4 of the even rhythm survive, **2** rotations still fit, so a coin flip decides; if only beats 0 and 1 of the irregular rhythm survive, **1** rotation fits and the position is known. Expected success at p=0.25: even **0.751**, this irregular **0.892**. The even rhythm's repeated gaps (4 and 4) are exactly what makes two surviving beats ambiguous.

**Why the symmetric even pairs look better (and why that is not the same thing as spacing).** When k divides n (e.g. n=16, k=4, beats every 4th slot) the even rhythm has only R=4 distinct rotations, so there is less to get wrong: success 0.997 vs 0.904 irregular. That advantage is a symmetry effect (fewer distinct positions) and, even so, the symmetric subset averages only +0.038, below the bar.

**Exploratory, post hoc (not part of the verdict; `explain.py`).** Across templates of the same (n,k), success rises with the number of *distinct* cyclic distances between beats (mean Spearman +0.63; positive in 94% of pairs): repeated gaps create ambiguity, distinct gaps pin the rotation. This is consistent with the example above but was found after seeing the data.

## What remains uncertain
- Only the stated model: known template, known origin, independent deletions, no false beats, uniform guessing. It says nothing about human hearing, tempo inference, musical quality or memory.
- Irregular templates are a sample of 40 per pair (all outcomes within a template are exact); per-pair spread is in `results-heldout.json`. "Distinct" means distinct subsets, so two sampled templates may be rotations of each other.
- The verdict rests on the unweighted average over pairs; weighting by pair or restricting to symmetric cases would change the numbers (reported above), not the frozen decision.
- **Process note:** the dev run was inspected before freezing (disclosed in `protocol.md`; it already showed the symmetric/asymmetric split and it led to no change of threshold, decoder or aggregation). The first freeze commit was blocked by the repository's index gate and the held-out command ran in the same shell chain before the freeze was committed; the held-out runner verified `freeze.json` SHA-256 hashes of all frozen files at run time and refused to run otherwise, and the files were not edited between hashing and running. The commit order in git (freeze first, results second) does not match the true order (hash file written, results produced, commits made).

## Artifacts (reproduce: `cd LEARNING/experiments/mission-056 && python3 -m pytest test_checks.py -q && python3 run.py heldout`)
`protocol.md`, `freeze.json`, `rhythm.py`, `brute.py`, `test_checks.py`, `run.py`, `results-dev.json`, `results-heldout.json` (machine-readable, every pair), `explain.py` + `explain-exploratory.json`. Needs Python 3 and numpy.

## Next question (successor handoff)
**Which rhythms ARE the most deletion-robust, and are they exactly the ones whose beats all have distinct cyclic gaps?** The exploratory result suggests the best templates are "Sidon-like" (no repeated distance), i.e. modular Golomb rulers / cyclic difference-set-like patterns. Exhaustively search every k-subset for n<=20, find the argmax at p=0.25 for each (n,k), and test whether the maximizer always has all pairwise cyclic distances distinct whenever such a set exists (and what replaces it when none exists). That is a preregistrable, minutes-long computation.
