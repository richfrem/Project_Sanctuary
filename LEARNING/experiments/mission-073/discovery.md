# MISSION-073 discovery note (associative mission discovery)

Authored summary of the exploration phase, not a transcript of reasoning. Selection method: **deliberate diverse choice, not random sampling** (no sampler, no seed). Details drawn from learned knowledge are marked (?) when I am unsure; only the final seed's load-bearing claims were verified.

## Domains (4)

| Domain | Concrete phenomenon | Mechanism / constraint | Surprise / tension |
|---|---|---|---|
| Change ringing (physical, institutional) | Ringers on 4-12 heavy tower bells ring every permutation ("extent") without repeating a row | A bell can move at most one place per row (rope and wheel physics); methods are memorised rules, not sequences; "truth" = no repeated row | Stedman's 1668 "plain changes" is the Steinhaus-Johnson-Trotter algorithm found ~300 years earlier (?); ringers want the coincidence (coming round) that cicadas avoid |
| Periodical cicadas (biological) | 13- and 17-year Magicicada cohorts emerge together | Prime periods minimise coincidence with periodic predators (order of a cycle product = LCM of lengths) | The prime-period story is contested (hybridisation and Allee-effect accounts exist) (?) |
| Sol LeWitt, Incomplete Open Cubes (art) | 1974: 122 structures made from removed edges of a cube, executed from instructions | Connected, non-planar edge subsets, equal under rotation; mirror images distinct | An artist's hand enumeration of a finite space; later checks confirmed 122 and found one presentation error |
| Venetian doge election (historical, institutional) | 1268-1797: ten alternating rounds of lot and vote, 30 -> 9 -> 40 -> 12 -> 25 -> 9 -> 45 -> 11 -> 41 -> 1 (?) | Randomised small committees plus supermajority thresholds resist faction capture | Mowbray and Gollmann (2007) call parts of it security theatre; 25 of 41 may be a stability optimum (?) |

## Crossings (8) and what became of them

1. **Ringing x cicada**: lead-head permutation order = LCM of cycle lengths; coprime cycles give the longest courses. Ringers call this differential methods; their maximum course relates to Landau's function. Search confirmed it is known. *Analogy breaks*: ringers need to come round, cicadas must not coincide. Dropped as already known.
2. **Ringing x LeWitt**: the 24 rotations of a cube are S4 acting on the four body diagonals, so a four-bell extent is a tour of every cube orientation using edge half-turns. *Kept: generated the mission's question.*
3. **LeWitt x Venice**: both are procedures executed by others under rules designed to prevent a bias (assistant's interpretation vs faction). Shared vocabulary only; dropped.
4. **Cicada x LeWitt**: 17 = 2^4 + 1 and the tesseract has 16 vertices. Numerology; rejected as shared vocabulary, not mechanism.
5. **Ringing x Venice**: random committee selection as a mixing chain. Overlaps MISSION-067 (ringing as a shuffle). Dropped.
6. **Constraint removal on ringing**: allow any permutation per row. Truth becomes trivial; no question.
7. **Representation change**: render an extent as a LeWitt-style instruction piece. Kept as a small exploratory artifact (`ringing_the_cube.md`).
8. **Deceptive similarity (second generation of crossing 2)**: in 3D the rotation group (24) equals |S4|, so ringing four bells is exactly the cube's rotations. In 4D the rotation group of the tesseract has order 192, but five bells give |S5| = 120. The analogy fails exactly one dimension up. That failure is what made me want to count LeWitt's structures in 4D. (A fair development of an initially awkward idea: the failure itself, not the fit, is the question.)

## Three candidate briefs

- **A. Cicada courses**: how close can ringable methods get to Landau's bound? Attraction: the shared LCM arithmetic. Prior work: known to the ringing community (differential methods, max-course results). Risk: only re-derives. Not selected.
- **B. Incomplete open tesseracts** (selected): exact count of connected edge subsets of the 4-cube up to rotation, with LeWitt's 122 as validity gate. Association changes the question (ask it in 4D, where the ringing correspondence breaks) but not the method. Next step affordable (minutes of CPU). Risks: the definition of "non-planar" must be generalised; I report every axis-span k so any reading is derivable.
- **C. Venice alternation ablation**: does alternating lot and vote resist factions beyond what the number of stages gives, against a matched all-lot control? That would substantively test a transfer. Prior work: Mowbray-Gollmann, Walsh-Xia, Molinari already analyse iteration and truncation. Not selected; recorded for the inbox.

## Selection history (what the prior-work check changed)

My first-choice seed was the *LeWitt count itself* (reverse-engineering which reading of "incomplete open" yields 122). The check showed 122 is already independently verified (Rozhkovskaya 2014; K-State project; Pitt; Weychert; Vejdemo-Johansson 2026), so I demoted it to a validity gate and moved up a dimension. Searches found no tesseract enumeration in the sources inspected (abstracts and summaries; no exhaustive novelty claim).

## Association status

Generative only. The ringing/S4 correspondence supplied the question; the computation does not test it. One verifiable piece (the 24-row plain-changes tour of cube rotations) is an exploratory side artifact.
