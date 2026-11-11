# MISSION-073: Incomplete Open Tesseracts

**Question.** LeWitt's *Incomplete Open Cubes* (1974) are the connected, non-planar subsets of a cube's 12 edges, counted up to rotation (mirror images distinct): 122. How many connected edge shapes does the 4-cube have under the same rule, and does the same engine reproduce 122 in 3D?

**Verdict: CORROBORATED** (all frozen checks G, H1, H2, H3 hold). Scope: these definitions, on Q3 and Q4 only; an exhaustive finite computation, not a theorem for all n.

## Results (exact, Burnside over the full group; `burnside_n4.jsonl`, `summary_n4.json`)

Class k = number of axis directions the edge set uses. Sets are non-empty, not the complete edge set, and connected.

| | k=1 | k=2 | k=3 | k=4 (full span) |
|---|---|---|---|---|
| Q3 rotations (24) | 1 | 3 | **122** | - |
| Q3 all symmetries (48) | 1 | 3 | 77 | - |
| Q4 labelled sets (identity term) | 32 | 216 | 18,936 | 2,805,229,948 |
| **Q4 rotations (192)** | 1 | 3 | 123 | **14,632,580** |
| **Q4 all symmetries (384)** | 1 | 3 | 78 | **7,338,217** |

- Primary class (frozen): k=4. **14,632,580** rotation orbits; 7,338,217 with reflections, so, by orbit arithmetic (not separately enumerated), 2*7,338,217 - 14,632,580 = 43,854 shapes are achiral (equal to their mirror image up to rotation) and 7,294,363 are chiral pairs (2*7,294,363 + 43,854 = 14,632,580).
- Gate G: Q3 gives 122 (LeWitt's count) and 77, matching an independent canonical-form brute force. 
- H2: frozen interval [14.4M, 14.8M] contained the result (forecast from 300,000 random masks: identity term about 2.8017e9; actual 2.8052e9, 0.12% apart).
- H3: 7,316,290 < 7,338,217 < 14,632,580.
- H1: Burnside sums divisible by |G| for every (group, k); identity counts depend only on k; the engine's counts equal an independent pure-Python brute force for all 308 group elements with at most 12 edge cycles (0 mismatches); 384 elements, 192 with det +1; identity has 32 edge cycles; C and Python property functions agree on 300,034 masks.

## Observations not in the frozen hypotheses (post hoc, arithmetic only)

- The Q4 k=3 count is 123 = 122 + 1 (and 78 = 77 + 1). Every connected set using only three directions lies in one 3-face. The extra orbit is the complete 3-face (12 edges), legal in Q4 because it is not all 32 edges but excluded in Q3 as the complete cube. Exact check: 4 omitted axes x 2 sides x (2,366 + 1) = 18,936 = observed identity count. Likewise k=2: 6 axis pairs x 4 faces x 9 = 216.
- Mirror images of a 3D shape do not merge under 4D rotations here: they are still distinct at k=3 (the facet stabiliser keeps det +1 on the 3D part), though in 4D a rotation can mirror a 3D object by passing it through the fourth dimension. That is only relevant to full-span shapes (k=4), where 7.3M vs 14.6M shows chirality is common.

## What the ringing association did (and did not do)

Generative only. The correspondence "24 cube rotations = S4 = rows of a four-bell extent" gave the question; the count does not test it. `ringing_the_cube.py` (exploratory, unfrozen, `ringing_the_cube_results.json`) verifies it exactly: the 24 rotations act faithfully as all of S4 on the four body diagonals, the three adjacent-place swaps are order-2 rotations (edge half-turns: fix two edges as sets, no vertices), and a 24-row plain-changes extent walks every orientation once. Where the analogy breaks is also exact: the 4-cube has 192 rotations, not |S5| = 120, so five bells do not ring the tesseract.

## Disclosures and limits

- **Missed expectation.** Before sampling I expected the k=4 rotation count in [1e6, 1e7] (about 60% confidence), assuming ~15% of random edge sets are connected. Dev sampling showed 65.2% are; the actual count is 1.46e7. The miss is kept in `protocol.md` and `freeze.json`; the frozen forecast was set after that sampling, so H2 is a test of a sampling-based forecast, not of the original expectation.
- **Order of operations.** The review, protocol, freeze (commit `442ca079a`) and dev validations precede the Q4 enumeration. `check_h1.py` was written after the run but implements the frozen H1-H3 definitions verbatim; `freeze.json` hashes were re-verified unchanged afterwards.
- **Independence.** The Python checker shares no code with the C engine but I wrote both; it re-derives edge indexing, group, connectivity (union-find vs flood fill) and fixed-set counts. It does not independently recount the 2^32 identity term; that term is supported by the axis-symmetry check, the arithmetic reconstructions above, 300,034 differential masks and exact Q3 agreement, not by a second full enumeration.
- **Definition.** "Non-planar" has several 4D generalisations; all k are reported. Counts by k=3 include the complete facet. Not reproduced against any published tesseract count because none was found in sources inspected (abstracts and summaries; no exhaustive novelty claim).
- **Review.** One cheap cross-family call (`qwen/qwen3-coder-next`, 3,348 in / 1,037 out tokens, $0.00123136) found no real defect; its one 'critical' finding was a false positive (see `review.md`). Agreement is not ground truth.
- **Cost.** Experiment: 155.4 s wall, 761 s CPU on 8 threads, peak resident memory 1.77 MB (limit 3600 s, 1 GiB). Paid inference: that one call. Parent-model tokens are not measured here (unknown).

## Files

`discovery.md` (domains, crossings, candidates, selection history), `protocol.md` + `freeze.json` (frozen), `burnside.c`, `aggregate.py`, `independent.py`, `check_h1.py`, `dev_*` (except the two data files below), `fix_py.jsonl`, `burnside_n4.jsonl`, `summary_n4.json`, `check_h1_results.json`, `review.md`, `review_openrouter_raw.md`, `ringing_the_cube.*`.

Reproduce: `cc -O2 -pthread -o burnside burnside.c; ./burnside 4 8 > burnside_n4.jsonl; python3 aggregate.py burnside_n4.jsonl; python3 check_h1.py` (about 2.5 minutes on 8 cores).

## Amendment: removed oversize dev test data (post-merge)

`dev_props_in.txt` (2.7 MB) and `dev_props_expected.txt` (4.1 MB), the 300,034-mask C-vs-Python differential test data, exceeded the soul-record limit (`max_record_bytes` 400,000) at closure and were removed from the repository; no frozen result changes. They are deterministic and regenerate byte-identically (checked): `python3 independent.py props 4 300000 7` writes `props_in.txt` and `props_expected.txt`; their SHA-256 values are the `dev_props_in.txt` and `dev_props_expected.txt` entries in `freeze.json`, which still records them as frozen. All other frozen files are unchanged.

Also corrected: the manifest's `projected_impact` listed `LEARNING/research/` (an optional synthesis that was not written); the entry was removed so it no longer appears to claim a path held by another lease.
