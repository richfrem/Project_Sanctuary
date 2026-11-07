# MISSION-062 Registration Design Note

**Mission:** `LEARNING/missions/MISSION-062.json` (PROPOSED, verdict null). **Registered:** 2026-11-07 by claude-code; the article and question are the registering agent's own choice (not a copy of MISSION-061's framing).
**Not started:** no computation, pilot, dataset, script or worker dispatch exists for this mission. No sumset size was counted by the registrant; the only numbers below come from the paper's text. Bibliographic claims are context, not evidence for the hypothesis.

## 1. Selection record (at most five abstracts inspected)

I first browsed titles only (the 100 most recent math.CO entries, 5-6 Nov 2026), then read exactly five abstracts:

| arXiv | Title | Why considered | Disposition |
|:---|:---|:---|:---|
| 2611.07303 | Gaps in the Most Frequent Sizes of Integer Sumsets | Proven theorem with explicit structure; authors call their threshold "far from sharp" | **Selected** |
| 2611.08230 | The probabilistic bound for perfect hashing is sharp only for binary alphabets | Settles part of a conjecture | Rejected: asymptotic proofs; no cheap local test of the claim |
| 2611.06362 | Four-colour counterexamples for monochromatic path partitions | Smallest colour count where a 1989 conjecture fails | Rejected: claim is a construction for n >= 118 whose verification needs search at scale; abstract gives no method |
| 2611.08079 | Idiosynchromatic Poetry | Delighted by the title; exact Ramsey-type values for ordinals | Rejected: infinite combinatorics, nothing computable here; the arXiv page also shows a 2014 journal reference that conflicts with the 2026 submission (metadata anomaly, unverified) |
| 2611.05745 | A toric approach to social choice theory | Convex geometry meets voting rules | Rejected: the abstract does not state its results, so no defensible question could be formed |

## 2. What attracts me (my own interest, not a value judgement)

A plain counting question (how big is the h-fold sumset hA of a k-element set A?) has an orderly hidden answer: across all k-subsets of {1,...,q}, the most frequent sizes are a short explicit list separated by simplex numbers, and Nathanson's triangular numbers are the k = 4 case. What pulls me in is the gap between a result and its reach. The theorem is machine-checked in Lean, so the open uncertainty is not "is it true" but "where does it start being true". The authors prove the ranking only when h = o(q^(1/(2k-1))), give an explicit sufficient threshold they describe as "far from sharp" (about 1.4 x 10^8 for (h,k) = (3,4), against a computed threshold of 16), and list "how fast h may grow with q" as an open question. That is a precise, checkable place where a proof's conditions and reality may differ, which suits this project's habit of surfacing such gaps. It is also unrelated to AI, and a finite computation settles it without statistics.

## 3. Verified bibliographic details

*Gaps in the Most Frequent Sizes of Integer Sumsets*. Authors: Saad A. Awan, Steven J. Miller, Linh Nguyen, Steven Senger, Elvis Wong. arXiv:2611.07303, v1 submitted Thu 5 Nov 2026 01:38:11 UTC. Primary math.CO, cross-list math.NT; MSC 11B13, 11B30. Links: https://arxiv.org/abs/2611.07303 and https://arxiv.org/html/2611.07303v1 (PDF at /pdf/2611.07303). The abstract page and the HTML text were fetched on 2026-11-07. **Reading limits (disclosed):** read through a fetch tool that summarizes with a small model; the extracted text appeared complete (abstract, Sections 1-5, questions, references) but quotes were short and transcription errors are possible. The Lean ancillary files and the PDF were NOT opened. The four threshold values and the formula for the candidate sizes below must be re-checked against the PDF before the protocol is frozen.

## 4. What the paper claims (author-reported; theorems machine-checked per the authors)

- hA is the set of sums of h elements of A with repetition. The family is the k-element subsets of [q] = {1,...,q}. M_{h,k} = C(h+k-1, k-1) is the largest possible size.
- Candidate sizes P_l = M_{h,k} - M_{l-1,k} = C(h+k-1, k-1) - C(l+k-2, k-1) for l = 0,...,h-1. For k = 3: P_l = C(h+2,2) - C(l+1,2); for (h,k) = (4,3) the paper lists 15, 14, 12, 9.
- Corollary 4.1: if h = o(q^(1/(2k-1))) then for all large q the h most frequent sizes of |hA| are exactly P_0,...,P_{h-1} (each listed value beats every unlisted value; the corollary does not state the order among them). The gaps P_l - P_{l+1} = C(l+k-2, k-2) are (k-2)-dimensional simplex numbers; triangular for k = 4.
- Theorem 1.1 (h = o(q^(1/k))): about q^k/k! sets are B_h-sets; sets with an unlisted size number O(h^(2k-2) q^(k-2)); sets of size P_l number Omega(q^(k-1)/(h-l+1)).
- Remark 4.2: an explicit sufficient threshold q > 2^k k! h (2h+1)^(2k-2), described as "far from sharp"; the Lean formalization uses 70 times that. Arithmetic check: for (3,4) it is 1152 x 7^6 = 135,531,648, matching "about 1.4 x 10^8".
- Computed threshold table q0(h,k), the smallest q beyond which Corollary 4.1's conclusion holds: (3,3) -> 5, (4,3) -> 6, (2,4) -> 8, (3,4) -> 16, with failures at q = 4, 5, 7, 15 respectively; verified for q <= 120, 120, 60, 50.
- Open Question 1: improve q0(h,k) and find how fast h may grow with q.
- Not found in the text read: any table of q0(h,3) for h >= 5.

## 5. Hypothesis (mine) versus the paper

The paper proves the ranking in a thin region and tabulates four small cases. The hypothesis below is the registrant's extension and is not claimed by the paper:

> For k = 3 and every h in {5,...,16}, at every q in [3h, 250] the h most frequent values of |hA| over the 3-element subsets A of {1,...,q} are exactly P_0,...,P_{h-1}, each occurring for strictly more sets than every other size. Equivalently, the empirical threshold q0(h,3) is at most 3h in this range, far below the h^5 scaling the proof requires.

Why 3h: the two published k = 3 cells give ratios q0/h of 1.67 and 1.5, so 3h leaves slack of about 2x above the published data. A heuristic (not evidence, unverified): reduced to the relation d.A = 0, the non-B_h fraction of 3-subsets should scale like h/q, which suggests a roughly linear threshold. The ranking also needs the P_l classes (l >= 1) to dominate every other class, which the heuristic does not address. My honest expectation is close to even odds; a falsification would be as informative as a corroboration because it locates a real boundary.

## 6. Design (not executed)

1. **Freeze first.** Write `protocol.md` and `freeze.json` (definitions, ok-test, constants, region, budget, analysis) and commit them BEFORE any held-out run, in a separate step from the run command (MISSION-056 lesson). Re-check the formula for P_l and the four table cells against the PDF at this step.
2. **Instrument.** Exhaustive counts f_q(v) = #{A in C([q],3): |hA| = v} accumulated by largest element so one pass serves every q <= 250; sumset sizes by exact deduplication of the C(h+2,2) sums x.A. Preferred: vectorized NumPy or a small C program; a separate pure-Python reference implementation for small cases.
3. **Validity gate (must pass or the run is UNASSESSED):** (i) reproduce the four published cells: first q after which ok holds for all larger q tested equals 5, 6, 8, 16 for (h,k) = (3,3), (4,3), (2,4), (3,4), and ok fails at q = 4, 5, 7, 15; (ii) the fast and the reference implementation agree exactly on f_q for every h <= 6, q <= 40; (iii) for E = {1,2,3}, |hE| = 2h+1 for h >= 2.
4. **Definition.** ok(q,h,k) is true iff every P_l (l = 0..h-1) has f_q(P_l) > 0 and f_q(P_l) > max over all other sizes v of f_q(v) (strict; a tie counts as failure).
5. **Development vs held-out.** Development/calibration (published or hand-checkable, may be examined while debugging): the four published cells, k = 3 with h = 2, 3, 4. Held-out primary run: k = 3, h = 5,...,16, q in [3h, 250].
6. **Primary decision.** CORROBORATED if ok holds at every (h,q) in the held-out region. FALSIFIED if any (h,q) in it fails, after the failing cell is recomputed by the independent reference implementation. UNASSESSED if the validity gate fails or the budget is exhausted first.
7. **Secondary, descriptive only, cannot rescue a failed primary claim:** empirical q0(h,3) for h = 2..16, the ratio q0/h, a log-log slope of q0 on h, the comparison with the Remark 4.2 threshold, and a small k = 4 grid (h = 2..4, q <= 80).
8. **Stopping.** Stop on validity-gate failure; stop at the first confirmed counterexample (record the cell and its f_q table); stop on completion of the region; stop at budget. Do not silently shrink the region to fit the budget: checkpoint and report UNASSESSED with the coverage reached.

## 7. Budget (estimated; no pilot run)

Compute: at most 1 wall-hour (about 2 CPU-hours across 8 cores), at most 4 GiB RAM, no network at execution. Rough size: C(250,3) = 2,573,000 triples; the sums per triple over h = 2..16 total 965, about 2.5 x 10^9 sum evaluations. Estimate unverified. Model tokens: at most 40,000 additional, for writing and reviewing code and interpretation; no model call is needed inside the computation. Footprint: `LEARNING/experiments/mission-062/` only; no skill, evaluator or shared-code change.

## 8. Limits of any outcome

A bounded exhaustive check over k = 3, h <= 16, q <= 250 says nothing about larger q or h, other k, or asymptotic truth. CORROBORATED would be evidence that the proof's condition is far from necessary in this range, not a proof. The conclusion depends on the correctness of the instrument (guarded by the validity gate) and of my transcription of the paper's definitions.

## 9. Scoring and readiness (provisional; formula inputs are estimates, not measured ROI)

Strategic leverage 2 (little direct project capability; it exercises the self-directed research loop on an externally checkable question and may add a lesson), epistemic depth 3 (new empirical data the paper does not contain, bounded in scope), cost weight 0.1 (deterministic local compute, as in MISSIONS 056 and 057), operational friction 2 (instrument correctness and the freeze discipline). Priority = (2 x 3) / (0.1 x 2) = 30.0. The formula rewards cheapness: with cost weight 0.15 the priority is 20.0 and with leverage 1 it is 15.0. It is not a statement of importance; MISSION-061 has more strategic leverage and a lower number. Readiness is assessed separately and is high but not complete: tools verified present (NumPy 2.4.6, clang, 8 cores, 16 GiB); remaining prerequisites are the PDF check of the formulas, the frozen protocol, and two implementations plus the validity gate, none of which exist yet.

## 10. Links and non-links

Linked: MISSION-056 and MISSION-057 (practice: exact computation, freeze before scoring, UNASSESSED on invalid instrument, compare against an established baseline), MISSION-060 (illustrative: value, interest and readiness recorded separately). Not linked: MISSION-058 (a discovery pilot over a different source; its verdict concerns that source), MISSION-061 (registered the same day, no mechanism in common), MISSION-041 (searching for counterexamples to agent-written tests, a different object).
