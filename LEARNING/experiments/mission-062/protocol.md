# MISSION-062 confirmatory protocol — frozen before held-out execution

## Source and interpretation

Primary source: https://arxiv.org/pdf/2611.07303v1, downloaded 2026-11-11; PDF SHA-256 baa72a18211f9555e005dd1c38774b363aba79d2707936af84a0b4969ef57562. Rechecked the actual PDF's Corollary 4.1 (pp.10–11), Remark 4.2 (p.11), and Question 1/table (p.18), using PDF text and screenshot requests. No Lean verification was performed. v2 exists but this protocol uses the registration's v1 explicitly. P_l=binomial(h+k-1,k-1)-binomial(l+k-2,k-1), l=0..h-1; binomial(n,r)=0 for n<r. The source table gives thresholds (3,3)=5, (4,3)=6, (2,4)=8, (3,4)=16; preceding cells 4,5,7,15 fail. The verified upper q bounds are 120,120,60,50. These are calibration, not primary evidence.

## Frozen hypothesis and rule

Exactly the registered hypothesis: k=3, each h=5..16, every integer q=3h..250, all h candidate sizes must have positive frequency and strictly exceed every unlisted size. A tie or missing candidate fails. No frequency ordering among the candidates is imposed. If no unlisted sizes occur their maximum frequency is zero. The region has 2634 cells. CORROBORATED requires every cell; FALSIFIED requires an independently reproduced failing histogram. Any instrument-validity failure, missing coverage or budget exhaustion is UNASSESSED with verdict null. Do not change constants, omit expensive cells, weaken criteria, or use secondary observations to rescue a primary failure.

## Instrument and independence

fast.c enumerates each normalized shape once, enumerates weak compositions into integer sums, and integrates counts by maximum twice. A shape of span v represents q-v actual subsets; prefix integration therefore counts every literal subset, not a sample. reference.py uses actual combinations of {1,...,q} and repeated Minkowski addition with Python sets; it does not normalize, use weak compositions or consume the C output. Evaluator validates total counts against binomial(q,k), malformed/duplicate/missing histogram rows, and resource bounds.

The OpenRouter worker supplied three drafts. The first violated compilation/output contracts; the second had a development-case out-of-bounds loop; the third still counted shapes instead of translations. Before freezing, the supervisor corrected exactly hist=prefix to hist+=prefix and added the source header. No fourth code-generation attempt or fallback was used. Literal-reference comparisons on development h=2,3,4 for k=3 and published h=2,3 for k=4, q<=12, pass. No h>=5 cell has been computed before this freeze.

## Order, validity gate, and stopping

Commit this protocol, freeze.json and hashed instruments in a distinct command BEFORE executing evaluate.py. Rebuild the C executable with the recorded compiler/flags if needed and verify its frozen hash. First compare full fast/reference frequency maps for BOTH k=3 and k=4, h=2..6, every q=k..40. This intentionally strengthens the unspecific registered cross-check and includes h=5,6 only after the freeze. Reproduce all four published threshold cells through the published verification bounds and each preceding failure. For E={1,2,3}, hE is the full interval [h,3h] by induction on interval addition, so its size is 2h+1 for every integer h>=1; executable checks cover h=2..64.

After all validity checks pass, process primary h in ascending order, generating all q histograms for that h once, evaluating q from 3h upwards. At the first failing candidate, independently recompute its literal-subset histogram, then stop if confirmed. If any independent mismatch occurs, mark UNASSESSED. Stop on validity failure, first confirmed counterexample, full required coverage, or the original resource budget. Preserve partial outputs rather than silently rerun or redefine the claim.

## Secondary analyses and limits

Only after a complete passing primary run: finite thresholds q0(h,3) for h=2..16 through q=250, q0/h, descriptive least-squares log-log slope, and the source's explicit sufficient bound 2^k*k!*h*(2h+1)^(2k-2)+1; k=4, h=2..4 through q=80. These are descriptive and cannot imply an asymptotic theorem. Report no conclusion for q>250, h>16, k!=3, or the infinite-domain threshold. Any unexpected pattern is a candidate inquiry, not a new authorized mission.

## Budget and accounting

Original ceiling: 3600 wall-seconds of experiment compute, roughly 2 CPU-hours, 4 GiB RAM; no network or model calls during evaluation. Conservatively reserve 300 of the 3600 wall-seconds for prior development work including a paused macOS sanitizer process that the parent terminated; this is an upper-bound allowance, not measured experiment runtime. The frozen run has at most 3300 additional wall-seconds and bounded child timeouts. Memory is checked as the sum of parent and child maximum RSS (a conservative upper bound on simultaneous use); allocation sizes are bounded by CLI limits. Raw data and results are preserved.

Model ceiling: 40000 additional model tokens for the mission, including orchestration/review/repair; worker output limit was 5000 per call, at most three coding calls; OpenRouter mission ceiling $0.10 plus the client's existing monthly cap. Actual API tokens and catalog-priced cost are recorded from the client, with model-only costs separate from unmeasured parent orchestration. No claim of complete total-cost telemetry or measured savings. Record the Python TLS failure, rejected drafts, supervisor correction, paused sanitizer and recovery separately from scientific results.
