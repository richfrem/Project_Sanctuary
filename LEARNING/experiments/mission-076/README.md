# MISSION-076: how fast does a cross-reference rescue a dying annotation tradition?

**Question.** A process with an absorbing dead state (a chain of copied marginal notes, an ant trail, a reef patch) sits exactly at its survival threshold. If we add a little extra recruitment, how long until it visibly raises the survival probability (here: doubles it), and does that time depend on whether the extra recruitment is long-range (a "see also f. 12v" or distant larvae) or local (next door)?

**Model.** Bond directed percolation (1+1) on a ring of 2^20 sites, p = p_c = 0.6447001. Long-range arm: each active site also activates one uniformly random site with probability phi. Matched local substitute: p = p_c + phi/2 (same added expected offspring). t*(phi) = first time P(t)/P0(t) >= 2; a = minus slope of log t* against log phi. See `protocol.md` (frozen before the run), `discovery.md` (associative exploration), `sim.c`, `gate.py`, `analyze.py`.

**Result (held-out seeds, 11 configurations, 30,000-40,000 runs each): all frozen claims hold.**

| Claim | Frozen criterion | Observed |
|---|---|---|
| G independent implementation agrees | all \|z\| < 4 | worst 2.56 (dev seeds, 24 comparisons) |
| H1 baseline exponent | delta in [0.14, 0.18] | 0.1617 [0.157, 0.167] (theory 0.1595) |
| H2 long-range rescue scaling | a_LR in [0.60, 0.90], >= 4 points | 0.834 [0.801, 0.862], 6 points |
| H3 local substitute scaling | a_NN in [1.4, 2.1], >= 3 points | 1.517 [1.469, 1.561], 4 points |
| H4 local vs distant | LR < NN at 0.05 and 0.025; LR >= NN at 0.2 | 645 < 929; 1167 < 2890; 210 >= 123 |

t* (generations): LR at phi = 0.2 / 0.1 / 0.05 / 0.025 / 0.0125 / 0.00625 = 210 / 369 / 645 / 1167 / 2120 / 3734. NN at 0.2 / 0.1 / 0.05 / 0.025 = 123 / 325 / 929 / 2890. The ratio t*_NN / t*_LR is 0.59, 0.88, 1.44, 2.48: local extra recruitment wins when a lot is added, and distant wins increasingly as the amount shrinks (crossover near phi = 0.08). Verdict (registry): CORROBORATED, for this model, window and definition of t*.

**Things that did not go as I expected (observed vs interpretation).**
- *Amplitude* (observed): my heuristic implied t*_LR(0.1) of about 6 generations; development gave 373 and held-out 369. Only exponents were ever claimed, but the heuristic's amplitude was wrong by a factor of 60. Interpretation: requiring a doubling of the survival probability is far stricter than "first long-range event".
- *Exponents* (observed): a_LR = 0.834 is inside the frozen interval but its bootstrap interval excludes both heuristic values (0.76 and 0.68); the local slope between neighbouring phi stays between 0.80 and 0.86 across the whole range. a_NN = 1.517 excludes the asymptotic 1.734. Interpretation (untested): finite-time corrections to scaling and the doubling definition of t*; a different t* threshold or much longer times would be needed to separate them. I have no derivation of 0.83.
- The corroboration is therefore of the *frozen qualitative-plus-bracketed claims*, not of my heuristic exponent.

**Limits.** Finite model (one dimension, one seed site, cap rule at 2000 active sites applied identically to all runs, doubling threshold, window up to t = 6000). Bootstrap intervals capture sampling noise over 20 batches only, not systematic error. This is a simulation of a mechanism under its own assumptions: it does not show that manuscript annotation, ant trails or reef recruitment behave this way. The known result that shortcuts change the DP universality class (mean-field on small worlds, arXiv 1307.6186, snippet-level reading) is background, not claimed. Real ant recruitment is nonlinear (a quorum/Allee effect), which this model does not have; that is the analogy's weak point and the basis of the follow-up.

**Association status.** Generative: the three source domains produced the question and the matched local control. The mathematics would be identical if the topic had been epidemics. No claim that the association improved the result.

**Process disclosures.** Development pilot ran before registration (during elaboration) and is disclosed in `protocol.md`; freeze committed (`363cdbb7b`) and verified successful before the held-out run started (no chained commit this time); held-out outputs unchanged after analysis (`heldout_raw_sha256.json`). Held-out run took 20 min 8 s wall (5 parallel processes, 2761 s user CPU) against my 10-15 min estimate. One cross-family review (`review.md`, 0.00084 USD). Parent token usage unknown.

**Files.** `discovery.md`, `protocol.md`, `freeze.json`, `sim.c`, `gate.py`, `analyze.py`, `run_dev.sh`, `run_heldout.sh`, `dev/`, `heldout/` (raw counts), `heldout_result.json`, `heldout_raw_sha256.json`, `review.md`, `review_openrouter_raw.md`.
