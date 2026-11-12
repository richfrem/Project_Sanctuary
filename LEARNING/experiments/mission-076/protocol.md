# MISSION-076 protocol (frozen before the held-out run; see freeze.json)

## Model
Bond directed percolation (1+1), the contact process in discrete generations, on a ring of 2^20 sites. One active site at t=0. Each active site at generation t activates each of its two neighbours at t+1 independently with probability p (nearest-neighbour recruitment). Optional long-range recruitment: with probability phi, one site chosen uniformly on the whole ring is also activated ("cross-reference" / "larval dispersal"). Absorbing state: no active site. A run is called established (alive at every later checkpoint) once 2000 sites are active; this rule is the same for every configuration.
Interpretation (analogy, not evidence): a note in one copy prompts notes in adjacent positions of the next copy; a cross-reference prompts a note somewhere else on the page. p_c = 0.6447001 is the known critical value of bond DP on the 1+1 square lattice (sourced from literature as background, not verified here; development confirms the exponent).

## Provenance and pre-registered expectations
Theory background (transferred, not verified): 1D DP exponents delta = 0.1595, theta = 0.3137 (mean active number of all runs ~ t^theta), nu_parallel = 1.734. Heuristic for the long-range channel (mine, introduced for this inquiry): a long-range recruit rescues a dying cluster once the expected number of long-range events phi * integral N dt reaches order 1, giving t*_LR ~ phi^(-1/(1+theta)) = phi^-0.76 (phi^-0.68 if only surviving clusters are counted). A nearest-neighbour increase Delta = phi/2 of p is a relevant perturbation, t*_NN ~ Delta^(-nu_parallel) = Delta^-1.73. No amplitude is predicted.

## Quantities
P(t): fraction of runs alive at t (checkpoints on a 1.12-ratio grid up to 6000). P0(t): the same at p = p_c, phi = 0 (baseline).
t*(config): first time at which P(t)/P0(t) >= 2 (log-log interpolation between checkpoints). A configuration with no crossing by t = 6000 is censored.
a_LR, a_NN: minus the OLS slope of log t* on log phi over uncensored phi.
Confidence intervals: bootstrap over the 20 batches (300 resamples, seed 76).

## Held-out configurations (seed base 7600, disjoint from development seed base 100)
- baseline: p_c, phi = 0, 20 x 2000 runs
- long-range (LR): p = p_c, phi in {0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625}, 20 x 1500 runs each
- nearest-neighbour matched substitute (NN): p = p_c + phi/2 (same added expected offspring), no long-range links, phi in {0.2, 0.1, 0.05, 0.025}, 20 x 1500 runs each

## Hypotheses (finite claims about this model and window)
- **G (instrument gate, development):** sim.c agrees with the independent pure-Python implementation (gate.py): |z| < 4 for every checkpoint comparison in the four configurations.
- **H1 (baseline):** the held-out baseline survival exponent delta, fitted on t in [100, 1000], is in [0.14, 0.18].
- **H2 (long-range rescue scaling):** a_LR is in [0.60, 0.90] (the two heuristic values 0.68 and 0.76 lie inside), with at least four uncensored phi.
- **H3 (matched substitute):** a_NN is in [1.4, 2.1] with at least three uncensored phi. This is a check of the instrument against the sourced exponent nu_parallel.
- **H4 (local versus distant):** t*_LR < t*_NN at phi = 0.05 and at phi = 0.025 (distant recruitment rescues sooner when little is added), while t*_LR >= t*_NN at phi = 0.2 (the ordering reverses when a lot is added). A censored t* is treated as greater than 6000; if both members of a pair are censored the pair is undecided and H4 is then UNASSESSED for that pair.

## Forecasts (recorded before the held-out run; my subjective probabilities)
G 0.95 (already observed on dev seeds, worst |z| = 2.56); H1 0.9; H2 0.55 (dev gave 0.82 on only three points); H3 0.55; H4 0.5 (dev showed LR 372 vs NN 312 at 0.1 and LR 986 vs NN 2152 at 0.03).

## Verdict rules
UNASSESSED (verdict null) if the gate fails, a run does not finish, or censoring leaves fewer than four LR or three NN points: instrument failure or an under-resolved design is not a finding. FALSIFIED if the gate holds and any of H1-H4 fails; the measured values with bootstrap intervals are then the result. CORROBORATED only if G, H1, H2, H3 and H4 all hold. Corroboration is limited to this model, ring size, cap rule, definition of t* and fit window; it does not establish the mechanism in ants, manuscripts or reefs, and it does not establish asymptotic exponents.

## Development disclosures (before freeze)
- The first design had no development pilot at registration; the pilot was run during elaboration with dev seeds (base 100, tmax 3000, 20 x 2000 runs, phi in {0.1,0.03,0.01,0.003,0.001} for LR and {0.1,0.03,0.01} for NN). Its outputs are in dev/ and hashed in freeze.json.
- Dev result: baseline delta = 0.159 [0.154, 0.163]; a_LR = 0.82 from three uncensored points (0.01, 0.03, 0.1); NN censored at 0.01.
- **Premise miss:** my heuristic implied t*_LR(0.1) of about 6 generations. The pilot gave 373. The amplitude was wrong by a factor of 60 (the doubling criterion and the survivor requirement are much stricter than "first long-range event"); only the exponent is claimed.
- The H4 reversal at phi = 0.2 and the intervals in H2/H3 were chosen after seeing development outcomes; the held-out seeds are disjoint.
- The development runs took 9.7 minutes of wall time on one core for nine configurations; the held-out grid was sized from that.
