# MISSION-066 protocol (frozen before held-out run)

Measure: E(N) = worst absolute error in cents over 3/2, 5/4, 6/5, each rounded independently to the nearest step of N-EDO (patent rounding per interval; not a val-consistent mapping). Record: E(N) < E(M) for all M < N (strict).

- Gate G: N=12 errors 1.955 / 13.686 / 15.641 cents within 0.001; running-min scan equals brute pairwise check on N 1..30.
- H1: 53 and 118 are both records in 31..200.
- H2: every record in 31..200 lies in {31, 34, 53, 65, 87, 118, 171}.
- Verdict: G pass and H1 and H2 -> CORROBORATED; G pass and either fails -> FALSIFIED; G fail -> UNASSESSED.
- Dev: N 1..30 only. Held-out: one run of `evaluate_et.py --mode heldout` after the freeze commit.
- Budget: < 1 min CPU, no model dispatch; stop if claude headroom < 15%. Optional cross-model review skipped unless a result looks anomalous.
- Descriptive only: error in steps (worst_steps), sign of each error.
