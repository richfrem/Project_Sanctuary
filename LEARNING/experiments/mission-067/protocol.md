# MISSION-067 protocol (frozen before held-out run)

- Chain C: each step applies a uniformly random nonempty ringing change (disjoint adjacent swaps; F(n+1)-1 choices). Chain A: lazy adjacent transposition (stay 1/2). Start: rounds (identity). Exact distribution evolution on S_n.
- t_X(n): first t with TV to uniform <= 0.25; cap 5000 steps.
- Gate G: F(n+1)-1 move count; mass drift < 1e-9; chain A TV non-increasing; all runs finish under the cap.
- H1: R(n) = t_A/t_C satisfies R(7) > R(6), R(8) > R(6), R(8) > R(7).
- H2: t_C*s_C < t_A*0.5 at n = 7 and 8 (fewer total expected swaps).
- Verdict: G and H1 and H2 -> CORROBORATED; G and not both -> FALSIFIED; G fails -> UNASSESSED.
- Dev n in {4,5,6}. Held-out: one run `--mode heldout` (computes n = 6,7,8; n = 6 is dev, reused as the H1 reference).
- Budget: 20 min CPU, no model dispatch, no independent review planned (deterministic; gate checks). Stop if claude headroom < 15%.
- Predictions (subjective): H1 70%, H2 40%. Chain C is aperiodic for n >= 3, since changes have 1 or 2 swaps; laziness is applied only to A, which is periodic otherwise. This asymmetry is part of the design and is disclosed.

## Dev disclosure (added before freeze, after dev run; hypotheses unchanged)
Dev: R = 2.20, 2.50, 2.88 at n = 4, 5, 6. swaps_C/swaps_A = 1.14, 1.14, 1.16, i.e. in dev chain C uses *more* swaps, against the direction of H2. H2 is kept as written.
