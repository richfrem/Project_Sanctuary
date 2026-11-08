# MISSION-065 Frozen Protocol: Golden-angle finite-N optimality

Model: Vogel spiral, point k = 1..N at r = sqrt(k), theta = 2*pi*k*t. Metric: d_min(t, N) = minimum pairwise distance.
t_g = 2 - phi (137.5078 deg).

Held-out N: {100, 200, 400, 800}. Dev N: {25, 50} (instrument checks only).

Search: step = 0.05 / (2*pi*N^1.5) over t_g +/- 1 deg (all N) and over [0.02, 0.5] (N in {100, 200}),
followed by 3 rounds of x10 local refinement (+/-20 steps) around the best found.
This is a grid search: "best" means best found, a lower bound on the true supremum.

- G (gate): grid-hash d_min == brute force (1e-12) on dev configs and on two configs at each held-out N; t = 1/2 gives sqrt(N) - sqrt(N-2).
- H1: window-best d_min >= 1.01 x d_min(t_g) at every held-out N.
- H2: |window-best angle - t_g| <= 0.1 deg at every held-out N.
- H3: the full-range best for N in {100, 200} lies within the +/-1 deg window.

Verdict: UNASSESSED if G fails or the run does not finish; CORROBORATED if H1, H2 and H3 all hold; otherwise FALSIFIED.
Disclosure: the dev run (dev_results.json) was observed before this freeze; the hypotheses predate it and were not changed.
