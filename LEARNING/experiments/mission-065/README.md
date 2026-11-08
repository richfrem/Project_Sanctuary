# MISSION-065 Results: is the golden angle the finite-N optimum of a Vogel spiral?

**Verdict: FALSIFIED** (gate G passed; H1 failed; H2 and H3 held). Frozen commit `3044ab5a8` came before the held-out run. Runtime: 111.3 s CPU, standard library only, no model calls.

| N | golden d_min | window-best angle | offset | best d_min | ratio | full-range best in window |
|---|---|---|---|---|---|---|
| 100 | 1.60195 | 137.58153° | +0.0738° | 1.60578 | 1.00239 | yes (60,319 evals) |
| 200 | 1.60195 | 137.52306° | +0.0153° | 1.60274 | 1.00050 | yes (170,607 evals) |
| 400 | 1.60195 | 137.51965° | +0.0119° | 1.60257 | 1.00038 | n/a |
| 800 | 1.60195 | 137.51965° | +0.0119° | 1.60257 | 1.00038 | n/a |

- **H1 failed.** Nearby angles do beat golden, but by at most 0.24%, well short of the 1% I predicted. The margin shrinks as N grows.
- **H2 held.** Every best angle lies within 0.074° of golden, offset on the same (+) side.
- **H3 held.** Across the whole range [0.02, 0.5], the best angle for N = 100 and 200 lies inside the golden window. No distant angle competes.

## Descriptive follow-up (not frozen, run after the verdict)
- At the golden angle, the closest pair is **seeds 1 and 4**, at every N. Golden d_min is constant (1.6019502) from N = 25 to N = 800 because the irregular inner core sets it.
- At the best angle for N = 800 (137.51965°), the closest pair moves out to **seeds 208 and 242**. The tiny angle shift fixes the core and moves the bottleneck outward.
- If the first 20 seeds are left out, the golden spiral's d_min rises to about 1.67 and beats the "better" angle (1.60). That suggests golden is the better choice for the bulk, and the finite-N advantage comes from the core. This figure is heuristic: it only compares neighbours within 60 indices.

## Interpretation and limitations
- On the min-distance measure, the golden angle is almost optimal at finite N: within 0.04 to 0.24% of the best angle found. The leftover gap comes from the core's boundary effect, not from bulk packing.
- This is a grid search, so the best values found are lower bounds on the true optimum. Only one measure was tested (min distance) and one radial law (r = sqrt(k)). Other measures (packing density, Voronoi regularity) or Vogel's sqrt(k + 1/2) variant could rank angles differently.
- The dev run was seen before the freeze (disclosed in protocol.md). The hypotheses predate it.

Files: `protocol.md`, `evaluate_vogel.py`, `freeze.json`, `dev_results.json`, `results.json`, `heldout_log.txt`.
