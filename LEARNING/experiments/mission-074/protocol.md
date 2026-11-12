# MISSION-074 protocol (written before the held-out run)

Question: random Truchet quarter-arc tilings on a torus; do their loops show the 7/4 hull dimension of critical 2D percolation, and which biased tilings stay critical?

Model: L x L torus; each tile has orientation A (arcs N-W and S-E) or B (arcs N-E and S-W). Loop length = number of arcs. Uniform bias: P(A)=p everywhere. Checker bias: P(A)=p on even i+j, 1-p on odd.

## Instruments (validated on development inputs)
- `tilings.c`: union-find on the 2L^2 edge midpoints; one JSON line per sample (longest loop, loop count, sum, counts of loops >= 128..4096).
- `gate.py`: independent walker on explicit midpoint coordinates; compares sorted loop lengths with the C output for 200 tilings (L 4..12, both biases, p in {0,.2,.35,.5,.8,1}); `dev_gate_result.json`: 200/200 agree, sum rule holds.
- `analyze.py`: slopes by ordinary least squares in log-log, 95% bootstrap intervals (2000 resamples, seed 74). Verdicts use the point estimate against the fixed interval; the interval is reported, not used.

## Hypotheses and fixed criteria
See MISSION-074 manifest. H1: slope of mean longest loop over L=64..1024, p=0.5, in [1.72,1.78]. H2: alpha of N(>=s) at L=1024, s=128..4096, in [1.08,1.20]. H3a: uniform p=0.35, longest-loop slope in [1.70,1.80]. H3b: checker 0.35, slope < 1.2. Sum rule (sum of loop lengths = 2L^2) and monotone counts must hold in every row.
Sample sizes (`run_heldout.sh`): 20000/10000/4000/2000/1000 per size L=64..1024, seeds 7401/7402/7403, per bias setting.

## Forecasts (before the held-out run)
H1 0.80, H2 0.60, H3a 0.75, H3b 0.90.

## Recorded development miss (honest record)
The manifest first registered H3 as "uniform p=0.35 gives slope < 1.2, forecast 0.9". That premise was my error: I assumed any bias away from 1/2 destroys criticality. A 20-sample-per-size development run (dev/, seeds 11-13) gave slope 1.76 for uniform 0.35 and local slopes 0.17-0.43 for checker 0.35. The reason is structural: orientation A on even tiles and B on odd tiles are the same bond state, so uniform bias is horizontal-bond probability p against vertical 1-p, which sums to 1 (the self-dual critical line), while checker bias makes both bonds open with probability p. H3 was replaced by H3a and H3b before the freeze. I had also seen dev-sized estimates for H1 (1.746) and H2 (1.122) when choosing intervals; the intervals were set from theory (7/4, 8/7) with finite-size slack, not tuned to them, but that is a disclosed exposure. No held-out data existed at that point.

## Budget and stop conditions
Local CPU only, single process, ceiling 600 s wall and 1 GiB RAM (expected well under 2 min). Optional review via OpenRouter capped at 0.05 USD after freeze. Stop and record UNASSESSED if the gate or sum rule fails or the ceiling is hit; no silent shrinking of sample sizes.
