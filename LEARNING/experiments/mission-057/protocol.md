# MISSION-057 frozen protocol

Question (my own choice): do consecutive primes remember their last digit - is the next prime less likely to end in the same digit than a random "prime-like" model predicts - and does that fade as primes grow? (Lemke Oliver and Soundararajan, 2016, reported this repulsion; I measure its size per digit, its scale dependence, and compare against the right null.)

## Data and model
All primes <= 10^8 (numpy sieve, instrument-checked against pi(10^6)=78498, pi(10^7)=664579, pi(10^8)=5761455 and trial division). Windows: DEV [10^4,10^5] (debugging only), SMALL [10^5,10^6), LARGE [10^7,10^8). Statistic: for each last digit d in {1,3,7,9}, the fraction of consecutive primes (both inside the window) whose next prime also ends in d.
Control (Cramer): every integer in the window coprime to 10 is included independently with probability 2.5/ln n (the prime density among those integers). Replicates: 20 (SMALL), 5 (LARGE), seeds 101..; control value = mean over replicates. Analytic control value: with inclusion probability q and candidates visited in the cyclic order 1,3,7,9, P(same digit next) = q(1-q)^3/(1-(1-q)^4); it approaches 0.25 only as q->0.

## Claims and decision (frozen in run.py)
R_d = control(d->d) - real(d->d).  H1: R_d >= 0.005 for every d in LARGE.  H2: R_d(SMALL) > R_d(LARGE) for every d.  H3 (validity): |simulated control - analytic| < 0.003 for every d in both windows.
Verdict: UNASSESSED if H3 fails or any check fails; else NOT FALSIFIED if H1 and H2 hold; else FALSIFIED. Secondary (cannot rescue): full 4x4 transition matrices, which digits repel most, ratio to the naive 0.25.

## Development disclosure (what changed before freezing)
The first draft used 0.25 as the null and required "deficit >= 3 pp" and "control within 0.5 pp of 0.25". The DEV run showed the random control itself gives about 0.16 (the waiting time to the next included candidate makes same-digit returns rarer), while real primes gave 0.13-0.15. That refuted the draft's null, so the comparison was redefined to the control and its analytic value, with thresholds set once, before any held-out number: 0.5 pp for H1 (the dev excess ranged 0.75-2.7 pp at a SMALLER scale, so H1 at LARGE is genuinely uncertain) and 0.3 pp for H3 (dev agreement was 0.25 pp). The held-out windows were not computed before the freeze.

## Budget
Model tokens <= 20,000 additional. Compute <= 15 minutes (dev: 0.3 s; the full sieve to 10^8 takes seconds).
