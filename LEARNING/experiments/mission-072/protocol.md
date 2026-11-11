# MISSION-072 frozen protocol: search, restart and the clock

## Status and motivation

Own-choice inquiry explicitly authorized by Richard after MISSION-062 completed its full committed/Hugging Face sequence. Registered PROPOSED in commit 4c45ba8eb and PR530. No held-out L>=9 mean, moment or reset outcome was computed before this freeze. Development L1..8 passed 216 exact reset-mean comparisons, exact baseline moments, a closed-form baseline identity and one-step timing controls. The pre-execution forecast is roughly 70% that the finite classifier will be exact; this is a subjective prediction, not scientific evidence.

## Precise stochastic process

Transient sites are 1..L, with target site0 absorbing. From i<L, an ordinary step goes to i-1 or i+1 with equal probabilities; from L it goes to L-1 with probability1. Start/restart location is x. At each event, with probability r the walker resets to x, spends **total duration c**, and takes no ordinary step. With probability 1-r it takes one ordinary step and spends one time unit. Absorption ends the process. Costs may be rational expected durations. r is strictly below1; no claim concerns the degenerate r=1 process. c=0 means an instantaneous reset, c=1 a timed reset occupying a full step, and c>1 a longer reset. An extra reset delay d would mean c=1+d; these names are not interchangeable.

## Derivation of the local predictor

Let P be the substochastic transient walk matrix, mu=E[T] and s=E[T^2] without restart. The baseline equations are (I-P)mu=1 and (I-P)s=2mu-1. The known walk identity mu_x=x(2L-x) also follows directly from the interior second difference and reflecting boundary equation; it is independently checked.

For u=(I-(1-r)P)^-1 1, the reset mean obeys m_i=(1-r)+r*c+(1-r)sum_j Pij*m_j+r*m_x. Solving this scalar renewal coupling gives

    m_x(r,c) = [1+r(c-1)] u_x / [1-r u_x].

Expansion at r=0 gives

    D(c) = d m_x/dr at zero = mu_x^2 - (s_x-mu_x)/2 + (c-1)mu_x.

Thus a negative derivative is equivalent to Var(T)>mu^2+(2c-1)mu. Define c_star=1+(s-mu)/(2mu)-mu; D<0 iff c<c_star. Equality is not a strict local improvement. This derivation supplies only a **local** condition; the claim that it classifies existence of improvement over the frozen finite grid is a separate empirical hypothesis.

Related primary context: Bonomo/Pal, https://arxiv.org/pdf/2102.00895 (PDF SHA-256 fea03002c31b65cc549fc8bf3f3568cdc594e62064bd9a8b67cbb802d1e72715), Eq24–25, derives a sufficient infinitesimal criterion for its own timing convention and a two-absorbing-boundary geometry. c=0 recovers its CV²>1−1/mu expression; c=1 yields CV²>1+1/mu in the explicitly specified model here. The paper is context, not a claim that the stronger global classifier is already proved. General renewal context: https://arxiv.org/pdf/1607.06048. No novelty or independent Lean/formal-proof claim is made.

## Frozen finite hypothesis and domain

For every integer L=9..32, every x=1..L, and every unique cost in

    C(L,x)={0,1,2,4,8,max(c_star-1/2,0),max(c_star,0),max(c_star+1/2,0)},

the local prediction D(c)<0 is equivalent to **at least one strict improvement** m_x(r,c)<mu_x on

    R={1/2^j : j=1..24} union {3/4,7/8}.

There are26 distinct exact rates per case. Costs are constructed solely from baseline moments according to this fixed rule, not fitted to reset outcomes. No ordering, L bound, cost offset, rate resolution, equality rule or hypothesis is changed after held-out observations. The claim is about this grid, not every real rate, all L, arbitrary graphs, continuous diffusion or actual search algorithms.

## Independent instruments and validity

`discounted.py` uses exact Fraction tridiagonal elimination to compute discounted survival and a scalar renewal formula. `direct.py` builds the reset transition matrix explicitly and solves dense rational systems using pivoted Gauss-Jordan elimination; it does not import the discounted solver or use its renewal formula. Both were written by the same parent agent, so algorithmic independence is claimed, not independent authorship. Baseline identities and deterministic one-step controls reduce shared-model risk.

Before primary scoring, compare baseline first/second moments at L=9,16,24,32 and 48 exact reset means: those L, starts1/ceil(L/2)/L, costs0/2 and rates1/2/1/16. Any disagreement invalidates the instrument and yields UNASSESSED. No floating-point tolerances or simulation estimates enter any decision.

## Execution, coverage and stopping

Commit protocol.md, freeze.json and hashed source artifacts in a **separate command** before running evaluate.py. Verify the committed freeze and every source hash before executing held-out cases. Traverse L ascending, start ascending, unique cost ascending, and rates ascending. Keep full exact raw means in per-L deterministic gzip CSVs (mtime0), plus case-level prediction/observed/best-grid summaries.

If a case disagrees, recompute **all26 rates for that case** using the independent dense solver. A direct/fast mean mismatch is UNASSESSED, not theoretical falsification. If the direct vector agrees and the classifier disagrees with the observed grid, record FALSIFIED and stop primary execution; no later case rescues the hypothesis. If every registered case is covered and classified correctly, record CORROBORATED only for this domain. Missing coverage or budget exhaustion is UNASSESSED. Reconstruct the complete baseline-derived case inventory independently of success counts to verify coverage. All raw means are exact rationals; float ratios are descriptive only and the reported optimum is only the best sampled rate.

## Resource and publication boundaries

Compute ceiling900 wall-seconds including measured development time,1GiB memory. No network or model calls during the experiment and no paid worker dispatch planned. Parent reasoning/token usage is unattributed and not zero;20k parent tokens is a planning ceiling whose compliance cannot be independently certified from this interleaved session. State that limitation instead of claiming measured savings. Resource reporting uses platform-correct maximum RSS and parent/child CPU usage.

Registration, freeze, results, incidents, lessons, follow-up candidate and final check-in must be integrated before capture. Then use the approved bot's full closeout (export PR, Hugging Face sync and both-hash check, receipt PR, final local main sync). Remove task worktree/branch only after verifying retained changes and full preservation. Report both missions between19:00:45 and20:00:45 UTC, measured from Richard's instruction; do not launch another mission to fill waiting time.
