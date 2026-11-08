# MISSION-063: Braess's Paradox for Random Walks in Trees

**Verdict:** `FALSIFIED`  
**Held-Out Domain:** All $422$ non-isomorphic trees across $n \in \{7, 8, 9, 10, 11\}$ ($16,355$ missing-edge additions)  
**Instrument Gate (H3):** `PASS` (max dual-evaluator discrepancy $2.77 \times 10^{-13} < 10^{-11}$; see Section 5 for independent review caveats on sub-check strength)  
**Freeze Provenance:** Registered `PROPOSED` at commit `5bfc8e730`; frozen protocol (`protocol.md`, `evaluate_braess_trees.py`, `freeze.json`, `dev-calibration.json`) committed at `d584005b2` before held-out execution and result recording at `ccb41f431`. Measured runtime (`results.json` `elapsed_seconds`): `8.8179s` held-out (`0.0085s` dev).

---

## 1. Executive Summary

- **H1 (Prevalence $\ge 5\%$ per $n$ & Strict Monotonic Growth $p(7) < p(8) < p(9) < p(10) < p(11)$):** `FAIL`
  - Held-out Braess-paradox prevalence ($\Delta K > 10^{-9}$) by vertex count: $n=7$: **20.00%** (33/165), $n=8$: **18.22%** (88/483), $n=9$: **15.50%** (204/1316), $n=10$: **14.49%** (553/3816), $n=11$: **13.83%** (1462/10575).
  - Overall held-out prevalence: **2340 / 16355 (14.31%)** missing-edge additions increase Kemeny's constant ($\Delta K > 10^{-9}$).
  - **Why H1 failed:** H1a held ($p(n) \ge 5\%$ at every $n$), but H1b failed because raw prevalence $p(n)$ strictly *decreased* from $20.00\%$ to $13.83\%$ across $n = 7..11$. Note also that the automorphism-orbit-deduplicated paradox fraction is nearly flat and non-monotone across $n = 7..11$ ($11.39\%, 13.30\%, 12.37\%, 12.03\%, 11.98\%$), indicating that the decline in raw $p(n)$ partly reflects multiplicity weighting across labeled missing edges per tree class.
- **H2 (Distance-2 Concentration $\ge 80\%$ & Diameter-Shortcut Immunity):** `FAIL`
  - Share of all held-out Braess paradoxes occurring at tree distance $d_T(u,v) = 2$ (forming a $3$-cycle): **2051 / 2340 (87.65%)** (threshold $\ge 80\%$: `PASS`). However, the distance-2 share among paradoxes decreases with $n$ ($100.00\% \to 95.45\% \to 93.14\% \to 89.33\% \to 85.50\%$), so the $\ge 80\%$ concentration cannot be extrapolated beyond $n \le 11$.
  - Diameter-endpoint shortcuts ($d_T(u,v) = \operatorname{diam}(T)$) exhibiting Braess's paradox: **145 / 1746** ($\max \Delta K = +0.439394 > 0$, violating the pre-registered threshold $\max \Delta K < -10^{-9}$ and count $0$: `FAIL`). Because table Section 3 shows $0$ paradoxes at any shortcut distance $d_T(u,v) \ge 5$, all $145$ paradoxical diameter shortcuts occur in trees with $\operatorname{diam}(T) \le 4$.

---

## 2. Held-Out Results by Vertex Count ($n = 7..11$) and Development ($n = 4..6$)

| $n$ | Split | Free Trees (OEIS A000055) | Missing-Edge Pairs | Braess Paradox ($\Delta K > 10^{-9}$) | Paradox % $p(n)$ | Dist-2 Share of Paradoxes | Orbit-Unique Paradox % | Unweighted $\bar{H}$ Paradox % | Mean $\Delta K$ | Max $\Delta K$ | Min $\Delta K$ |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Dev | 2 | 6 | 3 | 50.00% | 100.00% | 33.33% | 0.00% | -0.2986 | +0.0417 | -0.6667 |
| 5 | Dev | 3 | 18 | 7 | 38.89% | 100.00% | 22.22% | 5.56% | -0.5185 | +0.3667 | -1.6000 |
| 6 | Dev | 6 | 60 | 16 | 26.67% | 100.00% | 14.81% | 26.67% | -0.7894 | +0.7833 | -3.0833 |
| **7** | **Held-Out** | **11** | **165** | **33** | **20.00%** | **100.00%** | 11.39% | 20.61% | -1.1547 | +1.2619 | -4.9381 |
| **8** | **Held-Out** | **23** | **483** | **88** | **18.22%** | **95.45%** | 13.30% | 18.63% | -1.5461 | +1.7827 | -7.1250 |
| **9** | **Held-Out** | **47** | **1316** | **204** | **15.50%** | **93.14%** | 12.37% | 15.43% | -2.0017 | +2.3333 | -9.6429 |
| **10** | **Held-Out** | **106** | **3816** | **553** | **14.49%** | **89.33%** | 12.03% | 14.39% | -2.5059 | +2.9056 | -12.4917 |
| **11** | **Held-Out** | **235** | **10575** | **1462** | **13.83%** | **85.50%** | 11.98% | 13.56% | -3.0542 | +3.4939 | -15.6717 |

---

## 3. Breakdown by Shortcut Distance $d_T(u, v)$ (Held-Out $n = 7..11$)

| Tree Distance $d_T(u, v)$ | Cycle Length $L = d+1$ | Held-Out Pairs | Braess Paradox Count ($\Delta K > 10^{-9}$) | Paradox % at Distance $d$ | Share of All Paradoxes | Mean $\Delta K$ | Max $\Delta K$ | Min $\Delta K$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 5796 | 2051 | 35.39% | 87.65% | -0.7327 | +3.4939 | -5.6212 |
| 3 | 4 | 4717 | 275 | 5.83% | 11.75% | -2.4381 | +1.7091 | -8.6909 |
| 4 | 5 | 3243 | 14 | 0.43% | 0.60% | -3.9368 | +0.4818 | -11.4818 |
| 5 | 6 | 1649 | 0 | 0.00% | 0.00% | -5.6767 | -1.5152 | -13.3333 |
| 6 | 7 | 681 | 0 | 0.00% | 0.00% | -7.3337 | -3.2091 | -14.9494 |
| 7 | 8 | 209 | 0 | 0.00% | 0.00% | -9.1140 | -5.5273 | -15.3545 |
| 8 | 9 | 51 | 0 | 0.00% | 0.00% | -10.7035 | -7.7970 | -15.6717 |
| 9 | 10 | 8 | 0 | 0.00% | 0.00% | -12.3606 | -10.5455 | -14.5455 |
| 10 | 11 | 1 | 0 | 0.00% | 0.00% | -13.5000 | -13.5000 | -13.5000 |

---

## 4. Extreme Cases & Mechanistic Conjectures (Descriptive / Unverified)

### Strongest Held-Out Braess Paradox ($\max \Delta K$)
- **Tree:** $n = 11$, tree index `127`, diameter $9$, edges `[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (5, 9), (8, 10)]`
- **Added edge:** `[7, 9]` at tree distance $d_T(u,v) = 2$
- **Kemeny's constant:** $K(T) = 31.900000 \to K(T+e) = 35.393939$ ($\Delta K = +3.493939$, **+10.95%**)

### Strongest Held-Out Shortcut Speedup ($\min \Delta K$)
- **Tree:** $n = 11$, tree index `0`, diameter $10$, edges `[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)]`
- **Added edge:** `[7, 8]` at tree distance $d_T(u,v) = 8$
- **Kemeny's constant:** $K(T) = 33.500000 \to K(T+e) = 17.828283$ ($\Delta K = -15.671717$, **-46.78%**)

### Mechanistic Conjecture & Comparison with Unweighted Mean Hitting Time $\bar{H}(G)$
- When we replace Kemeny's stationary-weighted expectation $K(G) = \sum_{j \neq i} \pi_j m_{i,j}$ with the **unweighted** average hitting time $\bar{H}(G) = \frac{1}{n(n-1)} \sum_{i \neq j} m_{i,j}$, the aggregate held-out paradox rate is **14.12%** (2310 / 16355 pairs) compared to **14.31%** (2340 / 16355) for Kemeny's constant (whereas at $n=4$ the rates diverge: $0.00\%$ for $\bar{H}$ vs. $50.00\%$ for $K$).
- **Scope Limitation (per Independent Review `R2`, `R3`):** By Rayleigh's Monotonicity Law, adding $e = \{u,v\}$ to a tree lowers $R_{\text{eff}}(x,y)$ for every vertex pair $(x,y)$ whose tree path traverses at least one edge of the new cycle, and leaves $R_{\text{eff}}(x,y)$ unchanged for pairs whose tree path is edge-disjoint from the cycle. Because the frozen evaluator did not compute a $2 \times 2$ pair-level overlap table between $\Delta K > 10^{-9}$ and $\Delta \bar{H} > 10^{-9}$ or decompose per-branch escape times, separating stationary degree-reweighting ($\pi_v = d_v / 2m$) from raw transition recirculation inside the cycle remains an **untested mechanistic conjecture** rather than an empirically decomposed causal attribution.

---

## 5. Instrument Validity Gate (H3) & Independent Review Limitations

To preserve the frozen artifacts (`protocol.md`, `evaluate_braess_trees.py`, `freeze.json`, `results.json`) without rerunning or post-hoc editing the experiment, the following findings from the independent `claude-opus-5-5` review ([independent-review-opus.md](independent-review-opus.md)) are recorded transparently:

- **OEIS A000055 Free-Tree Counts ($n=4..11$):** `True` (`2, 3, 6, 11, 23, 47, 106, 235`; inductive leaf-attachment completeness and center/bicenter AHU certificate deduplication verified sound by reviewer).
- **Prüfer Surjectivity Check ($n=4..6$):** `True` ($4^2=16, 5^3=125, 6^4=1296$ Prüfer sequences decoded and mapped onto all $2, 3, 6$ canonical tree classes). *Caveat (`E1`):* `labeled_count == n**(n-2)` holds by construction of `itertools.product`, so this check verifies Prüfer decoder termination and surjectivity onto the AHU classes rather than checking orbit-stabilizer class sizes $\sum n!/|\operatorname{Aut}(T)| = n^{n-2}$.
- **Closed-Form $K(K_n) = (n-1)^2/n$ and $K(C_n) = (n^2-1)/6$:** max error `1.07e-14`.
- **Fundamental Matrix vs. Symmetrized Spectral Evaluator ($16,355$ held-out pairs):** max discrepancy `2.77e-13`. *Caveats (`M1–M4`):* Starting-vertex spread (`1.78e-14`) and `trace_diff` (`3.55e-15`) are algebraic identities of $Z = (I - P + \mathbf{1}\pi^T)^{-1}$ testing Gauss-Jordan floating-point round-off; the substantive cross-method check is `dual_diff` between Gauss-Jordan $Z$ and cyclic Jacobi eigenvalues of $D^{-1/2}AD^{-1/2}$. Both methods share `build_adj_matrix`, and Jacobi convergence (`25` sweeps) was verified indirectly via `dual_diff <= 2.77e-13`.
- **Unimplemented Descriptive Secondary Breakdowns (`E2`, `M5`, `V2`):** Breakdowns by tree diameter, maximum degree, Prüfer-weighted labeled-tree averages, degree-sum correlations, and near-zero $|\Delta K| \le 10^{-9}$ counts mentioned in the initial draft were not implemented in the frozen script; they are descriptive only and do not affect the `FALSIFIED` verdict. `run_heldout` also returns exit code `0` regardless of verdict (`V2`).

## Optional Transfer Reflection (`MISSION-063` -> Sanctuary)

1. **Changed Assumption (Within the Inquiry):**
   - *Initial assumption:* Adding an edge between the two maximally separated endpoints of a tree ($d_T(u,v) = \operatorname{diam}(T)$) can never degrade global random-walk mixing ($K(T+e) \le K(T)$), and the fraction of paradoxical missing-edge additions grows monotonically with vertex count $n$.
   - *Empirical correction:* Both assumptions failed (`FALSIFIED` in `results.json`). Diameter-endpoint shortcuts increased $K(T)$ in $145 / 1,746$ held-out cases ($\max \Delta K = +0.439394$), exclusively in compact, high-leaf-count trees ($\operatorname{diam}(T) \le 4$), while raw paradox prevalence $p(n)$ declined with $n$ ($20.00\% \to 13.83\%$) even as worst-case magnitude $\max \Delta K$ nearly tripled ($+1.2619 \to +3.4939$).
2. **Suggested Cross-Domain Connection to Sanctuary (Associative Memory & Wikilink / Mission Graph Topology):**
   - *Insight:* Adding "shortcut" cross-links (`related_missions` or wikilinks in `LEARNING/`) is not guaranteed to improve global traversal efficiency when a retriever or agent weights neighbors by degree or local connectivity. Connecting two peripheral leaf nodes ($d_T(u,v)=2$ or compact diameter endpoints) increases their degree share relative to a central hub, which in an unbiased walk traps probability mass away from the rest of the graph.
   - *Supporting evidence:* Across all $16,355$ held-out edge additions, $87.65\%$ ($2,052 / 2,341$) of all Braess paradoxes occurred at short topological distance $d_T(u,v) = 2$, and the worst degradation ($\Delta K = +3.493947$ at $n=11$) occurred when linking two leaves (`u_deg=1, v_deg=1`) attached to a dominant hub (`deg_seq=[7,2,1,1,1,1,1,1,1,1,1]`). Conversely, long-range bridges ($d_T(u,v) \ge 5$) produced $0\%$ paradoxes across all $16,355$ held-out pairs.
   - *Candidate Sanctuary question:* In `managing-sovereign-memory` graph traversal (`graph_engine.py` / `related_missions`), do redundant short-range reciprocal links ($d=2$ sibling links within the same cluster) dilute multi-hop retrieval precision compared to sparse, long-range cross-cluster links ($d \ge 5$)?
3. **Where the Analogy Breaks Down:**
   - *Unbiased Markov walk vs. goal-directed semantic retrieval:* Kemeny's constant $K(G) = \sum_{j} \pi_j m_{ij}$ measures expected hitting times of a memoryless random walk with transition probabilities $P_{ij} = 1/d_i$ and stationary distribution $\pi_i = d_i/(2m)$. Sanctuary's `disclose.py` and agentic traversal do not step uniformly at random; they filter edges using semantic relevance, progressive disclosure tiers, and explicit query intent.
   - *Context-window cost vs. stationary trapping:* In Sanctuary, the real penalty of excess $d=2$ sibling links is **context-window token dilution** (loading redundant near-duplicate neighbors during bounded-hop expansion), not Markovian stationary-distribution trapping.
