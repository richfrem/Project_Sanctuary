# MISSION-063 Frozen Experimental Protocol

**Mission ID:** `MISSION-063`
**Title:** Braess's Paradox for Random Walks in Trees: Does Adding a Shortcut Edge Increase Kemeny's Constant, and Is the Paradox Concentrated at Distance 2?
**Author:** `antigravity`
**Date Frozen:** `2026-11-08`

---

## 1. Definitions & Mathematical Model

Let $G = (V, E)$ be a simple, connected, undirected, unweighted graph on $n = |V|$ vertices with $m = |E|$ edges and degrees $d_v = \deg_G(v)$.
The simple random walk on $G$ has transition probabilities $P_{u,v} = A_{u,v} / d_u$ and unique stationary distribution $\pi_v = d_v / (2m)$.

**Kemeny's constant** $K(G)$ is defined as the expected hitting time from any starting vertex $i \in V$ to a target vertex $j$ drawn from $\pi$:
$$K(G) = \sum_{j \in V \setminus \{i\}} \pi_j m_{i,j}$$
where $m_{i,j}$ is the mean first passage time from $i$ to $j$.

For any non-isomorphic free tree $T$ on $n$ vertices ($m = n-1$) and any missing edge $e = \{u, v\} \notin E(T)$ (so tree distance $d_T(u, v) \ge 2$), adding $e$ produces the unicyclic graph $T + e$ on $n$ vertices and $n$ edges containing a unique cycle of length $L = d_T(u, v) + 1$.

The change in Kemeny's constant is:
$$\Delta K(T, e) = K(T + e) - K(T)$$
A pair $(T, e)$ exhibits **random-walk Braess's paradox** if and only if:
$$\Delta K(T, e) > \varepsilon, \quad \varepsilon = 10^{-9}$$

---

## 2. Development vs. Held-Out Partition

1. **Development Set (`--mode dev`)**:
   - All non-isomorphic free trees on $n \in \{4, 5, 6\}$ vertices ($2 + 3 + 6 = 11$ trees; $2\times 3 + 3\times 6 + 6\times 10 = 84$ missing-edge additions).
   - Closed-form reference families $K_n$ and $C_n$ on $n \in \{4, \dots, 11\}$.
   - Cayley labeled-tree count check on $n \in \{4, 5, 6\}$ via Prüfer bijection vs. AHU canonical tree classes.
   - Used strictly to validate tree generation and linear-algebra evaluators before freezing.
2. **Held-Out Set (`--mode heldout`)**:
   - All non-isomorphic free trees on $n \in \{7, 8, 9, 10, 11\}$ vertices ($11 + 23 + 47 + 106 + 235 = 422$ trees).
   - Every missing edge $e = \{u, v\} \notin E(T)$ for each held-out tree ($\binom{n-1}{2}$ edges per tree; $165 + 483 + 1316 + 3816 + 10575 = 16,355$ total `(tree, missing_edge)` pairs).
   - Never executed or inspected prior to the freeze commit.

---

## 3. Pre-Registered Primary Hypotheses & Validity Gate

### H3 — Instrument Validity Gate (Must Pass or Outcome is `UNASSESSED`)
- **H3.1 (Tree Enumeration Exactness)**: The canonical AHU center-rooted free-tree generator produces exact OEIS A000055 counts $t_n = (2, 3, 6, 11, 23, 47, 106, 235)$ for $n = 4..11$, with zero duplicate canonical certificates, and matches all $n^{n-2}$ labeled trees via Prüfer sequences on $n = 4, 5, 6$.
- **H3.2 (Starting-Vertex Invariance & Dual-Evaluator Agreement)**: On every evaluated graph $G$ (both $T$ and $T+e$), the fundamental-matrix evaluator $K_{\text{hit}}(G)$ satisfies $\max_i K_i(G) - \min_i K_i(G) < 10^{-11}$, agrees with $\operatorname{Tr}(Z) - 1$ within $10^{-11}$, and agrees with the symmetrized spectral evaluator $K_{\text{spec}}(G) = \sum_{k=2}^n \frac{1}{1 - \lambda_k(D^{-1/2}AD^{-1/2})}$ within $10^{-11}$.
- **H3.3 (Closed-Form Literature Identities)**: For all $n \in \{4, \dots, 11\}$, $|K(K_n) - \frac{(n-1)^2}{n}| < 10^{-11}$ and $|K(C_n) - \frac{n^2-1}{6}| < 10^{-11}$.

### H1 — Prevalence & Strict Monotonic Growth with $n$
For each held-out $n \in \{7, 8, 9, 10, 11\}$, let $p(n)$ be the fraction of all $(T, e)$ pairs on $n$ vertices (counting each non-isomorphic tree $T \in \mathcal{T}_n$ once and all $\binom{n-1}{2}$ missing edges $e \notin E(T)$) such that $\Delta K(T, e) > 10^{-9}$.
- **H1a**: $p(n) \ge 0.05$ for every $n \in \{7, 8, 9, 10, 11\}$.
- **H1b**: $p(7) < p(8) < p(9) < p(10) < p(11)$ (strict monotonic increase across the held-out range).

### H2 — Distance-2 Triangle-Trap Concentration vs. Diameter Shortcuts
Across all held-out $(T, e)$ pairs on $n \in \{7, 8, 9, 10, 11\}$:
- **H2a (Distance-2 Concentration)**: Among all held-out pairs exhibiting Braess's paradox ($\Delta K(T, e) > 10^{-9}$), at least $80\%$ ($0.80$) have tree distance $d_T(u, v) = 2$ (forming a $3$-cycle).
- **H2b (Diameter-Shortcut Immunity)**: Among all held-out pairs that connect diameter endpoints ($d_T(u, v) = \operatorname{diam}(T)$), **zero** pairs exhibit Braess's paradox ($\max_{d_T(u,v) = \operatorname{diam}(T)} \Delta K(T, e) < -10^{-9}$).

---

## 4. Verdict Rules & Secondary Analyses

- **`UNASSESSED`**: If any check in **H3** fails, if `freeze.json` SHA-256 verification fails at held-out runtime, or if execution aborts before covering all $422$ held-out trees ($16,355$ pairs).
- **`CORROBORATED`**: If **H3** passes and **H1a**, **H1b**, **H2a**, and **H2b** all hold on the held-out domain $n \in \{7, 8, 9, 10, 11\}$.
- **`FALSIFIED`**: If **H3** passes and any of **H1a**, **H1b**, **H2a**, or **H2b** fails on the held-out domain.

**Secondary Descriptive Outputs (cannot rescue a failed primary claim)**:
1. Breakdown of Braess-paradox prevalence and mean $\Delta K(T, e)$ by edge distance $d_T(u, v) \in \{2, 3, \dots, 10\}$.
2. Automorphism-orbit-deduplicated missing-edge paradox fractions $p_{\text{orbit}}(n)$ (counting equivalent missing edges under $\operatorname{Aut}(T)$ once).
3. Comparison with unweighted mean hitting time $\bar{H}(G) = \frac{1}{n(n-1)} \sum_{i \neq j} m_{i,j}$ (where stationary degree-weighting $\pi_j$ is removed) to isolate how much of Braess's paradox comes from stationary weight shift vs. raw transition trapping.
4. Extreme examples: the held-out $(T, e)$ pair maximizing $\Delta K(T, e)$ (strongest Braess paradox) and minimizing $\Delta K(T, e)$ (strongest shortcut speedup).
